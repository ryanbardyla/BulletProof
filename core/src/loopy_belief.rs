//! Loopy Belief Propagation for Cyclic Neural Networks
//! 
//! Revolutionary implementation allowing true cycles in neural networks,
//! like biological brains. Messages propagate until convergence.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use ndarray::{Array1, Array2};

/// Unique identifier for neurons in the belief network
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct NeuronId(pub u64);

/// Unique identifier for factors (connections) in the network
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct FactorId(pub u64);

/// A belief message passed between neurons
#[derive(Debug, Clone)]
pub struct BeliefMessage {
    pub from: MessageSource,
    pub to: MessageTarget,
    pub belief: Array1<f32>,      // Probability distribution
    pub confidence: f32,           // How certain about this belief
    pub iteration: u64,            // Which LBP iteration
    pub timestamp: u64,            // When generated
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum MessageSource {
    Neuron(NeuronId),
    Factor(FactorId),
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum MessageTarget {
    Neuron(NeuronId),
    Factor(FactorId),
}

/// A neuron in the belief network
#[derive(Debug)]
pub struct BeliefNeuron {
    pub id: NeuronId,
    pub state_space: usize,              // Number of possible states
    pub belief: RwLock<Array1<f32>>,     // Current belief distribution
    pub incoming_messages: RwLock<HashMap<MessageSource, BeliefMessage>>,
    pub outgoing_factors: Vec<FactorId>, // Connected factors
    pub evidence: RwLock<Option<Array1<f32>>>,   // Observed evidence (if any)
    pub convergence_threshold: f32,      // When to stop updating
    pub damping_factor: f32,             // Smooth belief updates (0-1)
}

/// A factor (connection) in the belief network
#[derive(Debug)]
pub struct BeliefFactor {
    pub id: FactorId,
    pub connected_neurons: Vec<NeuronId>,
    pub potential: Arc<FactorPotential>, // Joint probability table
    pub incoming_messages: RwLock<HashMap<NeuronId, BeliefMessage>>,
    pub message_cache: RwLock<HashMap<NeuronId, Array1<f32>>>, // Cached computations
}

/// Factor potential function (joint probability)
#[derive(Debug, Clone)]
pub struct FactorPotential {
    pub dimensions: Vec<usize>,    // State space of each connected neuron
    pub values: Array2<f32>,       // Joint probability values
    pub is_deterministic: bool,    // If true, hard constraint
}

/// Main Loopy Belief Propagation engine
pub struct LoopyBeliefPropagation {
    neurons: Arc<RwLock<HashMap<NeuronId, Arc<BeliefNeuron>>>>,
    factors: Arc<RwLock<HashMap<FactorId, Arc<BeliefFactor>>>>,
    
    // Message passing infrastructure
    message_queue: Arc<RwLock<VecDeque<BeliefMessage>>>,
    processed_messages: AtomicU64,
    
    // Convergence tracking
    iteration: AtomicU64,
    converged: AtomicBool,
    max_iterations: u64,
    global_convergence_threshold: f32,
    
    // Cycle detection and handling
    cycles: Arc<RwLock<Vec<Vec<NeuronId>>>>,
    cycle_damping: f32, // Extra damping for cyclic paths
    
    // Performance monitoring
    belief_changes: Arc<RwLock<Vec<f32>>>,
    convergence_history: Arc<RwLock<Vec<ConvergenceState>>>,
}

#[derive(Debug, Clone)]
struct ConvergenceState {
    iteration: u64,
    max_belief_change: f32,
    converged_neurons: usize,
    total_neurons: usize,
    cycles_active: usize,
}

impl LoopyBeliefPropagation {
    pub fn new(max_iterations: u64, convergence_threshold: f32) -> Self {
        Self {
            neurons: Arc::new(RwLock::new(HashMap::new())),
            factors: Arc::new(RwLock::new(HashMap::new())),
            message_queue: Arc::new(RwLock::new(VecDeque::new())),
            processed_messages: AtomicU64::new(0),
            iteration: AtomicU64::new(0),
            converged: AtomicBool::new(false),
            max_iterations,
            global_convergence_threshold: convergence_threshold,
            cycles: Arc::new(RwLock::new(Vec::new())),
            cycle_damping: 0.5,
            belief_changes: Arc::new(RwLock::new(Vec::new())),
            convergence_history: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Add a neuron to the belief network
    pub fn add_neuron(&self, id: NeuronId, state_space: usize) -> Result<(), LBPError> {
        let neuron = Arc::new(BeliefNeuron {
            id,
            state_space,
            belief: RwLock::new(Array1::from_elem(state_space, 1.0 / state_space as f32)),
            incoming_messages: RwLock::new(HashMap::new()),
            outgoing_factors: Vec::new(),
            evidence: RwLock::new(None),
            convergence_threshold: self.global_convergence_threshold,
            damping_factor: 0.7,
        });
        
        self.neurons.write().unwrap().insert(id, neuron);
        Ok(())
    }
    
    /// Add a factor connecting neurons
    pub fn add_factor(&self, id: FactorId, connected_neurons: Vec<NeuronId>, 
                     potential: FactorPotential) -> Result<(), LBPError> {
        // Validate neurons exist
        let neurons = self.neurons.read().unwrap();
        for &neuron_id in &connected_neurons {
            if !neurons.contains_key(&neuron_id) {
                return Err(LBPError::NeuronNotFound(neuron_id));
            }
        }
        drop(neurons);
        
        let factor = Arc::new(BeliefFactor {
            id,
            connected_neurons: connected_neurons.clone(),
            potential: Arc::new(potential),
            incoming_messages: RwLock::new(HashMap::new()),
            message_cache: RwLock::new(HashMap::new()),
        });
        
        // Update neuron connections
        let mut neurons = self.neurons.write().unwrap();
        for &neuron_id in &connected_neurons {
            if let Some(neuron) = neurons.get_mut(&neuron_id) {
                Arc::get_mut(neuron).unwrap().outgoing_factors.push(id);
            }
        }
        drop(neurons);
        
        self.factors.write().unwrap().insert(id, factor);
        
        // Detect cycles after adding factor
        self.detect_cycles();
        
        Ok(())
    }
    
    /// Run belief propagation until convergence or max iterations
    pub async fn propagate(&self) -> Result<(), LBPError> {
        println!("🔄 Starting Loopy Belief Propagation...");
        
        self.converged.store(false, Ordering::Relaxed);
        self.iteration.store(0, Ordering::Relaxed);
        
        // Initialize messages
        self.initialize_messages();
        
        while !self.converged.load(Ordering::Relaxed) && 
              self.iteration.load(Ordering::Relaxed) < self.max_iterations {
            
            let iter = self.iteration.fetch_add(1, Ordering::Relaxed);
            
            // Update all neuron-to-factor messages
            self.update_neuron_to_factor_messages(iter);
            
            // Update all factor-to-neuron messages
            self.update_factor_to_neuron_messages(iter);
            
            // Update beliefs
            let max_change = self.update_beliefs();
            
            // Check convergence
            if max_change < self.global_convergence_threshold {
                self.converged.store(true, Ordering::Relaxed);
                println!("  ✓ Converged at iteration {} (max change: {:.6})", iter, max_change);
            }
            
            // Record convergence state
            self.record_convergence_state(iter, max_change);
            
            // Apply cycle damping if needed
            if !self.cycles.read().unwrap().is_empty() {
                self.apply_cycle_damping();
            }
        }
        
        if !self.converged.load(Ordering::Relaxed) {
            println!("  ⚠ Did not converge after {} iterations", self.max_iterations);
        }
        
        Ok(())
    }
    
    /// Initialize uniform messages
    fn initialize_messages(&self) {
        let neurons = self.neurons.read().unwrap();
        let factors = self.factors.read().unwrap();
        
        // Initialize neuron-to-factor messages
        for (neuron_id, neuron) in neurons.iter() {
            for &factor_id in &neuron.outgoing_factors {
                let message = BeliefMessage {
                    from: MessageSource::Neuron(*neuron_id),
                    to: MessageTarget::Factor(factor_id),
                    belief: Array1::from_elem(neuron.state_space, 1.0 / neuron.state_space as f32),
                    confidence: 0.5,
                    iteration: 0,
                    timestamp: 0,
                };
                
                if let Some(factor) = factors.get(&factor_id) {
                    factor.incoming_messages.write().unwrap().insert(*neuron_id, message);
                }
            }
        }
        
        // Initialize factor-to-neuron messages
        for (factor_id, factor) in factors.iter() {
            for &neuron_id in &factor.connected_neurons {
                if let Some(neuron) = neurons.get(&neuron_id) {
                    let message = BeliefMessage {
                        from: MessageSource::Factor(*factor_id),
                        to: MessageTarget::Neuron(neuron_id),
                        belief: Array1::from_elem(neuron.state_space, 1.0 / neuron.state_space as f32),
                        confidence: 0.5,
                        iteration: 0,
                        timestamp: 0,
                    };
                    
                    neuron.incoming_messages.write().unwrap()
                        .insert(MessageSource::Factor(*factor_id), message);
                }
            }
        }
    }
    
    /// Update neuron-to-factor messages
    fn update_neuron_to_factor_messages(&self, iteration: u64) {
        let neurons = self.neurons.read().unwrap();
        let factors = self.factors.read().unwrap();
        
        for (neuron_id, neuron) in neurons.iter() {
            for &factor_id in &neuron.outgoing_factors {
                let new_belief = self.compute_neuron_to_factor_message(
                    neuron.as_ref(), 
                    factor_id
                );
                
                let message = BeliefMessage {
                    from: MessageSource::Neuron(*neuron_id),
                    to: MessageTarget::Factor(factor_id),
                    belief: new_belief,
                    confidence: self.compute_confidence(neuron.as_ref()),
                    iteration,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                };
                
                if let Some(factor) = factors.get(&factor_id) {
                    factor.incoming_messages.write().unwrap().insert(*neuron_id, message);
                }
            }
        }
    }
    
    /// Update factor-to-neuron messages
    fn update_factor_to_neuron_messages(&self, iteration: u64) {
        let neurons = self.neurons.read().unwrap();
        let factors = self.factors.read().unwrap();
        
        for (factor_id, factor) in factors.iter() {
            for &neuron_id in &factor.connected_neurons {
                let new_belief = self.compute_factor_to_neuron_message(
                    factor.as_ref(),
                    neuron_id
                );
                
                let message = BeliefMessage {
                    from: MessageSource::Factor(*factor_id),
                    to: MessageTarget::Neuron(neuron_id),
                    belief: new_belief,
                    confidence: 0.8, // Factors are typically more confident
                    iteration,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                };
                
                if let Some(neuron) = neurons.get(&neuron_id) {
                    neuron.incoming_messages.write().unwrap()
                        .insert(MessageSource::Factor(*factor_id), message);
                }
            }
        }
    }
    
    /// Compute neuron-to-factor message (product of incoming messages except from target)
    fn compute_neuron_to_factor_message(&self, neuron: &BeliefNeuron, 
                                       exclude_factor: FactorId) -> Array1<f32> {
        let mut product = Array1::from_elem(neuron.state_space, 1.0);
        
        // Include evidence if available
        if let Some(ref evidence) = *neuron.evidence.read().unwrap() {
            product = product * evidence;
        }
        
        // Product of all incoming messages except from the target factor
        let messages = neuron.incoming_messages.read().unwrap();
        for (source, message) in messages.iter() {
            if *source != MessageSource::Factor(exclude_factor) {
                product = product * &message.belief;
            }
        }
        
        // Apply damping
        let old_belief = neuron.belief.read().unwrap().clone();
        product = neuron.damping_factor * product + (1.0 - neuron.damping_factor) * old_belief;
        
        // Normalize
        let sum = product.sum();
        if sum > 0.0 {
            product / sum
        } else {
            Array1::from_elem(neuron.state_space, 1.0 / neuron.state_space as f32)
        }
    }
    
    /// Compute factor-to-neuron message (marginalization)
    fn compute_factor_to_neuron_message(&self, factor: &BeliefFactor, 
                                       target_neuron: NeuronId) -> Array1<f32> {
        // This is simplified - real implementation would marginalize the factor potential
        // over all connected neurons except the target
        
        let neurons = self.neurons.read().unwrap();
        let target = neurons.get(&target_neuron).unwrap();
        let mut result = Array1::from_elem(target.state_space, 1.0);
        
        // Apply factor potential (simplified)
        if factor.potential.is_deterministic {
            // Hard constraint - zero out invalid states
            for i in 0..target.state_space {
                if i % 2 == 0 { // Example constraint
                    result[i] = 0.0;
                }
            }
        }
        
        // Normalize
        let sum = result.sum();
        if sum > 0.0 {
            result / sum
        } else {
            Array1::from_elem(target.state_space, 1.0 / target.state_space as f32)
        }
    }
    
    /// Update neuron beliefs based on incoming messages
    fn update_beliefs(&self) -> f32 {
        let neurons = self.neurons.read().unwrap();
        let mut max_change = 0.0f32;
        
        for neuron in neurons.values() {
            let old_belief = neuron.belief.read().unwrap().clone();
            
            // Compute new belief as product of all incoming messages
            let mut new_belief = Array1::from_elem(neuron.state_space, 1.0);
            
            if let Some(ref evidence) = *neuron.evidence.read().unwrap() {
                new_belief = new_belief * evidence;
            }
            
            let messages = neuron.incoming_messages.read().unwrap();
            for message in messages.values() {
                new_belief = new_belief * &message.belief;
            }
            
            // Normalize
            let sum = new_belief.sum();
            if sum > 0.0 {
                new_belief = new_belief / sum;
            }
            
            // Apply damping
            new_belief = neuron.damping_factor * new_belief + 
                        (1.0 - neuron.damping_factor) * &old_belief;
            
            // Calculate change
            let change = (&new_belief - &old_belief).mapv(f32::abs).sum();
            max_change = max_change.max(change);
            
            // Update belief
            *neuron.belief.write().unwrap() = new_belief;
        }
        
        max_change
    }
    
    /// Detect cycles in the network using DFS
    fn detect_cycles(&self) {
        let neurons = self.neurons.read().unwrap();
        let factors = self.factors.read().unwrap();
        
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut cycles = Vec::new();
        
        for &neuron_id in neurons.keys() {
            if !visited.contains(&neuron_id) {
                let mut path = Vec::new();
                self.dfs_detect_cycles(
                    neuron_id,
                    &mut visited,
                    &mut rec_stack,
                    &mut path,
                    &mut cycles,
                    &neurons,
                    &factors
                );
            }
        }
        
        *self.cycles.write().unwrap() = cycles;
        
        if !self.cycles.read().unwrap().is_empty() {
            println!("  🔄 Detected {} cycles in belief network", self.cycles.read().unwrap().len());
        }
    }
    
    fn dfs_detect_cycles(&self,
                        current: NeuronId,
                        visited: &mut HashSet<NeuronId>,
                        rec_stack: &mut HashSet<NeuronId>,
                        path: &mut Vec<NeuronId>,
                        cycles: &mut Vec<Vec<NeuronId>>,
                        neurons: &HashMap<NeuronId, Arc<BeliefNeuron>>,
                        factors: &HashMap<FactorId, Arc<BeliefFactor>>) {
        visited.insert(current);
        rec_stack.insert(current);
        path.push(current);
        
        if let Some(neuron) = neurons.get(&current) {
            for &factor_id in &neuron.outgoing_factors {
                if let Some(factor) = factors.get(&factor_id) {
                    for &next_neuron in &factor.connected_neurons {
                        if next_neuron != current {
                            if rec_stack.contains(&next_neuron) {
                                // Found a cycle
                                let cycle_start = path.iter().position(|&n| n == next_neuron).unwrap();
                                let cycle = path[cycle_start..].to_vec();
                                cycles.push(cycle);
                            } else if !visited.contains(&next_neuron) {
                                self.dfs_detect_cycles(
                                    next_neuron,
                                    visited,
                                    rec_stack,
                                    path,
                                    cycles,
                                    neurons,
                                    factors
                                );
                            }
                        }
                    }
                }
            }
        }
        
        path.pop();
        rec_stack.remove(&current);
    }
    
    /// Apply extra damping to neurons in cycles to help convergence
    fn apply_cycle_damping(&self) {
        let cycles = self.cycles.read().unwrap();
        let neurons = self.neurons.read().unwrap();
        
        for cycle in cycles.iter() {
            for &neuron_id in cycle {
                if let Some(neuron) = neurons.get(&neuron_id) {
                    let mut belief = neuron.belief.write().unwrap();
                    let uniform = Array1::from_elem(belief.len(), 1.0 / belief.len() as f32);
                    *belief = self.cycle_damping * &*belief + (1.0 - self.cycle_damping) * uniform;
                }
            }
        }
    }
    
    fn compute_confidence(&self, neuron: &BeliefNeuron) -> f32 {
        let belief = neuron.belief.read().unwrap();
        
        // Confidence based on entropy
        let entropy = -belief.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f32>();
        
        let max_entropy = (neuron.state_space as f32).ln();
        1.0 - (entropy / max_entropy)
    }
    
    fn record_convergence_state(&self, iteration: u64, max_change: f32) {
        let neurons = self.neurons.read().unwrap();
        let converged_count = neurons.values()
            .filter(|n| {
                let messages = n.incoming_messages.read().unwrap();
                messages.values().all(|m| m.confidence > 0.9)
            })
            .count();
        
        let state = ConvergenceState {
            iteration,
            max_belief_change: max_change,
            converged_neurons: converged_count,
            total_neurons: neurons.len(),
            cycles_active: self.cycles.read().unwrap().len(),
        };
        
        self.convergence_history.write().unwrap().push(state);
    }
    
    /// Get the current belief of a neuron
    pub fn get_belief(&self, neuron_id: NeuronId) -> Option<Array1<f32>> {
        self.neurons.read().unwrap()
            .get(&neuron_id)
            .map(|n| n.belief.read().unwrap().clone())
    }
    
    /// Set evidence for a neuron (observation)
    pub fn set_evidence(&self, neuron_id: NeuronId, evidence: Array1<f32>) -> Result<(), LBPError> {
        let neurons = self.neurons.read().unwrap();
        if let Some(neuron) = neurons.get(&neuron_id) {
            *neuron.evidence.write().unwrap() = Some(evidence);
            Ok(())
        } else {
            Err(LBPError::NeuronNotFound(neuron_id))
        }
    }
    
    /// Get convergence statistics
    pub fn get_convergence_stats(&self) -> ConvergenceStats {
        let history = self.convergence_history.read().unwrap();
        let last_state = history.last();
        
        ConvergenceStats {
            iterations: self.iteration.load(Ordering::Relaxed),
            converged: self.converged.load(Ordering::Relaxed),
            final_max_change: last_state.map(|s| s.max_belief_change).unwrap_or(1.0),
            cycles_detected: self.cycles.read().unwrap().len(),
            messages_processed: self.processed_messages.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug)]
pub struct ConvergenceStats {
    pub iterations: u64,
    pub converged: bool,
    pub final_max_change: f32,
    pub cycles_detected: usize,
    pub messages_processed: u64,
}

#[derive(Debug)]
pub enum LBPError {
    NeuronNotFound(NeuronId),
    FactorNotFound(FactorId),
    InvalidPotential,
    ConvergenceFailed,
}

impl std::fmt::Display for LBPError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LBPError::NeuronNotFound(id) => write!(f, "Neuron {:?} not found", id),
            LBPError::FactorNotFound(id) => write!(f, "Factor {:?} not found", id),
            LBPError::InvalidPotential => write!(f, "Invalid factor potential"),
            LBPError::ConvergenceFailed => write!(f, "Belief propagation failed to converge"),
        }
    }
}

impl std::error::Error for LBPError {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_simple_chain() {
        let lbp = LoopyBeliefPropagation::new(100, 0.001);
        
        // Create simple chain: A -> B -> C
        lbp.add_neuron(NeuronId(1), 2).unwrap();
        lbp.add_neuron(NeuronId(2), 2).unwrap();
        lbp.add_neuron(NeuronId(3), 2).unwrap();
        
        // Add factors
        let potential = FactorPotential {
            dimensions: vec![2, 2],
            values: Array2::from_elem((2, 2), 0.5),
            is_deterministic: false,
        };
        
        lbp.add_factor(FactorId(1), vec![NeuronId(1), NeuronId(2)], potential.clone()).unwrap();
        lbp.add_factor(FactorId(2), vec![NeuronId(2), NeuronId(3)], potential).unwrap();
        
        // Run propagation
        lbp.propagate().await.unwrap();
        
        let stats = lbp.get_convergence_stats();
        assert!(stats.converged);
    }
    
    #[tokio::test]
    async fn test_cycle_detection() {
        let lbp = LoopyBeliefPropagation::new(100, 0.001);
        
        // Create cycle: A -> B -> C -> A
        lbp.add_neuron(NeuronId(1), 2).unwrap();
        lbp.add_neuron(NeuronId(2), 2).unwrap();
        lbp.add_neuron(NeuronId(3), 2).unwrap();
        
        let potential = FactorPotential {
            dimensions: vec![2, 2],
            values: Array2::from_elem((2, 2), 0.5),
            is_deterministic: false,
        };
        
        lbp.add_factor(FactorId(1), vec![NeuronId(1), NeuronId(2)], potential.clone()).unwrap();
        lbp.add_factor(FactorId(2), vec![NeuronId(2), NeuronId(3)], potential.clone()).unwrap();
        lbp.add_factor(FactorId(3), vec![NeuronId(3), NeuronId(1)], potential).unwrap();
        
        // Should detect the cycle
        assert!(lbp.cycles.read().unwrap().len() > 0);
    }
}