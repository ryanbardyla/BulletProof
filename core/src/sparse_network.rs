//! Sparse Tryte Network Layer
//! 
//! Exploits biological sparsity where 95% of neurons are at baseline.
//! Only processes active neurons (Inhibited or Activated), achieving
//! massive computational savings that mirror brain efficiency.

use crate::tryte::{Tryte, TryteNeuron, PackedTrytes};
use std::collections::{HashMap, HashSet};
use crossbeam::channel::{unbounded, Sender, Receiver};

/// Sparse representation of active neurons only
#[derive(Debug, Clone)]
pub struct SparseActivation {
    /// Only non-baseline neurons stored
    pub active_neurons: HashMap<NeuronId, Tryte>,
    /// Total network size (including baseline)
    pub total_size: usize,
    /// Timestamp of activation
    pub timestamp: u64,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct NeuronId(pub usize);

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct LayerId(pub usize);

/// Connection between neurons with trinary weight
#[derive(Debug, Clone)]
pub struct TryteSynapse {
    pub from: NeuronId,
    pub to: NeuronId,
    pub weight: f32,
    pub plasticity: Tryte, // LTD (-1), stable (0), LTP (+1)
}

/// Sparse layer that only computes active neurons
pub struct SparseTryteLayer {
    pub id: LayerId,
    pub neurons: Vec<TryteNeuron>,
    pub packed_states: PackedTrytes,
    
    /// Sparse connectivity - only store existing connections
    pub synapses: HashMap<NeuronId, Vec<TryteSynapse>>,
    
    /// Active set - neurons that need computation
    pub active_set: HashSet<NeuronId>,
    
    /// Activation history for temporal dynamics
    pub history: Vec<SparseActivation>,
    pub history_size: usize,
    
    /// Statistics
    pub total_computations_saved: u64,
    pub total_energy_saved: f64,
}

impl SparseTryteLayer {
    pub fn new(id: LayerId, size: usize) -> Self {
        Self {
            id,
            neurons: vec![TryteNeuron::new(); size],
            packed_states: PackedTrytes::new(size),
            synapses: HashMap::new(),
            active_set: HashSet::new(),
            history: Vec::new(),
            history_size: 10,
            total_computations_saved: 0,
            total_energy_saved: 0.0,
        }
    }
    
    /// Add sparse connection (only non-zero weights)
    pub fn add_synapse(&mut self, from: NeuronId, to: NeuronId, weight: f32) {
        if weight.abs() < 0.001 {
            return; // Don't store near-zero connections
        }
        
        let synapse = TryteSynapse {
            from,
            to,
            weight,
            plasticity: Tryte::Baseline,
        };
        
        self.synapses.entry(to).or_insert_with(Vec::new).push(synapse);
    }
    
    /// Process only active neurons (95% savings!)
    pub fn forward_sparse(&mut self, input: &SparseActivation) -> SparseActivation {
        let mut output_active = HashMap::new();
        let mut computations_skipped = 0;
        
        // Update active set from input
        self.active_set.clear();
        for (&neuron_id, _) in &input.active_neurons {
            self.active_set.insert(neuron_id);
            
            // Propagate to connected neurons
            if let Some(synapses) = self.synapses.get(&neuron_id) {
                for synapse in synapses {
                    self.active_set.insert(synapse.to);
                }
            }
        }
        
        // Process ONLY active neurons
        for &neuron_id in &self.active_set {
            let idx = neuron_id.0;
            
            // Compute weighted sum from active inputs only
            let mut sum = 0.0;
            if let Some(synapses) = self.synapses.get(&neuron_id) {
                for synapse in synapses {
                    if let Some(&input_state) = input.active_neurons.get(&synapse.from) {
                        sum += synapse.weight * (input_state as i8 as f32);
                    }
                    // Baseline inputs contribute 0, so we skip them!
                }
            }
            
            // Process through neuron
            let output_state = self.neurons[idx].process(sum);
            
            // Only store if non-baseline
            if output_state != Tryte::Baseline {
                output_active.insert(neuron_id, output_state);
                self.packed_states.set(idx, output_state);
            } else {
                computations_skipped += 1;
            }
        }
        
        // Calculate savings
        let total_neurons = self.neurons.len();
        let computed = self.active_set.len();
        let skipped = total_neurons - computed;
        
        self.total_computations_saved += skipped as u64;
        self.total_energy_saved += (skipped as f64) * 1.0; // 1 unit per skipped neuron
        
        println!("⚡ Sparse processing: {} active, {} skipped ({:.1}% saved!)",
                computed, skipped, (skipped as f32 / total_neurons as f32) * 100.0);
        
        // Record history
        let activation = SparseActivation {
            active_neurons: output_active,
            total_size: total_neurons,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        };
        
        self.history.push(activation.clone());
        if self.history.len() > self.history_size {
            self.history.remove(0);
        }
        
        activation
    }
    
    /// Update synaptic plasticity based on activity
    pub fn update_plasticity(&mut self, learning_rate: f32) {
        for synapses in self.synapses.values_mut() {
            for synapse in synapses {
                // Hebbian learning: neurons that fire together wire together
                let from_active = self.active_set.contains(&synapse.from);
                let to_active = self.active_set.contains(&synapse.to);
                
                if from_active && to_active {
                    // Both active - strengthen (LTP)
                    synapse.weight += learning_rate;
                    synapse.plasticity = Tryte::Activated;
                } else if from_active && !to_active {
                    // Pre but not post - weaken (LTD)
                    synapse.weight -= learning_rate * 0.5;
                    synapse.plasticity = Tryte::Inhibited;
                }
                
                // Clip weights
                synapse.weight = synapse.weight.clamp(-2.0, 2.0);
            }
        }
    }
    
    /// Get sparsity statistics
    pub fn get_stats(&self) -> SparseStats {
        let total = self.neurons.len();
        let active = self.active_set.len();
        let sparsity = 1.0 - (active as f32 / total as f32);
        
        SparseStats {
            total_neurons: total,
            active_neurons: active,
            sparsity_percentage: sparsity * 100.0,
            computations_saved: self.total_computations_saved,
            energy_saved: self.total_energy_saved,
            synapse_count: self.synapses.values().map(|v| v.len()).sum(),
        }
    }
}

/// Multi-layer sparse network
pub struct SparseTryteNetwork {
    pub layers: Vec<SparseTryteLayer>,
    pub inter_layer_connections: HashMap<(LayerId, LayerId), Vec<TryteSynapse>>,
    
    /// Parallel processing channels
    pub layer_channels: Vec<(Sender<SparseActivation>, Receiver<SparseActivation>)>,
}

impl SparseTryteNetwork {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let layer_sizes = vec![input_size, output_size];
        let mut layers = Vec::new();
        let mut channels = Vec::new();
        
        for (i, &size) in layer_sizes.iter().enumerate() {
            layers.push(SparseTryteLayer::new(LayerId(i), size));
            channels.push(unbounded());
        }
        
        Self {
            layers,
            inter_layer_connections: HashMap::new(),
            layer_channels: channels,
        }
    }
    
    pub fn new_multi_layer(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        let mut channels = Vec::new();
        
        for (i, &size) in layer_sizes.iter().enumerate() {
            layers.push(SparseTryteLayer::new(LayerId(i), size));
            channels.push(unbounded());
        }
        
        Self {
            layers,
            inter_layer_connections: HashMap::new(),
            layer_channels: channels,
        }
    }
    
    /// Forward pass with sparse inputs (for benchmarking)
    pub fn forward_sparse(&mut self, inputs: &[Tryte], skip_baseline: bool) -> Vec<Tryte> {
        // Convert tryte inputs to sparse activation
        let mut active_input = HashMap::new();
        
        for (i, &tryte) in inputs.iter().enumerate() {
            if skip_baseline && tryte == Tryte::Baseline {
                continue; // Skip baseline neurons for efficiency!
            }
            if tryte != Tryte::Baseline {
                active_input.insert(NeuronId(i), tryte);
            }
        }
        
        let mut activation = SparseActivation {
            active_neurons: active_input,
            total_size: inputs.len(),
            timestamp: 0,
        };
        
        // Process through first layer
        if !self.layers.is_empty() {
            activation = self.layers[0].forward_sparse(&activation);
        }
        
        // Convert back to dense representation for benchmarking
        let mut output = vec![Tryte::Baseline; self.layers.last().map_or(inputs.len(), |l| l.neurons.len())];
        
        for (&neuron_id, &state) in &activation.active_neurons {
            if neuron_id.0 < output.len() {
                output[neuron_id.0] = state;
            }
        }
        
        output
    }
    
    /// Connect layers with sparse random connections
    pub fn connect_layers(&mut self, from: LayerId, to: LayerId, density: f32) {
        let from_size = self.layers[from.0].neurons.len();
        let to_size = self.layers[to.0].neurons.len();
        
        let mut connections = Vec::new();
        
        for i in 0..from_size {
            for j in 0..to_size {
                if fastrand::f32() < density {
                    // Random weight with biological distribution
                    let weight = (fastrand::f32() - 0.5) * 2.0;
                    
                    connections.push(TryteSynapse {
                        from: NeuronId(i),
                        to: NeuronId(j),
                        weight,
                        plasticity: Tryte::Baseline,
                    });
                    
                    self.layers[to.0].add_synapse(NeuronId(i), NeuronId(j), weight);
                }
            }
        }
        
        println!("Connected layer {} → {} with {} synapses ({:.1}% density)",
                from.0, to.0, connections.len(),
                connections.len() as f32 * 100.0 / (from_size * to_size) as f32);
        
        self.inter_layer_connections.insert((from, to), connections);
    }
    
    /// Forward pass through entire network (sparse)
    pub fn forward(&mut self, input: &[f32]) -> SparseActivation {
        // Convert dense input to sparse
        let mut active_input = HashMap::new();
        for (i, &val) in input.iter().enumerate() {
            if val.abs() > 0.01 {
                let state = if val > 0.5 {
                    Tryte::Activated
                } else if val < -0.5 {
                    Tryte::Inhibited
                } else {
                    continue; // Skip near-baseline
                };
                active_input.insert(NeuronId(i), state);
            }
        }
        
        let mut activation = SparseActivation {
            active_neurons: active_input,
            total_size: input.len(),
            timestamp: 0,
        };
        
        // Process through layers
        for layer in &mut self.layers {
            activation = layer.forward_sparse(&activation);
        }
        
        activation
    }
    
    /// Get network-wide statistics
    pub fn get_network_stats(&self) -> NetworkStats {
        let mut total_neurons = 0;
        let mut total_active = 0;
        let mut total_synapses = 0;
        let mut total_energy_saved = 0.0;
        
        for layer in &self.layers {
            let stats = layer.get_stats();
            total_neurons += stats.total_neurons;
            total_active += stats.active_neurons;
            total_synapses += stats.synapse_count;
            total_energy_saved += stats.energy_saved;
        }
        
        NetworkStats {
            layers: self.layers.len(),
            total_neurons,
            total_active,
            total_synapses,
            overall_sparsity: (1.0 - total_active as f32 / total_neurons as f32) * 100.0,
            total_energy_saved,
        }
    }
}

#[derive(Debug)]
pub struct SparseStats {
    pub total_neurons: usize,
    pub active_neurons: usize,
    pub sparsity_percentage: f32,
    pub computations_saved: u64,
    pub energy_saved: f64,
    pub synapse_count: usize,
}

#[derive(Debug)]
pub struct NetworkStats {
    pub layers: usize,
    pub total_neurons: usize,
    pub total_active: usize,
    pub total_synapses: usize,
    pub overall_sparsity: f32,
    pub total_energy_saved: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sparse_processing() {
        let mut layer = SparseTryteLayer::new(LayerId(0), 1000);
        
        // Create sparse input (5% active)
        let mut active = HashMap::new();
        for i in (0..1000).step_by(20) {
            active.insert(NeuronId(i), Tryte::Activated);
        }
        
        let input = SparseActivation {
            active_neurons: active,
            total_size: 1000,
            timestamp: 0,
        };
        
        let output = layer.forward_sparse(&input);
        
        // Should be very sparse
        assert!(output.active_neurons.len() < 100);
        
        let stats = layer.get_stats();
        assert!(stats.sparsity_percentage > 90.0);
    }
    
    #[test]
    fn test_multi_layer_network() {
        let mut network = SparseTryteNetwork::new_multi_layer(&[100, 50, 10]);
        
        // Connect layers sparsely
        network.connect_layers(LayerId(0), LayerId(1), 0.1);
        network.connect_layers(LayerId(1), LayerId(2), 0.2);
        
        // Process sparse input
        let mut input = vec![0.0; 100];
        for i in 0..5 {
            input[i * 20] = 1.0;
        }
        
        let output = network.forward(&input);
        
        // Check sparsity preserved
        assert!(output.active_neurons.len() < output.total_size / 2);
    }
}