//! Revolutionary Fire-and-Forget Autodiff Engine
//! 
//! This autodiff system is designed from the ground up for asynchronous, 
//! fire-and-forget neurons with temporal gradient accumulation and auto-routing.

use crate::ir::*;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use crossbeam::queue::{ArrayQueue, SegQueue};
use crossbeam::atomic::AtomicCell;
use std::sync::atomic::{AtomicU64, Ordering};

/// Temporal gradient accumulation for fire-and-forget neurons
#[derive(Debug, Clone)]
pub struct TemporalGradient {
    pub value_id: ValueId,
    pub gradient: Arc<Tensor>,
    pub timestamp: u64,
    pub path_id: PathId,
    pub spike_count: u32,
}

/// Unique identifier for neuron firing paths
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PathId(u64);

/// Spike event in the fire-and-forget system
#[derive(Debug, Clone)]
pub struct SpikeEvent {
    pub neuron_id: ValueId,
    pub timestamp: u64,
    pub activation: f32,
    pub path_id: PathId,
    pub refractory_until: u64,
}

/// Lock-free spike routing queue
pub struct SpikeRouter {
    /// High-priority spikes (gradients, errors)
    priority_queue: ArrayQueue<SpikeEvent>,
    /// Normal forward propagation spikes  
    normal_queue: SegQueue<SpikeEvent>,
    /// Work-stealing queues per thread
    worker_queues: Vec<ArrayQueue<SpikeEvent>>,
    /// Global timestamp counter
    global_time: AtomicU64,
    /// Path performance tracking for auto-routing
    path_rewards: Arc<HashMap<PathId, AtomicCell<f32>>>,
}

impl SpikeRouter {
    pub fn new(num_workers: usize) -> Self {
        let mut worker_queues = Vec::new();
        for _ in 0..num_workers {
            worker_queues.push(ArrayQueue::new(10000)); // Pre-allocated spike pools
        }
        
        Self {
            priority_queue: ArrayQueue::new(50000),
            normal_queue: SegQueue::new(),
            worker_queues,
            global_time: AtomicU64::new(0),
            path_rewards: Arc::new(HashMap::new()),
        }
    }
    
    /// Route spike using multi-armed bandit for path selection
    pub fn route_spike(&self, spike: SpikeEvent, worker_id: Option<usize>) {
        let current_time = self.global_time.fetch_add(1, Ordering::Relaxed);
        let mut spike = spike;
        spike.timestamp = current_time;
        
        // Priority routing for gradients
        if self.is_gradient_spike(&spike) {
            let _ = self.priority_queue.push(spike);
            return;
        }
        
        // Work-stealing for normal spikes
        if let Some(worker_id) = worker_id {
            if worker_id < self.worker_queues.len() {
                if self.worker_queues[worker_id].push(spike.clone()).is_ok() {
                    return;
                }
            }
        }
        
        // Fallback to global queue
        self.normal_queue.push(spike);
    }
    
    /// Multi-armed bandit path selection
    pub fn select_optimal_path(&self, available_paths: &[PathId]) -> PathId {
        if available_paths.is_empty() {
            return PathId(0);
        }
        
        // Upper Confidence Bound (UCB1) for path selection
        let mut best_path = available_paths[0];
        let mut best_score = f32::NEG_INFINITY;
        
        for &path in available_paths {
            let reward = self.path_rewards
                .get(&path)
                .map(|r| r.load())
                .unwrap_or(0.0);
            
            // UCB1 exploration bonus
            let exploration = (2.0 * (self.global_time.load(Ordering::Relaxed) as f32).ln() 
                / (self.get_path_usage_count(path) as f32 + 1.0)).sqrt();
            
            let score = reward + exploration;
            if score > best_score {
                best_score = score;
                best_path = path;
            }
        }
        
        best_path
    }
    
    fn is_gradient_spike(&self, spike: &SpikeEvent) -> bool {
        // Heuristic: negative activations often indicate gradients
        spike.activation < 0.0
    }
    
    fn get_path_usage_count(&self, _path: PathId) -> u32 {
        // TODO: Implement path usage tracking
        1
    }
}

/// Event-sourced autodiff system for temporal gradient accumulation
pub struct TemporalAutodiff {
    /// Gradient accumulation buffers indexed by value and time
    gradient_buffers: HashMap<ValueId, VecDeque<TemporalGradient>>,
    /// Spike routing engine
    router: SpikeRouter,
    /// Time window for gradient accumulation (microseconds)
    accumulation_window: u64,
    /// Path success tracking for auto-routing
    path_performance: HashMap<PathId, PathStats>,
    /// Forward computation graph for dependency tracking
    computation_graph: HashMap<ValueId, Vec<ValueId>>,
}

#[derive(Debug, Clone)]
struct PathStats {
    success_rate: f32,
    avg_latency: f32,
    gradient_flow_quality: f32,
    usage_count: u32,
}

impl TemporalAutodiff {
    pub fn new(num_workers: usize) -> Self {
        Self {
            gradient_buffers: HashMap::new(),
            router: SpikeRouter::new(num_workers),
            accumulation_window: 1000, // 1ms window
            path_performance: HashMap::new(),
            computation_graph: HashMap::new(),
        }
    }
    
    /// Register a fire-and-forget computation
    pub fn register_computation(&mut self, instruction: &Instruction) -> PathId {
        let path_id = self.generate_path_id();
        
        match instruction {
            Instruction::TensorAdd { result, left, right, .. } => {
                // Record dependencies for gradient routing
                self.computation_graph
                    .entry(*result)
                    .or_default()
                    .extend(&[*left, *right]);
            },
            Instruction::NeuronFire { result, input, .. } => {
                self.computation_graph
                    .entry(*result)
                    .or_default()
                    .push(*input);
            },
            _ => {}
        }
        
        path_id
    }
    
    /// Accumulate gradients from fire-and-forget neurons
    pub fn accumulate_gradient(&mut self, grad: TemporalGradient) {
        let current_time = self.router.global_time.load(Ordering::Relaxed);
        
        // Add to accumulation buffer
        let buffer = self.gradient_buffers
            .entry(grad.value_id)
            .or_insert_with(VecDeque::new);
            
        buffer.push_back(grad.clone());
        
        // Clean old gradients outside accumulation window
        while let Some(front) = buffer.front() {
            if current_time - front.timestamp > self.accumulation_window {
                buffer.pop_front();
            } else {
                break;
            }
        }
        
        // Update path performance based on gradient quality
        self.update_path_performance(&grad);
    }
    
    /// Compute final accumulated gradient for a value
    pub fn get_accumulated_gradient(&self, value_id: ValueId) -> Option<Arc<Tensor>> {
        let buffer = self.gradient_buffers.get(&value_id)?;
        
        if buffer.is_empty() {
            return None;
        }
        
        // Weighted accumulation based on spike timing and path quality
        let mut accumulated = buffer[0].gradient.as_ref().clone();
        let mut total_weight = self.get_path_weight(buffer[0].path_id);
        
        for grad in buffer.iter().skip(1) {
            let weight = self.get_path_weight(grad.path_id);
            // Temporal decay: more recent gradients have higher weight
            let time_weight = 1.0 / (1.0 + (buffer.back().unwrap().timestamp - grad.timestamp) as f32 / 1000.0);
            let final_weight = weight * time_weight;
            
            accumulated = accumulated + (grad.gradient.as_ref() * final_weight);
            total_weight += final_weight;
        }
        
        Some(Arc::new(accumulated / total_weight))
    }
    
    /// Auto-routing: discover new paths through failed neuron recovery
    pub fn discover_recovery_path(&mut self, failed_neuron: ValueId, target: ValueId) -> Option<PathId> {
        // Breadth-first search for alternative paths
        let mut queue = VecDeque::new();
        let mut visited = std::collections::HashSet::new();
        
        // Find all neurons that can reach the target
        for (&neuron, dependencies) in &self.computation_graph {
            if dependencies.contains(&target) && neuron != failed_neuron {
                queue.push_back((neuron, vec![neuron]));
                visited.insert(neuron);
            }
        }
        
        while let Some((current, path)) = queue.pop_front() {
            if self.computation_graph
                .get(&current)
                .map_or(false, |deps| deps.contains(&failed_neuron)) {
                // Found a path that bypasses the failed neuron
                let path_id = self.generate_path_id();
                
                // Initialize path with low performance to encourage exploration
                self.path_performance.insert(path_id, PathStats {
                    success_rate: 0.5,
                    avg_latency: 1000.0, // High initial latency
                    gradient_flow_quality: 0.3,
                    usage_count: 0,
                });
                
                return Some(path_id);
            }
            
            // Explore further
            if let Some(dependencies) = self.computation_graph.get(&current) {
                for &dep in dependencies {
                    if !visited.contains(&dep) && dep != failed_neuron {
                        visited.insert(dep);
                        let mut new_path = path.clone();
                        new_path.push(dep);
                        queue.push_back((dep, new_path));
                    }
                }
            }
        }
        
        None
    }
    
    fn generate_path_id(&self) -> PathId {
        PathId(self.router.global_time.fetch_add(1, Ordering::Relaxed))
    }
    
    fn get_path_weight(&self, path_id: PathId) -> f32 {
        self.path_performance
            .get(&path_id)
            .map(|stats| stats.success_rate * stats.gradient_flow_quality)
            .unwrap_or(1.0)
    }
    
    fn update_path_performance(&mut self, grad: &TemporalGradient) {
        let stats = self.path_performance
            .entry(grad.path_id)
            .or_insert(PathStats {
                success_rate: 1.0,
                avg_latency: 0.0,
                gradient_flow_quality: 1.0,
                usage_count: 0,
            });
        
        stats.usage_count += 1;
        
        // Update gradient flow quality based on gradient magnitude and spike count
        let gradient_magnitude = grad.gradient.as_ref().abs().sum();
        let quality = (gradient_magnitude / (grad.spike_count as f32 + 1.0)).min(1.0);
        
        // Exponential moving average
        let alpha = 0.1;
        stats.gradient_flow_quality = alpha * quality + (1.0 - alpha) * stats.gradient_flow_quality;
    }
}

/// Mock tensor type for compilation
#[derive(Debug, Clone)]
pub struct Tensor;

impl Tensor {
    pub fn abs(&self) -> Tensor { Tensor }
    pub fn sum(&self) -> f32 { 1.0 }
}

impl std::ops::Add for Tensor {
    type Output = Tensor;
    fn add(self, _other: Tensor) -> Tensor { Tensor }
}

impl std::ops::Mul<f32> for &Tensor {
    type Output = Tensor;
    fn mul(self, _scalar: f32) -> Tensor { Tensor }
}

impl std::ops::Div<f32> for Tensor {
    type Output = Tensor;
    fn div(self, _scalar: f32) -> Tensor { Tensor }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spike_routing() {
        let router = SpikeRouter::new(4);
        let spike = SpikeEvent {
            neuron_id: ValueId(1),
            timestamp: 0,
            activation: 0.5,
            path_id: PathId(1),
            refractory_until: 100,
        };
        
        router.route_spike(spike, Some(0));
        // Test passes if no panic
    }
    
    #[test]
    fn test_temporal_gradient_accumulation() {
        let mut autodiff = TemporalAutodiff::new(4);
        
        let grad = TemporalGradient {
            value_id: ValueId(1),
            gradient: Arc::new(Tensor),
            timestamp: 100,
            path_id: PathId(1),
            spike_count: 5,
        };
        
        autodiff.accumulate_gradient(grad);
        let result = autodiff.get_accumulated_gradient(ValueId(1));
        assert!(result.is_some());
    }
    
    #[test] 
    fn test_path_recovery() {
        let mut autodiff = TemporalAutodiff::new(4);
        
        // Create a simple computation graph: 1 -> 2 -> 3
        autodiff.computation_graph.insert(ValueId(2), vec![ValueId(1)]);
        autodiff.computation_graph.insert(ValueId(3), vec![ValueId(2)]);
        autodiff.computation_graph.insert(ValueId(4), vec![ValueId(1)]); // Alternative path
        
        let recovery_path = autodiff.discover_recovery_path(ValueId(2), ValueId(3));
        assert!(recovery_path.is_some());
    }
}