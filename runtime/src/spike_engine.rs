//! High-Performance Spike Routing Engine for Fire-and-Forget Neurons
//! 
//! This engine handles millions of neuron spikes per second with lock-free
//! work-stealing and intelligent path optimization using reinforcement learning.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicU32, AtomicBool, Ordering};
use std::collections::HashMap;
use std::time::{Instant, Duration};
use crossbeam::queue::{ArrayQueue, SegQueue};
use crossbeam::atomic::AtomicCell;
use crossbeam::utils::Backoff;
use crossbeam::thread;

/// Unique identifier for neurons in the network
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct NeuronId(pub u64);

/// Unique identifier for synaptic pathways
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PathId(pub u64);

/// Single spike event in the neural network
#[derive(Debug, Clone)]
pub struct Spike {
    pub id: u64,
    pub source: NeuronId,
    pub target: NeuronId,
    pub weight: f32,
    pub timestamp: u64,
    pub path_id: PathId,
    pub priority: SpikePriority,
    pub refractory_period: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpikePriority {
    Critical = 0,   // Error signals, gradient updates
    High = 1,       // Important forward propagation
    Normal = 2,     // Regular neural activity
    Background = 3, // Housekeeping, plasticity updates
}

/// Performance metrics for path optimization
#[derive(Debug)]
struct PathMetrics {
    success_count: AtomicU32,
    failure_count: AtomicU32,
    total_latency: AtomicU64,
    last_success: AtomicU64,
    congestion_score: AtomicCell<f32>,
    reward_estimate: AtomicCell<f32>,
}

impl PathMetrics {
    fn new() -> Self {
        Self {
            success_count: AtomicU32::new(0),
            failure_count: AtomicU32::new(0),
            total_latency: AtomicU64::new(0),
            last_success: AtomicU64::new(0),
            congestion_score: AtomicCell::new(0.0),
            reward_estimate: AtomicCell::new(0.5), // Start optimistic
        }
    }
    
    fn record_success(&self, latency: u64, timestamp: u64) {
        self.success_count.fetch_add(1, Ordering::Relaxed);
        self.total_latency.fetch_add(latency, Ordering::Relaxed);
        self.last_success.store(timestamp, Ordering::Relaxed);
        
        // Update reward with exponential moving average
        let old_reward = self.reward_estimate.load();
        let new_reward = 0.9 * old_reward + 0.1 * 1.0; // Success = +1.0
        self.reward_estimate.store(new_reward);
    }
    
    fn record_failure(&self, timestamp: u64) {
        self.failure_count.fetch_add(1, Ordering::Relaxed);
        
        // Penalize reward for failures
        let old_reward = self.reward_estimate.load();
        let new_reward = 0.9 * old_reward + 0.1 * (-0.5); // Failure = -0.5
        self.reward_estimate.store(new_reward);
    }
    
    fn get_ucb_score(&self, total_spikes: u64) -> f32 {
        let successes = self.success_count.load(Ordering::Relaxed) as f32;
        let failures = self.failure_count.load(Ordering::Relaxed) as f32;
        let total_trials = successes + failures + 1.0; // +1 to avoid division by zero
        
        let reward = self.reward_estimate.load();
        let exploration = (2.0 * (total_spikes as f32).ln() / total_trials).sqrt();
        let congestion_penalty = self.congestion_score.load();
        
        reward + exploration - congestion_penalty
    }
}

/// Work-stealing queue for each worker thread
struct WorkerQueue {
    /// High-priority queue for critical spikes
    priority_queue: ArrayQueue<Spike>,
    /// Normal operation queue
    work_queue: ArrayQueue<Spike>, 
    /// Victim queues for work stealing
    steal_victims: Vec<Arc<ArrayQueue<Spike>>>,
    /// Worker ID for debugging
    worker_id: usize,
}

impl WorkerQueue {
    fn new(worker_id: usize, capacity: usize) -> Self {
        Self {
            priority_queue: ArrayQueue::new(capacity / 4), // 25% for high priority
            work_queue: ArrayQueue::new(capacity),
            steal_victims: Vec::new(),
            worker_id,
        }
    }
    
    fn add_steal_victim(&mut self, victim: Arc<ArrayQueue<Spike>>) {
        self.steal_victims.push(victim);
    }
    
    /// Try to get work, first from own queues, then steal from others
    fn try_pop(&self) -> Option<Spike> {
        // Always check priority queue first
        if let Some(spike) = self.priority_queue.pop() {
            return Some(spike);
        }
        
        // Check own work queue
        if let Some(spike) = self.work_queue.pop() {
            return Some(spike);
        }
        
        // Work stealing with exponential backoff
        let backoff = Backoff::new();
        for victim in &self.steal_victims {
            if let Some(spike) = victim.pop() {
                return Some(spike);
            }
            backoff.snooze();
        }
        
        None
    }
    
    fn push(&self, spike: Spike) -> Result<(), Spike> {
        match spike.priority {
            SpikePriority::Critical => {
                self.priority_queue.push(spike)
            }
            _ => {
                self.work_queue.push(spike)
            }
        }
    }
}

/// Main spike routing engine
pub struct SpikeEngine {
    /// Worker queues for parallel processing
    workers: Vec<Arc<WorkerQueue>>,
    /// Global spike counter for UCB calculations
    total_spikes: AtomicU64,
    /// Path performance tracking
    path_metrics: HashMap<PathId, PathMetrics>,
    /// Global timestamp counter (microseconds)
    global_time: AtomicU64,
    /// Engine shutdown flag
    shutdown: AtomicBool,
    /// Neuron refractory periods
    refractory_states: HashMap<NeuronId, AtomicU64>,
    /// Network topology for path discovery
    topology: NetworkTopology,
}

/// Network topology for intelligent routing
struct NetworkTopology {
    /// Adjacency list: neuron -> list of connected neurons
    connections: HashMap<NeuronId, Vec<(NeuronId, f32)>>, // (target, weight)
    /// All possible paths between neuron pairs
    path_cache: HashMap<(NeuronId, NeuronId), Vec<PathId>>,
}

impl NetworkTopology {
    fn new() -> Self {
        Self {
            connections: HashMap::new(),
            path_cache: HashMap::new(),
        }
    }
    
    /// Add a synaptic connection between neurons
    fn add_connection(&mut self, from: NeuronId, to: NeuronId, weight: f32) {
        self.connections
            .entry(from)
            .or_default()
            .push((to, weight));
    }
    
    /// Find all possible paths between two neurons (up to depth 5)
    fn find_paths(&mut self, from: NeuronId, to: NeuronId) -> Vec<PathId> {
        if let Some(cached) = self.path_cache.get(&(from, to)) {
            return cached.clone();
        }
        
        let mut paths = Vec::new();
        let mut visited = std::collections::HashSet::new();
        
        self.dfs_paths(from, to, &mut visited, &mut paths, 0, 5);
        
        // Generate path IDs
        let path_ids: Vec<PathId> = (0..paths.len())
            .map(|i| PathId((from.0 << 32) | (to.0 << 16) | i as u64))
            .collect();
        
        self.path_cache.insert((from, to), path_ids.clone());
        path_ids
    }
    
    fn dfs_paths(&self, current: NeuronId, target: NeuronId, 
                 visited: &mut std::collections::HashSet<NeuronId>,
                 paths: &mut Vec<Vec<NeuronId>>, 
                 depth: usize, max_depth: usize) {
        if depth > max_depth {
            return;
        }
        
        if current == target {
            paths.push(vec![current]);
            return;
        }
        
        if let Some(neighbors) = self.connections.get(&current) {
            for &(neighbor, _weight) in neighbors {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    
                    let old_len = paths.len();
                    self.dfs_paths(neighbor, target, visited, paths, depth + 1, max_depth);
                    
                    // Prepend current to all new paths found
                    for path in paths.iter_mut().skip(old_len) {
                        path.insert(0, current);
                    }
                    
                    visited.remove(&neighbor);
                }
            }
        }
    }
}

impl SpikeEngine {
    /// Create new spike engine with specified number of worker threads
    pub fn new(num_workers: usize, queue_capacity: usize) -> Self {
        let mut workers = Vec::new();
        let worker_queues: Vec<Arc<ArrayQueue<Spike>>> = (0..num_workers)
            .map(|_| Arc::new(ArrayQueue::new(queue_capacity)))
            .collect();
        
        // Create workers with work-stealing setup
        for worker_id in 0..num_workers {
            let mut worker = WorkerQueue::new(worker_id, queue_capacity);
            
            // Add all other workers as steal victims
            for (vid, victim_queue) in worker_queues.iter().enumerate() {
                if vid != worker_id {
                    worker.add_steal_victim(victim_queue.clone());
                }
            }
            
            workers.push(Arc::new(worker));
        }
        
        Self {
            workers,
            total_spikes: AtomicU64::new(0),
            path_metrics: HashMap::new(),
            global_time: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
            refractory_states: HashMap::new(),
            topology: NetworkTopology::new(),
        }
    }
    
    /// Route a spike using intelligent path selection
    pub fn route_spike(&mut self, mut spike: Spike) {
        // Assign timestamp
        spike.timestamp = self.global_time.fetch_add(1, Ordering::Relaxed);
        
        // Check if target neuron is in refractory period
        if self.is_in_refractory_period(spike.target, spike.timestamp) {
            // Find alternative path or queue for later
            if let Some(alt_path) = self.find_alternative_path(spike.source, spike.target) {
                spike.path_id = alt_path;
            } else {
                // Drop spike if no alternative (biological realism)
                return;
            }
        }
        
        // Select optimal worker using load balancing
        let worker_id = self.select_optimal_worker();
        let worker = &self.workers[worker_id];
        
        // Route the spike
        if worker.push(spike.clone()).is_err() {
            // Queue full, try other workers
            for (i, fallback_worker) in self.workers.iter().enumerate() {
                if i != worker_id && fallback_worker.push(spike.clone()).is_ok() {
                    break;
                }
            }
        }
        
        self.total_spikes.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Start the spike processing engine
    pub fn start(&mut self, num_threads: usize) {
        self.shutdown.store(false, Ordering::Relaxed);
        
        thread::scope(|s| {
            // Spawn worker threads
            for thread_id in 0..num_threads {
                let worker = self.workers[thread_id % self.workers.len()].clone();
                let shutdown = &self.shutdown;
                
                s.spawn(move |_| {
                    let mut processed = 0u64;
                    let start_time = Instant::now();
                    
                    while !shutdown.load(Ordering::Relaxed) {
                        if let Some(spike) = worker.try_pop() {
                            self.process_spike(spike);
                            processed += 1;
                            
                            // Performance monitoring every 10K spikes
                            if processed % 10_000 == 0 {
                                let elapsed = start_time.elapsed();
                                let throughput = processed as f64 / elapsed.as_secs_f64();
                                println!("Worker {}: Processed {}K spikes, {:.0} spikes/sec", 
                                        thread_id, processed / 1000, throughput);
                            }
                        } else {
                            // No work available, brief backoff
                            std::thread::sleep(Duration::from_nanos(100));
                        }
                    }
                });
            }
        }).unwrap();
    }
    
    /// Process a single spike with timing and error handling
    fn process_spike(&self, spike: Spike) {
        let start_time = Instant::now();
        
        // Update refractory state
        if let Some(refractory) = self.refractory_states.get(&spike.target) {
            refractory.store(spike.timestamp + spike.refractory_period, Ordering::Relaxed);
        }
        
        // Simulate neural computation (replace with actual neuron logic)
        let computation_result = self.execute_neuron_computation(&spike);
        
        let latency = start_time.elapsed().as_micros() as u64;
        
        // Update path performance metrics
        if let Some(metrics) = self.path_metrics.get(&spike.path_id) {
            if computation_result.is_ok() {
                metrics.record_success(latency, spike.timestamp);
            } else {
                metrics.record_failure(spike.timestamp);
            }
        }
    }
    
    fn execute_neuron_computation(&self, _spike: &Spike) -> Result<f32, &'static str> {
        // Mock neural computation - replace with actual implementation
        std::thread::sleep(Duration::from_nanos(1000)); // 1μs computation time
        Ok(0.7) // Mock activation value
    }
    
    fn is_in_refractory_period(&self, neuron: NeuronId, current_time: u64) -> bool {
        self.refractory_states
            .get(&neuron)
            .map(|refractory_until| refractory_until.load(Ordering::Relaxed) > current_time)
            .unwrap_or(false)
    }
    
    fn find_alternative_path(&mut self, source: NeuronId, target: NeuronId) -> Option<PathId> {
        let available_paths = self.topology.find_paths(source, target);
        
        if available_paths.is_empty() {
            return None;
        }
        
        // Use UCB1 to select best alternative path
        let total_spikes = self.total_spikes.load(Ordering::Relaxed);
        let mut best_path = available_paths[0];
        let mut best_score = f32::NEG_INFINITY;
        
        for &path in &available_paths {
            let score = self.path_metrics
                .get(&path)
                .map(|m| m.get_ucb_score(total_spikes))
                .unwrap_or(1.0); // Optimistic for new paths
                
            if score > best_score {
                best_score = score;
                best_path = path;
            }
        }
        
        Some(best_path)
    }
    
    fn select_optimal_worker(&self) -> usize {
        // Simple round-robin for now - could be improved with load monitoring
        (self.total_spikes.load(Ordering::Relaxed) % self.workers.len() as u64) as usize
    }
    
    /// Shutdown the engine gracefully
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        
        // Wait for all queues to drain
        std::thread::sleep(Duration::from_millis(100));
        
        // Print final statistics
        let total = self.total_spikes.load(Ordering::Relaxed);
        println!("Spike engine shutdown. Total spikes processed: {}", total);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spike_routing() {
        let mut engine = SpikeEngine::new(4, 1000);
        
        let spike = Spike {
            id: 1,
            source: NeuronId(1),
            target: NeuronId(2),
            weight: 0.8,
            timestamp: 0,
            path_id: PathId(1),
            priority: SpikePriority::Normal,
            refractory_period: 100,
        };
        
        engine.route_spike(spike);
        // Test passes if no panic
    }
    
    #[test]
    fn test_path_metrics() {
        let metrics = PathMetrics::new();
        
        metrics.record_success(500, 1000);
        metrics.record_failure(1100);
        
        let score = metrics.get_ucb_score(100);
        assert!(score > 0.0);
    }
    
    #[test]
    fn test_topology_path_finding() {
        let mut topology = NetworkTopology::new();
        
        topology.add_connection(NeuronId(1), NeuronId(2), 0.8);
        topology.add_connection(NeuronId(2), NeuronId(3), 0.7);
        topology.add_connection(NeuronId(1), NeuronId(3), 0.5); // Direct path
        
        let paths = topology.find_paths(NeuronId(1), NeuronId(3));
        assert!(paths.len() >= 1); // Should find at least the direct path
    }
}