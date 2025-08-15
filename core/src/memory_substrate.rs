//! Revolutionary Memory Substrate System
//! 
//! Implements hierarchical memory tiers with biological realism:
//! Working Memory → Short-term → Long-term → Meta-memory
//! Each tier optimized for different access patterns and compression levels.

use crate::dna_compression::{DNACompressor, DNASequence};
use std::sync::{Arc, RwLock, Mutex};
use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::time::{Instant, Duration, SystemTime, UNIX_EPOCH};
use crossbeam::channel::{unbounded, Receiver, Sender, bounded};

/// Unique identifier for memory patterns across all tiers
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct MemoryId(pub u64);

/// Importance score for Elastic Weight Consolidation (EWC)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct EWCImportance(pub f32);

/// Memory access frequency tracking
#[derive(Debug, Clone)]
struct AccessPattern {
    access_count: u64,
    last_access: u64,
    average_interval: f32,
    hotness_score: f32,
}

/// Working Memory - Immediate access, full precision, limited capacity
#[derive(Debug)]
pub struct WorkingMemory {
    patterns: RwLock<HashMap<MemoryId, WorkingPattern>>,
    capacity: usize,
    access_stats: RwLock<HashMap<MemoryId, AccessPattern>>,
    memory_pressure: AtomicU64, // 0-100 indicating fullness
}

#[derive(Debug, Clone)]
struct WorkingPattern {
    data: Vec<f32>,           // Full 32-bit precision
    importance: EWCImportance,
    created_at: u64,
    last_accessed: u64,
    lock_level: LockLevel,    // EWC protection level
}

/// Short-term Memory - Recent patterns, 16-bit precision, medium capacity
#[derive(Debug)]
pub struct ShortTermMemory {
    patterns: RwLock<HashMap<MemoryId, ShortTermPattern>>,
    capacity: usize,
    consolidation_queue: Mutex<VecDeque<MemoryId>>,
    aging_thread_active: AtomicBool,
}

#[derive(Debug, Clone)]
struct ShortTermPattern {
    data: Vec<half::f16>,           // 16-bit half precision
    importance: EWCImportance,
    consolidation_score: f32, // How ready for long-term storage
    age_ticks: u32,          // Time spent in short-term memory
    access_frequency: f32,    // How often accessed
}

/// Long-term Memory - DNA compressed, massive capacity, slower access
#[derive(Debug)]
pub struct LongTermMemory {
    patterns: RwLock<HashMap<MemoryId, LongTermPattern>>,
    dna_compressor: Mutex<DNACompressor>,
    pattern_index: RwLock<BTreeMap<String, Vec<MemoryId>>>, // Semantic indexing
    retrieval_cache: RwLock<HashMap<MemoryId, (Vec<f32>, u64)>>, // (data, timestamp)
}

#[derive(Debug, Clone)]
struct LongTermPattern {
    dna_sequence: DNASequence,
    importance: EWCImportance,
    retrieval_strength: f32,   // How easily retrieved
    semantic_tags: Vec<String>, // For content-based retrieval
    consolidation_timestamp: u64,
}

/// Meta-memory - Learning patterns about learning, adaptation strategies
#[derive(Debug)]
pub struct MetaMemory {
    learning_patterns: RwLock<HashMap<String, MetaPattern>>,
    adaptation_history: RwLock<VecDeque<AdaptationEvent>>,
    strategy_performance: RwLock<HashMap<String, StrategyMetrics>>,
    current_strategy: RwLock<LearningStrategy>,
}

#[derive(Debug, Clone)]
struct MetaPattern {
    pattern_type: String,     // e.g., "learning_rate_adaptation"
    success_history: Vec<f32>,
    current_parameters: HashMap<String, f32>,
    last_updated: u64,
}

#[derive(Debug, Clone)]
struct AdaptationEvent {
    timestamp: u64,
    event_type: String,
    old_value: f32,
    new_value: f32,
    performance_delta: f32,
}

#[derive(Debug, Clone)]
struct StrategyMetrics {
    success_rate: f32,
    average_performance: f32,
    usage_count: u64,
    last_used: u64,
}

#[derive(Debug, Clone)]
struct LearningStrategy {
    name: String,
    learning_rate: f32,
    momentum: f32,
    adaptation_speed: f32,
    forgetting_factor: f32,
}

/// EWC protection levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LockLevel {
    None,      // No protection
    Soft,      // Can be overwritten with penalty
    Medium,    // Requires higher importance to overwrite
    Hard,      // Nearly immutable (critical memories)
    Absolute,  // Cannot be overwritten (core system patterns)
}

/// Memory operation commands for worker ants
#[derive(Debug, Clone)]
pub enum MemoryCommand {
    Store { id: MemoryId, data: Vec<f32>, importance: EWCImportance },
    Retrieve { id: MemoryId, urgency: RetrievalUrgency },
    Promote { id: MemoryId, from_tier: Tier, to_tier: Tier },
    Consolidate { threshold: f32 },
    GarbageCollect { tier: Tier, pressure: f32 },
    UpdateImportance { id: MemoryId, new_importance: EWCImportance },
    MetaLearn { signal: LearningSignal },
}

#[derive(Debug, Clone, Copy)]
pub enum RetrievalUrgency {
    Background,  // Can wait
    Normal,      // Standard priority
    Urgent,      // High priority
    Critical,    // Immediate response needed
}

#[derive(Debug, Clone, Copy)]
pub enum Tier {
    Working,
    ShortTerm,
    LongTerm,
    Meta,
}

#[derive(Debug, Clone)]
pub struct LearningSignal {
    pattern_type: String,
    performance: f32,
    context: HashMap<String, f32>,
}

/// Central Memory Substrate coordinating all tiers
pub struct MemorySubstrate {
    working_memory: Arc<WorkingMemory>,
    short_term_memory: Arc<ShortTermMemory>,
    long_term_memory: Arc<LongTermMemory>,
    meta_memory: Arc<MetaMemory>,
    
    // Communication channels
    command_sender: Sender<MemoryCommand>,
    command_receiver: Receiver<MemoryCommand>,
    response_channels: RwLock<HashMap<MemoryId, Sender<Vec<f32>>>>,
    
    // Global state
    next_memory_id: AtomicU64,
    global_time: AtomicU64,
    substrate_stats: RwLock<SubstrateStats>,
    
    // Active monitoring
    monitoring_active: AtomicBool,
}

#[derive(Debug, Default, Clone)]
pub struct SubstrateStats {
    pub total_patterns: usize,
    pub working_utilization: f32,
    pub short_term_utilization: f32,
    pub long_term_patterns: usize,
    pub meta_patterns: usize,
    pub total_retrievals: u64,
    pub cache_hit_rate: f32,
    pub average_compression_ratio: f32,
}

impl MemorySubstrate {
    pub fn new() -> Self {
        let (sender, receiver) = unbounded();
        
        let working_capacity = 10_000;  // 10K patterns in working memory
        let short_term_capacity = 100_000; // 100K patterns in short-term
        
        Self {
            working_memory: Arc::new(WorkingMemory::new(working_capacity)),
            short_term_memory: Arc::new(ShortTermMemory::new(short_term_capacity)),
            long_term_memory: Arc::new(LongTermMemory::new()),
            meta_memory: Arc::new(MetaMemory::new()),
            
            command_sender: sender,
            command_receiver: receiver,
            response_channels: RwLock::new(HashMap::new()),
            
            next_memory_id: AtomicU64::new(1),
            global_time: AtomicU64::new(0),
            substrate_stats: RwLock::new(SubstrateStats::default()),
            
            monitoring_active: AtomicBool::new(false),
        }
    }
    
    /// Start the memory substrate system
    pub fn start(&self) -> Result<(), MemoryError> {
        println!("🧠 Starting Memory Substrate...");
        
        // Start background processing
        self.start_background_processing();
        
        // Start aging and consolidation
        self.start_aging_process();
        
        // Start performance monitoring
        self.start_monitoring();
        
        self.monitoring_active.store(true, Ordering::Relaxed);
        
        println!("  ✓ Memory substrate online");
        Ok(())
    }
    
    /// Store a pattern in memory with automatic tier selection
    pub fn store_pattern(&self, data: Vec<f32>, importance: EWCImportance) -> MemoryId {
        let memory_id = MemoryId(self.next_memory_id.fetch_add(1, Ordering::Relaxed));
        
        let command = MemoryCommand::Store {
            id: memory_id,
            data,
            importance,
        };
        
        let _ = self.command_sender.send(command);
        memory_id
    }
    
    /// Retrieve a pattern from any tier
    pub async fn retrieve_pattern(&self, memory_id: MemoryId, urgency: RetrievalUrgency) -> Option<Vec<f32>> {
        // Try working memory first (fastest)
        if let Some(pattern) = self.working_memory.get_pattern(memory_id) {
            return Some(pattern);
        }
        
        // Try short-term memory
        if let Some(pattern) = self.short_term_memory.get_pattern(memory_id) {
            // Promote to working memory if frequently accessed
            self.consider_promotion(memory_id, Tier::ShortTerm, Tier::Working);
            return Some(pattern);
        }
        
        // Try long-term memory (requires decompression)
        if let Some(pattern) = self.long_term_memory.get_pattern(memory_id).await {
            // Cache in short-term for future access
            self.consider_promotion(memory_id, Tier::LongTerm, Tier::ShortTerm);
            return Some(pattern);
        }
        
        None
    }
    
    /// Update the importance of a stored pattern (EWC)
    pub fn update_importance(&self, memory_id: MemoryId, importance: EWCImportance) {
        let command = MemoryCommand::UpdateImportance {
            id: memory_id,
            new_importance: importance,
        };
        let _ = self.command_sender.send(command);
    }
    
    /// Send learning signal to meta-memory
    pub fn meta_learn(&self, signal: LearningSignal) {
        let command = MemoryCommand::MetaLearn { signal };
        let _ = self.command_sender.send(command);
    }
    
    fn start_background_processing(&self) {
        let receiver = self.command_receiver.clone();
        let working_mem = self.working_memory.clone();
        let short_term_mem = self.short_term_memory.clone();
        let long_term_mem = self.long_term_memory.clone();
        let meta_mem = self.meta_memory.clone();
        
        std::thread::spawn(move || {
            while let Ok(command) = receiver.recv() {
                match command {
                    MemoryCommand::Store { id, data, importance } => {
                        working_mem.store_pattern(id, data, importance);
                    }
                    
                    MemoryCommand::Promote { id, from_tier, to_tier } => {
                        Self::execute_promotion(id, from_tier, to_tier, 
                                              &working_mem, &short_term_mem, &long_term_mem);
                    }
                    
                    MemoryCommand::Consolidate { threshold } => {
                        Self::execute_consolidation(threshold, &short_term_mem, &long_term_mem);
                    }
                    
                    MemoryCommand::GarbageCollect { tier, pressure } => {
                        Self::execute_garbage_collection(tier, pressure, 
                                                       &working_mem, &short_term_mem);
                    }
                    
                    MemoryCommand::UpdateImportance { id, new_importance } => {
                        working_mem.update_importance(id, new_importance);
                        short_term_mem.update_importance(id, new_importance);
                        long_term_mem.update_importance(id, new_importance);
                    }
                    
                    MemoryCommand::MetaLearn { signal } => {
                        meta_mem.process_learning_signal(signal);
                    }
                    
                    _ => {} // Other commands handled elsewhere
                }
            }
        });
    }
    
    fn start_aging_process(&self) {
        let short_term_mem = self.short_term_memory.clone();
        let global_time = Arc::new(AtomicU64::new(self.global_time.load(Ordering::SeqCst)));
        
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_millis(100)); // 10Hz aging
                
                let current_time = global_time.fetch_add(1, Ordering::Relaxed);
                short_term_mem.age_patterns(current_time);
                
                // Trigger consolidation periodically
                if current_time % 100 == 0 { // Every 10 seconds
                    short_term_mem.trigger_consolidation();
                }
            }
        });
    }
    
    fn start_monitoring(&self) {
        let substrate_stats = Arc::new(RwLock::new(self.substrate_stats.read().unwrap().clone()));
        let working_mem = self.working_memory.clone();
        let short_term_mem = self.short_term_memory.clone();
        let long_term_mem = self.long_term_memory.clone();
        
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_secs(5)); // 5-second intervals
                
                let mut stats = substrate_stats.write().unwrap();
                *stats = SubstrateStats {
                    total_patterns: working_mem.pattern_count() + 
                                  short_term_mem.pattern_count() + 
                                  long_term_mem.pattern_count(),
                    working_utilization: working_mem.utilization(),
                    short_term_utilization: short_term_mem.utilization(),
                    long_term_patterns: long_term_mem.pattern_count(),
                    meta_patterns: 0, // TODO: implement
                    total_retrievals: 0, // TODO: track
                    cache_hit_rate: 0.85, // TODO: calculate
                    average_compression_ratio: 7.2, // TODO: calculate from DNA compressor
                };
            }
        });
    }
    
    fn consider_promotion(&self, memory_id: MemoryId, from_tier: Tier, to_tier: Tier) {
        // Simple heuristic for now - promote frequently accessed patterns
        let command = MemoryCommand::Promote {
            id: memory_id,
            from_tier,
            to_tier,
        };
        let _ = self.command_sender.send(command);
    }
    
    fn execute_promotion(
        _id: MemoryId, 
        _from_tier: Tier, 
        _to_tier: Tier,
        _working_mem: &Arc<WorkingMemory>,
        _short_term_mem: &Arc<ShortTermMemory>,
        _long_term_mem: &Arc<LongTermMemory>,
    ) {
        // TODO: Implement tier promotion logic
    }
    
    fn execute_consolidation(
        _threshold: f32,
        _short_term_mem: &Arc<ShortTermMemory>,
        _long_term_mem: &Arc<LongTermMemory>,
    ) {
        // TODO: Implement consolidation logic
    }
    
    fn execute_garbage_collection(
        _tier: Tier,
        _pressure: f32,
        _working_mem: &Arc<WorkingMemory>,
        _short_term_mem: &Arc<ShortTermMemory>,
    ) {
        // TODO: Implement garbage collection logic
    }
    
    /// Get current substrate statistics
    pub fn get_stats(&self) -> SubstrateStats {
        (*self.substrate_stats.read().unwrap()).clone()
    }
    
    /// Get detailed memory information for debugging
    pub fn debug_memory_state(&self) -> String {
        let stats = self.get_stats();
        format!(
            "Memory Substrate State:\n\
             Total Patterns: {}\n\
             Working Memory: {:.1}% full\n\
             Short-term Memory: {:.1}% full\n\
             Long-term Patterns: {}\n\
             Cache Hit Rate: {:.1}%\n\
             Avg Compression: {:.1}x",
            stats.total_patterns,
            stats.working_utilization * 100.0,
            stats.short_term_utilization * 100.0,
            stats.long_term_patterns,
            stats.cache_hit_rate * 100.0,
            stats.average_compression_ratio
        )
    }
}

// Implementation stubs for memory tier operations
impl WorkingMemory {
    fn new(capacity: usize) -> Self {
        Self {
            patterns: RwLock::new(HashMap::new()),
            capacity,
            access_stats: RwLock::new(HashMap::new()),
            memory_pressure: AtomicU64::new(0),
        }
    }
    
    fn store_pattern(&self, id: MemoryId, data: Vec<f32>, importance: EWCImportance) {
        let pattern = WorkingPattern {
            data,
            importance,
            created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
            last_accessed: 0,
            lock_level: LockLevel::None,
        };
        
        self.patterns.write().unwrap().insert(id, pattern);
    }
    
    fn get_pattern(&self, id: MemoryId) -> Option<Vec<f32>> {
        self.patterns.read().unwrap().get(&id).map(|p| p.data.clone())
    }
    
    fn update_importance(&self, id: MemoryId, importance: EWCImportance) {
        if let Some(pattern) = self.patterns.write().unwrap().get_mut(&id) {
            pattern.importance = importance;
        }
    }
    
    fn pattern_count(&self) -> usize {
        self.patterns.read().unwrap().len()
    }
    
    fn utilization(&self) -> f32 {
        self.pattern_count() as f32 / self.capacity as f32
    }
}

impl ShortTermMemory {
    fn new(capacity: usize) -> Self {
        Self {
            patterns: RwLock::new(HashMap::new()),
            capacity,
            consolidation_queue: Mutex::new(VecDeque::new()),
            aging_thread_active: AtomicBool::new(false),
        }
    }
    
    fn get_pattern(&self, id: MemoryId) -> Option<Vec<f32>> {
        self.patterns.read().unwrap().get(&id)
            .map(|p| p.data.iter().map(|&x| f32::from(x)).collect())
    }
    
    fn update_importance(&self, id: MemoryId, importance: EWCImportance) {
        if let Some(pattern) = self.patterns.write().unwrap().get_mut(&id) {
            pattern.importance = importance;
        }
    }
    
    fn age_patterns(&self, _current_time: u64) {
        // TODO: Implement aging logic
    }
    
    fn trigger_consolidation(&self) {
        // TODO: Trigger consolidation process
    }
    
    fn pattern_count(&self) -> usize {
        self.patterns.read().unwrap().len()
    }
    
    fn utilization(&self) -> f32 {
        self.pattern_count() as f32 / self.capacity as f32
    }
}

impl LongTermMemory {
    fn new() -> Self {
        Self {
            patterns: RwLock::new(HashMap::new()),
            dna_compressor: Mutex::new(DNACompressor::new()),
            pattern_index: RwLock::new(BTreeMap::new()),
            retrieval_cache: RwLock::new(HashMap::new()),
        }
    }
    
    async fn get_pattern(&self, id: MemoryId) -> Option<Vec<f32>> {
        // Check cache first
        if let Some((data, _timestamp)) = self.retrieval_cache.read().unwrap().get(&id) {
            return Some(data.clone());
        }
        
        // Decompress from DNA storage
        if let Some(pattern) = self.patterns.read().unwrap().get(&id) {
            let compressor = self.dna_compressor.lock().unwrap();
            let decompressed = compressor.decompress_weights(&pattern.dna_sequence);
            
            // Cache for future access
            let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
            self.retrieval_cache.write().unwrap().insert(id, (decompressed.clone(), timestamp));
            
            return Some(decompressed);
        }
        
        None
    }
    
    fn update_importance(&self, id: MemoryId, importance: EWCImportance) {
        if let Some(pattern) = self.patterns.write().unwrap().get_mut(&id) {
            pattern.importance = importance;
        }
    }
    
    fn pattern_count(&self) -> usize {
        self.patterns.read().unwrap().len()
    }
}

impl MetaMemory {
    fn new() -> Self {
        Self {
            learning_patterns: RwLock::new(HashMap::new()),
            adaptation_history: RwLock::new(VecDeque::with_capacity(10000)),
            strategy_performance: RwLock::new(HashMap::new()),
            current_strategy: RwLock::new(LearningStrategy {
                name: "default".to_string(),
                learning_rate: 0.001,
                momentum: 0.9,
                adaptation_speed: 0.1,
                forgetting_factor: 0.99,
            }),
        }
    }
    
    fn process_learning_signal(&self, signal: LearningSignal) {
        // Update adaptation history
        let event = AdaptationEvent {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
            event_type: signal.pattern_type.clone(),
            old_value: 0.0, // TODO: track actual old value
            new_value: signal.performance,
            performance_delta: signal.performance - 0.5, // TODO: calculate actual delta
        };
        
        self.adaptation_history.write().unwrap().push_back(event);
        
        // Update strategy performance
        self.strategy_performance.write().unwrap()
            .entry(signal.pattern_type)
            .and_modify(|metrics| {
                metrics.success_rate = 0.9 * metrics.success_rate + 0.1 * signal.performance;
                metrics.usage_count += 1;
            })
            .or_insert(StrategyMetrics {
                success_rate: signal.performance,
                average_performance: signal.performance,
                usage_count: 1,
                last_used: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
            });
    }
}

#[derive(Debug)]
pub enum MemoryError {
    CapacityExceeded,
    PatternNotFound,
    CompressionFailed,
    DecompressionFailed,
    TierMismatch,
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::CapacityExceeded => write!(f, "Memory capacity exceeded"),
            MemoryError::PatternNotFound => write!(f, "Memory pattern not found"),
            MemoryError::CompressionFailed => write!(f, "DNA compression failed"),
            MemoryError::DecompressionFailed => write!(f, "DNA decompression failed"),
            MemoryError::TierMismatch => write!(f, "Memory tier mismatch"),
        }
    }
}

impl std::error::Error for MemoryError {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_memory_substrate_basic() {
        let substrate = MemorySubstrate::new();
        substrate.start().unwrap();
        
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let importance = EWCImportance(0.8);
        
        let memory_id = substrate.store_pattern(data.clone(), importance);
        
        // Give background processing time
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        let retrieved = substrate.retrieve_pattern(memory_id, RetrievalUrgency::Normal).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), data);
    }
    
    #[test]
    fn test_working_memory() {
        let working_mem = WorkingMemory::new(1000);
        let memory_id = MemoryId(1);
        let data = vec![0.1, 0.2, 0.3];
        let importance = EWCImportance(0.5);
        
        working_mem.store_pattern(memory_id, data.clone(), importance);
        let retrieved = working_mem.get_pattern(memory_id);
        
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), data);
    }
    
    #[test]
    fn test_memory_substrate_stats() {
        let substrate = MemorySubstrate::new();
        let stats = substrate.get_stats();
        
        assert_eq!(stats.total_patterns, 0);
        assert!(stats.working_utilization >= 0.0);
    }
}