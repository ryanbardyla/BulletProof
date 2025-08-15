//! Biological Memory Hierarchy with Worker Ant System
//! 
//! Mimics the brain's memory architecture with specialized "worker ants" 
//! managing memory transfers between tiers, EWC protection, and meta-learning.

use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};
use std::collections::{HashMap, VecDeque, BinaryHeap};
use std::time::{Instant, Duration};
use crossbeam::thread;
use crossbeam::channel::{unbounded, Receiver, Sender};

/// Memory importance score for EWC protection
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ImportanceScore(pub f32);

/// Unique identifier for memory patterns
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PatternId(pub u64);

/// Different types of memory with biological inspiration
#[derive(Debug, Clone)]
pub enum MemoryType {
    /// Working memory - immediately accessible, limited capacity
    Working {
        pattern: Vec<f32>,
        importance: ImportanceScore,
        access_count: u32,
        last_access: u64,
    },
    /// Short-term memory - recent experiences, moderate capacity  
    ShortTerm {
        pattern: Vec<f32>,
        importance: ImportanceScore,
        consolidation_score: f32,
        age_ticks: u32,
    },
    /// Long-term memory - DNA compressed, massive capacity
    LongTerm {
        dna_sequence: Vec<u8>, // 4-bit DNA encoding
        importance: ImportanceScore,
        retrieval_strength: f32,
        ewc_protection: f32,
    },
    /// Meta-memory - patterns about learning patterns
    Meta {
        learning_pattern: Vec<f32>,
        adaptation_history: Vec<f32>,
        generalization_strength: f32,
    },
}

/// Worker ant specialized for different memory operations
#[derive(Debug)]
pub struct WorkerAnt {
    pub id: usize,
    pub specialty: AntSpecialty,
    pub energy: AtomicU64,        // Energy for operations
    pub efficiency: f32,          // How good at their job
    pub memory_moved: AtomicU64,  // Performance tracking
    pub active: AtomicBool,       // Currently working
}

#[derive(Debug, Clone)]
pub enum AntSpecialty {
    /// Moves frequently accessed patterns to working memory
    HotDataMover { threshold_accesses: u32 },
    /// Consolidates short-term to long-term memory
    Consolidator { consolidation_threshold: f32 },
    /// Applies DNA compression to reduce memory footprint
    Compressor { compression_ratio: f32 },
    /// Protects important memories using EWC
    EWCGuard { protection_threshold: f32 },
    /// Manages meta-learning patterns
    MetaLearner { adaptation_rate: f32 },
    /// Cleans up unused memories
    GarbageCollector { cleanup_threshold: u32 },
}

/// Central memory management system
pub struct BrainMemory {
    /// Memory storage tiers
    working_memory: Arc<RwLock<HashMap<PatternId, MemoryType>>>,
    short_term_memory: Arc<RwLock<HashMap<PatternId, MemoryType>>>,
    long_term_memory: Arc<RwLock<HashMap<PatternId, MemoryType>>>,
    meta_memory: Arc<RwLock<HashMap<PatternId, MemoryType>>>,
    
    /// Worker ant colony
    worker_ants: Vec<Arc<WorkerAnt>>,
    
    /// Communication channels for ants
    work_queue: (Sender<MemoryOperation>, Receiver<MemoryOperation>),
    
    /// Memory statistics
    total_patterns: AtomicUsize,
    memory_efficiency: AtomicU64, // Percentage utilization
    consolidation_rate: AtomicU64,
    
    /// Global memory clock for aging
    memory_time: AtomicU64,
    
    /// EWC importance tracking
    ewc_importance: Arc<RwLock<HashMap<PatternId, f32>>>,
    
    /// Meta-learning state
    meta_learning_state: Arc<RwLock<MetaLearningState>>,
}

#[derive(Debug)]
pub enum MemoryOperation {
    Store { pattern_id: PatternId, memory: MemoryType },
    Retrieve { pattern_id: PatternId },
    Consolidate { from_tier: MemoryTier, to_tier: MemoryTier },
    Compress { pattern_id: PatternId },
    ApplyEWC { pattern_id: PatternId, importance: f32 },
    MetaLearn { pattern_id: PatternId, learning_signal: f32 },
    GarbageCollect { tier: MemoryTier },
}

#[derive(Debug, Clone, Copy)]
pub enum MemoryTier {
    Working,
    ShortTerm,
    LongTerm,
    Meta,
}

/// Meta-learning state tracking adaptation patterns
#[derive(Debug)]
pub struct MetaLearningState {
    learning_rate_history: VecDeque<f32>,
    adaptation_patterns: HashMap<String, Vec<f32>>,
    generalization_performance: f32,
    catastrophic_forgetting_resistance: f32,
}

impl BrainMemory {
    pub fn new(num_worker_ants: usize) -> Self {
        let (sender, receiver) = unbounded();
        
        // Create specialized worker ants
        let mut worker_ants = Vec::new();
        for i in 0..num_worker_ants {
            let specialty = match i % 6 {
                0 => AntSpecialty::HotDataMover { threshold_accesses: 10 },
                1 => AntSpecialty::Consolidator { consolidation_threshold: 0.8 },
                2 => AntSpecialty::Compressor { compression_ratio: 8.0 },
                3 => AntSpecialty::EWCGuard { protection_threshold: 0.9 },
                4 => AntSpecialty::MetaLearner { adaptation_rate: 0.01 },
                5 => AntSpecialty::GarbageCollector { cleanup_threshold: 100 },
                _ => AntSpecialty::HotDataMover { threshold_accesses: 10 },
            };
            
            worker_ants.push(Arc::new(WorkerAnt {
                id: i,
                specialty,
                energy: AtomicU64::new(1000), // Start with full energy
                efficiency: 0.8 + (i as f32 * 0.02), // Slight variation
                memory_moved: AtomicU64::new(0),
                active: AtomicBool::new(false),
            }));
        }
        
        Self {
            working_memory: Arc::new(RwLock::new(HashMap::new())),
            short_term_memory: Arc::new(RwLock::new(HashMap::new())),
            long_term_memory: Arc::new(RwLock::new(HashMap::new())),
            meta_memory: Arc::new(RwLock::new(HashMap::new())),
            worker_ants,
            work_queue: (sender, receiver),
            total_patterns: AtomicUsize::new(0),
            memory_efficiency: AtomicU64::new(0),
            consolidation_rate: AtomicU64::new(0),
            memory_time: AtomicU64::new(0),
            ewc_importance: Arc::new(RwLock::new(HashMap::new())),
            meta_learning_state: Arc::new(RwLock::new(MetaLearningState {
                learning_rate_history: VecDeque::with_capacity(1000),
                adaptation_patterns: HashMap::new(),
                generalization_performance: 0.5,
                catastrophic_forgetting_resistance: 0.3,
            })),
        }
    }
    
    /// Start the worker ant colony
    pub fn start_worker_colony(&self) {
        let receiver = self.work_queue.1.clone();
        
        for worker_ant in &self.worker_ants {
            let ant = worker_ant.clone();
            let work_receiver = receiver.clone();
            let working_mem = self.working_memory.clone();
            let short_term_mem = self.short_term_memory.clone();
            let long_term_mem = self.long_term_memory.clone();
            let meta_mem = self.meta_memory.clone();
            let ewc_importance = self.ewc_importance.clone();
            let meta_state = self.meta_learning_state.clone();
            
            thread::spawn(move || {
                Self::worker_ant_loop(
                    ant,
                    work_receiver,
                    working_mem,
                    short_term_mem,
                    long_term_mem,
                    meta_mem,
                    ewc_importance,
                    meta_state,
                );
            });
        }
        
        // Start memory maintenance thread
        self.start_memory_maintenance();
    }
    
    fn worker_ant_loop(
        ant: Arc<WorkerAnt>,
        receiver: Receiver<MemoryOperation>,
        working_memory: Arc<RwLock<HashMap<PatternId, MemoryType>>>,
        short_term_memory: Arc<RwLock<HashMap<PatternId, MemoryType>>>,
        long_term_memory: Arc<RwLock<HashMap<PatternId, MemoryType>>>,
        meta_memory: Arc<RwLock<HashMap<PatternId, MemoryType>>>,
        ewc_importance: Arc<RwLock<HashMap<PatternId, f32>>>,
        meta_state: Arc<RwLock<MetaLearningState>>,
    ) {
        println!("🐜 Worker ant {} started with specialty: {:?}", ant.id, ant.specialty);
        
        loop {
            // Check energy level
            let energy = ant.energy.load(Ordering::Relaxed);
            if energy < 100 {
                // Rest to recover energy
                std::thread::sleep(Duration::from_millis(10));
                ant.energy.store((energy + 50).min(1000), Ordering::Relaxed);
                continue;
            }
            
            // Try to get work
            if let Ok(operation) = receiver.try_recv() {
                ant.active.store(true, Ordering::Relaxed);
                
                let start_time = Instant::now();
                let should_handle = Self::ant_should_handle_operation(&ant.specialty, &operation);
                
                if should_handle {
                    Self::execute_memory_operation(
                        &ant,
                        operation,
                        &working_memory,
                        &short_term_memory,
                        &long_term_memory,
                        &meta_memory,
                        &ewc_importance,
                        &meta_state,
                    );
                    
                    let operation_time = start_time.elapsed();
                    let energy_cost = (operation_time.as_millis() as u64).min(50);
                    ant.energy.fetch_sub(energy_cost, Ordering::Relaxed);
                    ant.memory_moved.fetch_add(1, Ordering::Relaxed);
                }
                
                ant.active.store(false, Ordering::Relaxed);
            } else {
                // No work available, brief sleep
                std::thread::sleep(Duration::from_micros(100));
            }
        }
    }
    
    fn ant_should_handle_operation(specialty: &AntSpecialty, operation: &MemoryOperation) -> bool {
        match (specialty, operation) {
            (AntSpecialty::HotDataMover { .. }, MemoryOperation::Store { .. }) => true,
            (AntSpecialty::HotDataMover { .. }, MemoryOperation::Retrieve { .. }) => true,
            (AntSpecialty::Consolidator { .. }, MemoryOperation::Consolidate { .. }) => true,
            (AntSpecialty::Compressor { .. }, MemoryOperation::Compress { .. }) => true,
            (AntSpecialty::EWCGuard { .. }, MemoryOperation::ApplyEWC { .. }) => true,
            (AntSpecialty::MetaLearner { .. }, MemoryOperation::MetaLearn { .. }) => true,
            (AntSpecialty::GarbageCollector { .. }, MemoryOperation::GarbageCollect { .. }) => true,
            _ => false, // Ant not specialized for this operation
        }
    }
    
    fn execute_memory_operation(
        ant: &Arc<WorkerAnt>,
        operation: MemoryOperation,
        working_memory: &Arc<RwLock<HashMap<PatternId, MemoryType>>>,
        short_term_memory: &Arc<RwLock<HashMap<PatternId, MemoryType>>>,
        long_term_memory: &Arc<RwLock<HashMap<PatternId, MemoryType>>>,
        meta_memory: &Arc<RwLock<HashMap<PatternId, MemoryType>>>,
        ewc_importance: &Arc<RwLock<HashMap<PatternId, f32>>>,
        meta_state: &Arc<RwLock<MetaLearningState>>,
    ) {
        match operation {
            MemoryOperation::Store { pattern_id, memory } => {
                Self::store_memory_pattern(pattern_id, memory, working_memory);
            }
            
            MemoryOperation::Consolidate { from_tier, to_tier } => {
                Self::consolidate_memories(from_tier, to_tier, 
                    working_memory, short_term_memory, long_term_memory);
            }
            
            MemoryOperation::Compress { pattern_id } => {
                Self::compress_memory_pattern(pattern_id, long_term_memory);
            }
            
            MemoryOperation::ApplyEWC { pattern_id, importance } => {
                Self::apply_ewc_protection(pattern_id, importance, ewc_importance);
            }
            
            MemoryOperation::MetaLearn { pattern_id, learning_signal } => {
                Self::update_meta_learning(pattern_id, learning_signal, meta_state, meta_memory);
            }
            
            MemoryOperation::GarbageCollect { tier } => {
                Self::garbage_collect_tier(tier, working_memory, short_term_memory, long_term_memory);
            }
            
            _ => {
                // Operation handled by different ant specialty
            }
        }
    }
    
    fn store_memory_pattern(
        pattern_id: PatternId,
        memory: MemoryType,
        working_memory: &Arc<RwLock<HashMap<PatternId, MemoryType>>>,
    ) {
        if let Ok(mut mem) = working_memory.write() {
            mem.insert(pattern_id, memory);
        }
    }
    
    fn consolidate_memories(
        from_tier: MemoryTier,
        to_tier: MemoryTier,
        working_memory: &Arc<RwLock<HashMap<PatternId, MemoryType>>>,
        short_term_memory: &Arc<RwLock<HashMap<PatternId, MemoryType>>>,
        long_term_memory: &Arc<RwLock<HashMap<PatternId, MemoryType>>>,
    ) {
        // Implementation would move memories between tiers based on importance/usage
        match (from_tier, to_tier) {
            (MemoryTier::ShortTerm, MemoryTier::LongTerm) => {
                // Move well-established short-term memories to long-term
                if let (Ok(mut short), Ok(mut long)) = (short_term_memory.write(), long_term_memory.write()) {
                    let mut to_promote = Vec::new();
                    
                    for (pattern_id, memory) in short.iter() {
                        if let MemoryType::ShortTerm { consolidation_score, .. } = memory {
                            if *consolidation_score > 0.8 {
                                to_promote.push(*pattern_id);
                            }
                        }
                    }
                    
                    for pattern_id in to_promote {
                        if let Some(memory) = short.remove(&pattern_id) {
                            // Convert to long-term format with DNA compression
                            let compressed_memory = Self::convert_to_long_term(memory);
                            long.insert(pattern_id, compressed_memory);
                        }
                    }
                }
            }
            _ => {
                // Other consolidation patterns
            }
        }
    }
    
    fn convert_to_long_term(memory: MemoryType) -> MemoryType {
        match memory {
            MemoryType::ShortTerm { pattern, importance, .. } => {
                // Apply DNA compression
                let dna_sequence = Self::compress_to_dna(&pattern);
                MemoryType::LongTerm {
                    dna_sequence,
                    importance,
                    retrieval_strength: 0.8,
                    ewc_protection: 0.0,
                }
            }
            other => other, // Already in correct format
        }
    }
    
    fn compress_to_dna(pattern: &[f32]) -> Vec<u8> {
        // Simplified DNA compression - map float ranges to 4-bit DNA codes
        let mut dna = Vec::new();
        
        for &value in pattern {
            let normalized = ((value + 1.0) / 2.0).clamp(0.0, 1.0); // Normalize to [0,1]
            let dna_code = (normalized * 3.0) as u8; // Map to 0-3 (A,T,C,G)
            dna.push(dna_code);
        }
        
        // Pack 2 DNA codes per byte (4-bit each)
        let mut packed = Vec::new();
        for chunk in dna.chunks(2) {
            let byte = if chunk.len() == 2 {
                (chunk[0] << 4) | chunk[1]
            } else {
                chunk[0] << 4
            };
            packed.push(byte);
        }
        
        packed
    }
    
    fn decompress_from_dna(dna_sequence: &[u8]) -> Vec<f32> {
        let mut pattern = Vec::new();
        
        for &byte in dna_sequence {
            // Unpack 2 DNA codes from byte
            let code1 = (byte >> 4) & 0x0F;
            let code2 = byte & 0x0F;
            
            // Convert back to float range [-1, 1]
            let value1 = (code1 as f32 / 3.0) * 2.0 - 1.0;
            let value2 = (code2 as f32 / 3.0) * 2.0 - 1.0;
            
            pattern.push(value1);
            if byte & 0x0F != 0 { // Check if second code exists
                pattern.push(value2);
            }
        }
        
        pattern
    }
    
    fn compress_memory_pattern(
        _pattern_id: PatternId,
        _long_term_memory: &Arc<RwLock<HashMap<PatternId, MemoryType>>>,
    ) {
        // Apply more aggressive compression to existing long-term memories
    }
    
    fn apply_ewc_protection(
        pattern_id: PatternId,
        importance: f32,
        ewc_importance: &Arc<RwLock<HashMap<PatternId, f32>>>,
    ) {
        if let Ok(mut ewc_map) = ewc_importance.write() {
            ewc_map.insert(pattern_id, importance);
        }
    }
    
    fn update_meta_learning(
        _pattern_id: PatternId,
        learning_signal: f32,
        meta_state: &Arc<RwLock<MetaLearningState>>,
        _meta_memory: &Arc<RwLock<HashMap<PatternId, MemoryType>>>,
    ) {
        if let Ok(mut state) = meta_state.write() {
            state.learning_rate_history.push_back(learning_signal);
            
            // Keep only recent history
            while state.learning_rate_history.len() > 1000 {
                state.learning_rate_history.pop_front();
            }
            
            // Update generalization performance based on learning signal
            let alpha = 0.01;
            state.generalization_performance = 
                alpha * learning_signal + (1.0 - alpha) * state.generalization_performance;
        }
    }
    
    fn garbage_collect_tier(
        _tier: MemoryTier,
        _working_memory: &Arc<RwLock<HashMap<PatternId, MemoryType>>>,
        _short_term_memory: &Arc<RwLock<HashMap<PatternId, MemoryType>>>,
        _long_term_memory: &Arc<RwLock<HashMap<PatternId, MemoryType>>>,
    ) {
        // Remove old, unused memories to free space
    }
    
    fn start_memory_maintenance(&self) {
        let sender = self.work_queue.0.clone();
        let memory_time = self.memory_time.clone();
        
        thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_millis(100)); // 10Hz maintenance
                
                let current_time = memory_time.fetch_add(1, Ordering::Relaxed);
                
                // Periodic consolidation
                if current_time % 50 == 0 { // Every 5 seconds
                    let _ = sender.send(MemoryOperation::Consolidate {
                        from_tier: MemoryTier::ShortTerm,
                        to_tier: MemoryTier::LongTerm,
                    });
                }
                
                // Periodic garbage collection
                if current_time % 100 == 0 { // Every 10 seconds
                    let _ = sender.send(MemoryOperation::GarbageCollect {
                        tier: MemoryTier::Working,
                    });
                }
            }
        });
    }
    
    /// Store a new pattern in memory
    pub fn store_pattern(&self, pattern: Vec<f32>, importance: ImportanceScore) -> PatternId {
        let pattern_id = PatternId(self.total_patterns.fetch_add(1, Ordering::Relaxed) as u64);
        
        let memory = MemoryType::Working {
            pattern,
            importance,
            access_count: 1,
            last_access: self.memory_time.load(Ordering::Relaxed),
        };
        
        let _ = self.work_queue.0.send(MemoryOperation::Store {
            pattern_id,
            memory,
        });
        
        pattern_id
    }
    
    /// Retrieve a pattern from memory
    pub fn retrieve_pattern(&self, pattern_id: PatternId) -> Option<Vec<f32>> {
        // Try working memory first
        if let Ok(working) = self.working_memory.read() {
            if let Some(MemoryType::Working { pattern, .. }) = working.get(&pattern_id) {
                return Some(pattern.clone());
            }
        }
        
        // Try short-term memory
        if let Ok(short_term) = self.short_term_memory.read() {
            if let Some(MemoryType::ShortTerm { pattern, .. }) = short_term.get(&pattern_id) {
                return Some(pattern.clone());
            }
        }
        
        // Try long-term memory (requires decompression)
        if let Ok(long_term) = self.long_term_memory.read() {
            if let Some(MemoryType::LongTerm { dna_sequence, .. }) = long_term.get(&pattern_id) {
                return Some(Self::decompress_from_dna(dna_sequence));
            }
        }
        
        None
    }
    
    /// Get memory statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        let working_count = self.working_memory.read().map(|m| m.len()).unwrap_or(0);
        let short_term_count = self.short_term_memory.read().map(|m| m.len()).unwrap_or(0);
        let long_term_count = self.long_term_memory.read().map(|m| m.len()).unwrap_or(0);
        let meta_count = self.meta_memory.read().map(|m| m.len()).unwrap_or(0);
        
        let ant_performance: Vec<_> = self.worker_ants
            .iter()
            .map(|ant| AntPerformance {
                id: ant.id,
                specialty: format!("{:?}", ant.specialty),
                energy: ant.energy.load(Ordering::Relaxed),
                memory_moved: ant.memory_moved.load(Ordering::Relaxed),
                active: ant.active.load(Ordering::Relaxed),
            })
            .collect();
        
        MemoryStats {
            total_patterns: self.total_patterns.load(Ordering::Relaxed),
            working_memory_count: working_count,
            short_term_memory_count: short_term_count,
            long_term_memory_count: long_term_count,
            meta_memory_count: meta_count,
            memory_efficiency: self.memory_efficiency.load(Ordering::Relaxed),
            consolidation_rate: self.consolidation_rate.load(Ordering::Relaxed),
            memory_time: self.memory_time.load(Ordering::Relaxed),
            ant_performance,
        }
    }
}

#[derive(Debug)]
pub struct MemoryStats {
    pub total_patterns: usize,
    pub working_memory_count: usize,
    pub short_term_memory_count: usize,
    pub long_term_memory_count: usize,
    pub meta_memory_count: usize,
    pub memory_efficiency: u64,
    pub consolidation_rate: u64,
    pub memory_time: u64,
    pub ant_performance: Vec<AntPerformance>,
}

#[derive(Debug)]
pub struct AntPerformance {
    pub id: usize,
    pub specialty: String,
    pub energy: u64,
    pub memory_moved: u64,
    pub active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dna_compression() {
        let original = vec![0.5, -0.3, 1.0, -1.0, 0.0];
        let compressed = BrainMemory::compress_to_dna(&original);
        let decompressed = BrainMemory::decompress_from_dna(&compressed);
        
        // Check that compression significantly reduces size
        assert!(compressed.len() < original.len() * 4); // 4 bytes per f32
        
        // Check that decompression is reasonably close
        for (orig, decomp) in original.iter().zip(decompressed.iter()) {
            assert!((orig - decomp).abs() < 0.5); // Allow some loss due to quantization
        }
    }
    
    #[test]
    fn test_memory_storage_retrieval() {
        let brain = BrainMemory::new(6);
        
        let pattern = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let importance = ImportanceScore(0.8);
        
        let pattern_id = brain.store_pattern(pattern.clone(), importance);
        
        // Give worker ants time to process
        std::thread::sleep(Duration::from_millis(10));
        
        let retrieved = brain.retrieve_pattern(pattern_id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), pattern);
    }
    
    #[test]
    fn test_worker_ant_creation() {
        let brain = BrainMemory::new(6);
        
        assert_eq!(brain.worker_ants.len(), 6);
        
        // Check that we have different specialties
        let mut specialties = std::collections::HashSet::new();
        for ant in &brain.worker_ants {
            specialties.insert(format!("{:?}", ant.specialty));
        }
        
        assert!(specialties.len() >= 4); // Should have multiple different specialties
    }
}