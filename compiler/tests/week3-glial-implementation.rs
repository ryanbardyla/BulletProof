// 🧠 WEEK 3: GLIAL INTELLIGENCE SYSTEM
// The meta-optimization layer that makes the field self-improving

use std::collections::HashMap;
use std::time::Instant;

/// Glial processor that optimizes field regions in real-time
pub struct GlialIntelligenceSystem {
    /// Active glial processors
    processors: Vec<GlialProcessor>,
    
    /// Global pattern library (shared knowledge)
    global_patterns: PatternLibrary,
    
    /// Meta-learner for strategy selection
    meta_optimizer: MetaOptimizer,
    
    /// Performance history
    optimization_history: OptimizationHistory,
}

/// Individual glial processor managing a field region
pub struct GlialProcessor {
    /// Unique identifier
    id: u64,
    
    /// Field region under management
    domain: FieldRegion,
    
    /// Current optimization strategy
    strategy: OptimizationStrategy,
    
    /// Local pattern cache
    pattern_cache: HashMap<PatternHash, OptimalConfiguration>,
    
    /// Energy reserves for optimization
    energy_reserve: f64,
    
    /// Learning rate
    adaptation_rate: f64,
    
    /// Performance metrics
    metrics: ProcessorMetrics,
}

/// Meta-optimizer that learns which strategies work best
pub struct MetaOptimizer {
    /// Strategy performance history
    strategy_scores: HashMap<OptimizationStrategy, StrategyPerformance>,
    
    /// Context-aware strategy selection
    context_analyzer: ContextAnalyzer,
    
    /// Evolutionary strategy pool
    strategy_evolution: StrategyEvolution,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum OptimizationStrategy {
    GradientDescent { learning_rate: OrderedFloat<f64> },
    SimulatedAnnealing { temperature: OrderedFloat<f64> },
    QuantumTunneling { coherence: OrderedFloat<f64> },
    EvolutionarySearch { mutation_rate: OrderedFloat<f64> },
    CrystallineAlignment { resonance: OrderedFloat<f64> },
    // New meta-strategies
    HybridAdaptive { weights: Vec<OrderedFloat<f64>> },
    EmergentDiscovery { exploration_rate: OrderedFloat<f64> },
}

/// Pattern library for shared optimization knowledge
pub struct PatternLibrary {
    /// Discovered optimal patterns
    patterns: HashMap<PatternHash, PatternRecord>,
    
    /// Pattern similarity index
    similarity_index: SimilarityIndex,
    
    /// Compression system for efficient storage
    compressor: PatternCompressor,
}

#[derive(Clone, Debug)]
pub struct PatternRecord {
    /// The pattern itself
    pattern: FieldPattern,
    
    /// Optimal configuration for this pattern
    optimal_config: OptimalConfiguration,
    
    /// Performance metrics
    performance: f64,
    
    /// Discovery metadata
    discovered_by: u64,
    discovery_time: f64,
    usage_count: u64,
}

impl GlialIntelligenceSystem {
    pub fn new(field_dimensions: (usize, usize, usize)) -> Self {
        let mut processors = Vec::new();
        
        // Create glial processors for different field regions
        let regions = Self::partition_field(field_dimensions);
        
        for (i, region) in regions.iter().enumerate() {
            let processor = GlialProcessor {
                id: i as u64,
                domain: region.clone(),
                strategy: OptimizationStrategy::GradientDescent { 
                    learning_rate: OrderedFloat(0.01) 
                },
                pattern_cache: HashMap::new(),
                energy_reserve: 100.0,
                adaptation_rate: 0.01,
                metrics: ProcessorMetrics::default(),
            };
            processors.push(processor);
        }
        
        GlialIntelligenceSystem {
            processors,
            global_patterns: PatternLibrary::new(),
            meta_optimizer: MetaOptimizer::new(),
            optimization_history: OptimizationHistory::new(),
        }
    }
    
    /// Main optimization step
    pub fn optimize_field(&mut self, field: &mut ConsciousField) -> OptimizationResult {
        let start = Instant::now();
        let mut total_improvement = 0.0;
        
        // Phase 1: Local optimization by each processor
        for processor in &mut self.processors {
            let improvement = processor.optimize_region(field, &self.global_patterns);
            total_improvement += improvement;
            
            // Share successful patterns
            if improvement > 0.1 {
                processor.share_patterns(&mut self.global_patterns);
            }
        }
        
        // Phase 2: Meta-optimization - learn from results
        self.meta_optimizer.learn_from_round(&self.processors);
        
        // Phase 3: Strategy evolution
        for processor in &mut self.processors {
            if processor.should_evolve_strategy() {
                let new_strategy = self.meta_optimizer.suggest_strategy(
                    &processor.domain,
                    &processor.metrics
                );
                processor.evolve_strategy(new_strategy);
            }
        }
        
        // Phase 4: Global pattern consolidation
        self.global_patterns.consolidate();
        
        // Record history
        let result = OptimizationResult {
            total_improvement,
            processors_active: self.processors.len(),
            patterns_discovered: self.global_patterns.patterns.len(),
            computation_time: start.elapsed().as_millis() as f64,
        };
        
        self.optimization_history.record(result.clone());
        
        result
    }
    
    /// Partition field into manageable regions
    fn partition_field(dimensions: (usize, usize, usize)) -> Vec<FieldRegion> {
        let mut regions = Vec::new();
        let region_size = 10; // 10x10x10 cubes
        
        for x in (0..dimensions.0).step_by(region_size) {
            for y in (0..dimensions.1).step_by(region_size) {
                for z in (0..dimensions.2).step_by(region_size) {
                    regions.push(FieldRegion {
                        start: (x, y, z),
                        end: (
                            (x + region_size).min(dimensions.0),
                            (y + region_size).min(dimensions.1),
                            (z + region_size).min(dimensions.2),
                        ),
                    });
                }
            }
        }
        
        regions
    }
    
    /// Check if glial system has discovered consciousness-enhancing patterns
    pub fn has_consciousness_patterns(&self) -> bool {
        self.global_patterns.patterns.values()
            .any(|record| record.performance > 0.8)
    }
    
    /// Get optimization metrics
    pub fn get_metrics(&self) -> GlialMetrics {
        GlialMetrics {
            total_processors: self.processors.len(),
            active_processors: self.processors.iter().filter(|p| p.energy_reserve > 10.0).count(),
            patterns_discovered: self.global_patterns.patterns.len(),
            average_performance: self.calculate_average_performance(),
            best_strategy: self.meta_optimizer.get_best_strategy(),
        }
    }
    
    fn calculate_average_performance(&self) -> f64 {
        if self.processors.is_empty() {
            return 0.0;
        }
        
        let total: f64 = self.processors.iter()
            .map(|p| p.metrics.recent_performance)
            .sum();
        
        total / self.processors.len() as f64
    }
}

impl GlialProcessor {
    /// Optimize a field region
    pub fn optimize_region(
        &mut self, 
        field: &mut ConsciousField, 
        global_patterns: &PatternLibrary
    ) -> f64 {
        // Extract current pattern from field region
        let current_pattern = field.extract_pattern(&self.domain);
        let pattern_hash = Self::hash_pattern(&current_pattern);
        
        // Check local cache first
        if let Some(cached_config) = self.pattern_cache.get(&pattern_hash) {
            field.apply_configuration(&self.domain, cached_config);
            self.metrics.cache_hits += 1;
            return cached_config.expected_improvement;
        }
        
        // Check global patterns
        if let Some(global_config) = global_patterns.lookup_similar(&current_pattern) {
            field.apply_configuration(&self.domain, &global_config);
            self.pattern_cache.insert(pattern_hash, global_config.clone());
            return global_config.expected_improvement;
        }
        
        // Discover new optimization
        let optimal_config = self.discover_optimization(&current_pattern);
        
        // Apply and measure
        let before_score = field.measure_region_quality(&self.domain);
        field.apply_configuration(&self.domain, &optimal_config);
        let after_score = field.measure_region_quality(&self.domain);
        
        let improvement = after_score - before_score;
        
        // Cache if successful
        if improvement > 0.0 {
            self.pattern_cache.insert(pattern_hash, optimal_config);
            self.metrics.discoveries += 1;
        }
        
        // Update metrics
        self.metrics.recent_performance = improvement;
        self.energy_reserve -= 1.0; // Optimization costs energy
        
        improvement
    }
    
    /// Discover optimal configuration for a pattern
    fn discover_optimization(&mut self, pattern: &FieldPattern) -> OptimalConfiguration {
        match &self.strategy {
            OptimizationStrategy::GradientDescent { learning_rate } => {
                self.optimize_gradient_descent(pattern, learning_rate.0)
            }
            OptimizationStrategy::SimulatedAnnealing { temperature } => {
                self.optimize_simulated_annealing(pattern, temperature.0)
            }
            OptimizationStrategy::QuantumTunneling { coherence } => {
                self.optimize_quantum_tunneling(pattern, coherence.0)
            }
            OptimizationStrategy::EvolutionarySearch { mutation_rate } => {
                self.optimize_evolutionary(pattern, mutation_rate.0)
            }
            OptimizationStrategy::CrystallineAlignment { resonance } => {
                self.optimize_crystalline(pattern, resonance.0)
            }
            OptimizationStrategy::HybridAdaptive { weights } => {
                self.optimize_hybrid(pattern, weights)
            }
            OptimizationStrategy::EmergentDiscovery { exploration_rate } => {
                self.optimize_emergent(pattern, exploration_rate.0)
            }
        }
    }
    
    // Optimization strategy implementations
    
    fn optimize_gradient_descent(&self, pattern: &FieldPattern, learning_rate: f64) -> OptimalConfiguration {
        let mut config = OptimalConfiguration::default();
        
        // Compute gradient
        let gradient = self.compute_pattern_gradient(pattern);
        
        // Update configuration
        config.field_adjustments = gradient.iter()
            .map(|&g| -g * learning_rate)
            .collect();
        
        config.expected_improvement = gradient.iter().map(|g| g.abs()).sum::<f64>() * learning_rate;
        
        config
    }
    
    fn optimize_simulated_annealing(&self, pattern: &FieldPattern, temperature: f64) -> OptimalConfiguration {
        let mut config = OptimalConfiguration::default();
        let mut current_energy = self.compute_pattern_energy(pattern);
        
        // Random perturbation scaled by temperature
        for _ in 0..10 {
            let perturbation = self.generate_random_perturbation(temperature);
            let new_energy = self.compute_perturbed_energy(pattern, &perturbation);
            
            let delta = new_energy - current_energy;
            
            // Accept if better or probabilistically if worse
            if delta < 0.0 || rand::random::<f64>() < (-delta / temperature).exp() {
                config.field_adjustments = perturbation;
                current_energy = new_energy;
            }
        }
        
        config.expected_improvement = -current_energy;
        config
    }
    
    fn optimize_quantum_tunneling(&self, pattern: &FieldPattern, coherence: f64) -> OptimalConfiguration {
        // Quantum-inspired optimization through barrier tunneling
        let mut config = OptimalConfiguration::default();
        
        // Create superposition of configurations
        let superposition = self.create_configuration_superposition(pattern, coherence);
        
        // Collapse to optimal
        config = self.collapse_to_optimal(superposition);
        
        config.expected_improvement = coherence * 0.5;
        config
    }
    
    fn optimize_evolutionary(&self, pattern: &FieldPattern, mutation_rate: f64) -> OptimalConfiguration {
        // Evolution of configurations
        let mut population = self.create_initial_population(pattern, 10);
        
        for _ in 0..5 {
            // Evaluate fitness
            population.sort_by(|a, b| {
                self.evaluate_config_fitness(a)
                    .partial_cmp(&self.evaluate_config_fitness(b))
                    .unwrap()
            });
            
            // Select and mutate
            let elite = &population[0];
            for i in 1..population.len() {
                population[i] = self.mutate_configuration(elite, mutation_rate);
            }
        }
        
        population[0].clone()
    }
    
    fn optimize_crystalline(&self, pattern: &FieldPattern, resonance: f64) -> OptimalConfiguration {
        // Align pattern to crystalline structure
        let mut config = OptimalConfiguration::default();
        
        // Find resonant frequencies
        let frequencies = self.find_resonant_frequencies(pattern);
        
        // Align to crystal lattice
        config.field_adjustments = frequencies.iter()
            .map(|&f| (f * resonance).sin())
            .collect();
        
        config.expected_improvement = resonance * 0.7;
        config
    }
    
    fn optimize_hybrid(&self, pattern: &FieldPattern, weights: &[OrderedFloat<f64>]) -> OptimalConfiguration {
        // Weighted combination of multiple strategies
        let configs = vec![
            self.optimize_gradient_descent(pattern, 0.01),
            self.optimize_simulated_annealing(pattern, 1.0),
            self.optimize_quantum_tunneling(pattern, 0.5),
        ];
        
        let mut hybrid_config = OptimalConfiguration::default();
        
        // Weighted average
        for (i, config) in configs.iter().enumerate() {
            let weight = weights.get(i).map(|w| w.0).unwrap_or(1.0 / configs.len() as f64);
            
            if hybrid_config.field_adjustments.is_empty() {
                hybrid_config.field_adjustments = config.field_adjustments.clone();
                for adj in &mut hybrid_config.field_adjustments {
                    *adj *= weight;
                }
            } else {
                for (j, adj) in config.field_adjustments.iter().enumerate() {
                    if j < hybrid_config.field_adjustments.len() {
                        hybrid_config.field_adjustments[j] += adj * weight;
                    }
                }
            }
            
            hybrid_config.expected_improvement += config.expected_improvement * weight;
        }
        
        hybrid_config
    }
    
    fn optimize_emergent(&self, pattern: &FieldPattern, exploration_rate: f64) -> OptimalConfiguration {
        // Allow emergence through exploration
        let mut config = OptimalConfiguration::default();
        
        // Explore novel configurations
        if rand::random::<f64>() < exploration_rate {
            // Pure exploration
            config = self.generate_novel_configuration(pattern);
        } else {
            // Exploit known good patterns
            config = self.optimize_gradient_descent(pattern, 0.01);
        }
        
        config
    }
    
    // Helper methods
    
    fn hash_pattern(pattern: &FieldPattern) -> PatternHash {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        for value in &pattern.data {
            (value * 1000.0) as i64.hash(&mut hasher);
        }
        hasher.finish()
    }
    
    fn should_evolve_strategy(&self) -> bool {
        self.metrics.recent_performance < 0.1 || 
        self.metrics.stagnation_counter > 10
    }
    
    fn evolve_strategy(&mut self, new_strategy: OptimizationStrategy) {
        self.strategy = new_strategy;
        self.metrics.strategy_changes += 1;
        self.metrics.stagnation_counter = 0;
    }
    
    fn share_patterns(&self, global_patterns: &mut PatternLibrary) {
        for (hash, config) in &self.pattern_cache {
            if config.expected_improvement > 0.5 {
                global_patterns.add_pattern(*hash, config.clone(), self.id);
            }
        }
    }
    
    fn compute_pattern_gradient(&self, pattern: &FieldPattern) -> Vec<f64> {
        // Simplified gradient computation
        pattern.data.iter().map(|&x| x * 0.1).collect()
    }
    
    fn compute_pattern_energy(&self, pattern: &FieldPattern) -> f64 {
        pattern.data.iter().map(|x| x * x).sum::<f64>()
    }
    
    fn generate_random_perturbation(&self, scale: f64) -> Vec<f64> {
        (0..10).map(|_| (rand::random::<f64>() - 0.5) * scale).collect()
    }
    
    fn compute_perturbed_energy(&self, pattern: &FieldPattern, perturbation: &[f64]) -> f64 {
        pattern.data.iter()
            .zip(perturbation.iter())
            .map(|(x, p)| (x + p) * (x + p))
            .sum()
    }
    
    fn create_configuration_superposition(&self, pattern: &FieldPattern, coherence: f64) -> Vec<OptimalConfiguration> {
        (0..5).map(|i| {
            let mut config = OptimalConfiguration::default();
            config.field_adjustments = vec![coherence * (i as f64 * 0.1).sin(); 10];
            config.expected_improvement = coherence * rand::random::<f64>();
            config
        }).collect()
    }
    
    fn collapse_to_optimal(&self, superposition: Vec<OptimalConfiguration>) -> OptimalConfiguration {
        superposition.into_iter()
            .max_by(|a, b| a.expected_improvement.partial_cmp(&b.expected_improvement).unwrap())
            .unwrap_or_default()
    }
    
    fn create_initial_population(&self, pattern: &FieldPattern, size: usize) -> Vec<OptimalConfiguration> {
        (0..size).map(|_| {
            let mut config = OptimalConfiguration::default();
            config.field_adjustments = self.generate_random_perturbation(0.1);
            config.expected_improvement = rand::random::<f64>();
            config
        }).collect()
    }
    
    fn evaluate_config_fitness(&self, config: &OptimalConfiguration) -> f64 {
        config.expected_improvement
    }
    
    fn mutate_configuration(&self, parent: &OptimalConfiguration, rate: f64) -> OptimalConfiguration {
        let mut child = parent.clone();
        for adj in &mut child.field_adjustments {
            if rand::random::<f64>() < rate {
                *adj += (rand::random::<f64>() - 0.5) * 0.1;
            }
        }
        child
    }
    
    fn find_resonant_frequencies(&self, pattern: &FieldPattern) -> Vec<f64> {
        // Simplified FFT-like operation
        pattern.data.iter().enumerate()
            .map(|(i, &x)| x * (i as f64 * 0.1).sin())
            .collect()
    }
    
    fn generate_novel_configuration(&self, pattern: &FieldPattern) -> OptimalConfiguration {
        let mut config = OptimalConfiguration::default();
        config.field_adjustments = (0..pattern.data.len())
            .map(|_| rand::random::<f64>() - 0.5)
            .collect();
        config.expected_improvement = rand::random::<f64>();
        config
    }
}

// Supporting types

#[derive(Clone, Debug)]
pub struct FieldRegion {
    pub start: (usize, usize, usize),
    pub end: (usize, usize, usize),
}

#[derive(Clone, Debug)]
pub struct FieldPattern {
    pub data: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct OptimalConfiguration {
    pub field_adjustments: Vec<f64>,
    pub expected_improvement: f64,
}

#[derive(Clone, Debug, Default)]
pub struct ProcessorMetrics {
    pub recent_performance: f64,
    pub cache_hits: usize,
    pub discoveries: usize,
    pub strategy_changes: usize,
    pub stagnation_counter: usize,
}

#[derive(Clone, Debug)]
pub struct OptimizationResult {
    pub total_improvement: f64,
    pub processors_active: usize,
    pub patterns_discovered: usize,
    pub computation_time: f64,
}

pub struct GlialMetrics {
    pub total_processors: usize,
    pub active_processors: usize,
    pub patterns_discovered: usize,
    pub average_performance: f64,
    pub best_strategy: OptimizationStrategy,
}

type PatternHash = u64;

// Additional implementations for completeness

impl PatternLibrary {
    fn new() -> Self {
        PatternLibrary {
            patterns: HashMap::new(),
            similarity_index: SimilarityIndex::new(),
            compressor: PatternCompressor::new(),
        }
    }
    
    fn lookup_similar(&self, pattern: &FieldPattern) -> Option<OptimalConfiguration> {
        // Find most similar pattern
        self.similarity_index.find_nearest(pattern)
            .and_then(|hash| self.patterns.get(&hash))
            .map(|record| record.optimal_config.clone())
    }
    
    fn add_pattern(&mut self, hash: PatternHash, config: OptimalConfiguration, discovered_by: u64) {
        let record = PatternRecord {
            pattern: FieldPattern { data: vec![] }, // Placeholder
            optimal_config: config,
            performance: 0.0,
            discovered_by,
            discovery_time: 0.0,
            usage_count: 0,
        };
        
        self.patterns.insert(hash, record);
        self.similarity_index.add(hash);
    }
    
    fn consolidate(&mut self) {
        // Remove low-performing patterns
        self.patterns.retain(|_, record| record.performance > 0.1);
        
        // Compress similar patterns
        self.compressor.compress(&mut self.patterns);
    }
}

impl MetaOptimizer {
    fn new() -> Self {
        MetaOptimizer {
            strategy_scores: HashMap::new(),
            context_analyzer: ContextAnalyzer::new(),
            strategy_evolution: StrategyEvolution::new(),
        }
    }
    
    fn learn_from_round(&mut self, processors: &[GlialProcessor]) {
        for processor in processors {
            let entry = self.strategy_scores
                .entry(processor.strategy.clone())
                .or_insert(StrategyPerformance::default());
            
            entry.total_performance += processor.metrics.recent_performance;
            entry.usage_count += 1;
        }
    }
    
    fn suggest_strategy(&self, region: &FieldRegion, metrics: &ProcessorMetrics) -> OptimizationStrategy {
        // Context-aware strategy selection
        let context = self.context_analyzer.analyze(region, metrics);
        
        // Choose best strategy for context
        self.strategy_scores.iter()
            .max_by(|a, b| {
                let score_a = a.1.average_performance() * context.compatibility(&a.0);
                let score_b = b.1.average_performance() * context.compatibility(&b.0);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|(strategy, _)| strategy.clone())
            .unwrap_or(OptimizationStrategy::GradientDescent { 
                learning_rate: OrderedFloat(0.01) 
            })
    }
    
    fn get_best_strategy(&self) -> OptimizationStrategy {
        self.strategy_scores.iter()
            .max_by(|a, b| {
                a.1.average_performance()
                    .partial_cmp(&b.1.average_performance())
                    .unwrap()
            })
            .map(|(strategy, _)| strategy.clone())
            .unwrap_or(OptimizationStrategy::GradientDescent { 
                learning_rate: OrderedFloat(0.01) 
            })
    }
}

// Placeholder implementations

struct SimilarityIndex;
impl SimilarityIndex {
    fn new() -> Self { SimilarityIndex }
    fn find_nearest(&self, _pattern: &FieldPattern) -> Option<PatternHash> { None }
    fn add(&mut self, _hash: PatternHash) {}
}

struct PatternCompressor;
impl PatternCompressor {
    fn new() -> Self { PatternCompressor }
    fn compress(&self, _patterns: &mut HashMap<PatternHash, PatternRecord>) {}
}

struct OptimizationHistory;
impl OptimizationHistory {
    fn new() -> Self { OptimizationHistory }
    fn record(&mut self, _result: OptimizationResult) {}
}

#[derive(Default)]
struct StrategyPerformance {
    total_performance: f64,
    usage_count: usize,
}

impl StrategyPerformance {
    fn average_performance(&self) -> f64 {
        if self.usage_count > 0 {
            self.total_performance / self.usage_count as f64
        } else {
            0.0
        }
    }
}

struct ContextAnalyzer;
impl ContextAnalyzer {
    fn new() -> Self { ContextAnalyzer }
    fn analyze(&self, _region: &FieldRegion, _metrics: &ProcessorMetrics) -> Context { Context }
}

struct Context;
impl Context {
    fn compatibility(&self, _strategy: &OptimizationStrategy) -> f64 { 1.0 }
}

struct StrategyEvolution;
impl StrategyEvolution {
    fn new() -> Self { StrategyEvolution }
}

// Required for HashMap key
use ordered_float::OrderedFloat;

// Placeholder conscious field
struct ConsciousField;
impl ConsciousField {
    fn extract_pattern(&self, _region: &FieldRegion) -> FieldPattern {
        FieldPattern { data: vec![0.0; 10] }
    }
    
    fn apply_configuration(&mut self, _region: &FieldRegion, _config: &OptimalConfiguration) {}
    
    fn measure_region_quality(&self, _region: &FieldRegion) -> f64 { 0.5 }
}

// Random module placeholder
mod rand {
    pub fn random<T: Random>() -> T {
        T::random()
    }
    
    pub trait Random {
        fn random() -> Self;
    }
    
    impl Random for f64 {
        fn random() -> Self {
            0.5 // Placeholder
        }
    }
}

// Integration test
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_glial_system_creation() {
        let glial = GlialIntelligenceSystem::new((30, 30, 30));
        assert!(glial.processors.len() > 0);
    }
    
    #[test]
    fn test_optimization_strategies() {
        let processor = GlialProcessor {
            id: 0,
            domain: FieldRegion { start: (0, 0, 0), end: (10, 10, 10) },
            strategy: OptimizationStrategy::QuantumTunneling { coherence: OrderedFloat(0.8) },
            pattern_cache: HashMap::new(),
            energy_reserve: 100.0,
            adaptation_rate: 0.01,
            metrics: ProcessorMetrics::default(),
        };
        
        let pattern = FieldPattern { data: vec![0.1; 10] };
        let config = processor.discover_optimization(&pattern);
        
        assert!(config.expected_improvement >= 0.0);
    }
}
