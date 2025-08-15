//! Elastic Weight Consolidation with Fisher Information
//! 
//! Prevents catastrophic forgetting by calculating importance of each weight
//! using Fisher Information and protecting critical memories.

use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use ndarray::{Array1, Array2, ArrayView1};

/// Fisher Information Matrix for neural network weights
#[derive(Debug, Clone)]
pub struct FisherInformation {
    /// Fisher information for each parameter
    parameter_importance: HashMap<ParameterId, f32>,
    /// Running average of gradients squared
    gradient_accumulator: HashMap<ParameterId, GradientStats>,
    /// Number of samples processed
    num_samples: u64,
    /// Diagonal approximation (more efficient than full matrix)
    use_diagonal: bool,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct ParameterId(pub u64);

#[derive(Debug, Clone)]
struct GradientStats {
    sum_gradients: f32,
    sum_gradients_squared: f32,
    count: u64,
}

/// EWC-protected weight with importance score
#[derive(Debug, Clone)]
pub struct ProtectedWeight {
    pub value: f32,
    pub reference_value: f32,      // Value after previous task
    pub importance: f32,            // Fisher information
    pub protection_level: ProtectionLevel,
    pub task_id: u32,              // Which task this weight is important for
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProtectionLevel {
    None = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// EWC Regularizer that prevents catastrophic forgetting
pub struct EWCRegularizer {
    /// Protected weights from previous tasks
    protected_weights: Arc<RwLock<HashMap<ParameterId, Vec<ProtectedWeight>>>>,
    /// Fisher information calculator
    fisher_calc: Arc<RwLock<FisherInformation>>,
    /// EWC hyperparameter (regularization strength)
    lambda: f32,
    /// Current task ID
    current_task: u32,
    /// Importance threshold for protection
    importance_threshold: f32,
}

impl EWCRegularizer {
    pub fn new(lambda: f32) -> Self {
        Self {
            protected_weights: Arc::new(RwLock::new(HashMap::new())),
            fisher_calc: Arc::new(RwLock::new(FisherInformation::new())),
            lambda,
            current_task: 0,
            importance_threshold: 0.01,
        }
    }
    
    /// Calculate EWC loss for current weights
    pub fn calculate_ewc_loss(&self, current_weights: &HashMap<ParameterId, f32>) -> f32 {
        let protected = self.protected_weights.read().unwrap();
        let mut total_loss = 0.0;
        
        for (param_id, current_value) in current_weights {
            if let Some(protected_list) = protected.get(param_id) {
                for protected_weight in protected_list {
                    // EWC loss: importance * (current - reference)^2
                    let diff = current_value - protected_weight.reference_value;
                    let penalty = protected_weight.importance * diff * diff;
                    total_loss += self.lambda * penalty;
                }
            }
        }
        
        total_loss
    }
    
    /// Update Fisher information with new gradient
    pub fn update_fisher(&self, param_id: ParameterId, gradient: f32) {
        let mut fisher = self.fisher_calc.write().unwrap();
        fisher.update_gradient(param_id, gradient);
    }
    
    /// Consolidate current task's weights
    pub fn consolidate_task(&mut self, current_weights: &HashMap<ParameterId, f32>) {
        println!("🛡️ Consolidating task {} with EWC protection...", self.current_task);
        
        let fisher = self.fisher_calc.read().unwrap();
        let mut protected = self.protected_weights.write().unwrap();
        
        let mut protected_count = 0;
        let mut critical_count = 0;
        
        for (param_id, &value) in current_weights {
            let importance = fisher.get_importance(*param_id);
            
            if importance > self.importance_threshold {
                let protection_level = self.calculate_protection_level(importance);
                
                if protection_level == ProtectionLevel::Critical {
                    critical_count += 1;
                }
                
                let protected_weight = ProtectedWeight {
                    value,
                    reference_value: value,
                    importance,
                    protection_level,
                    task_id: self.current_task,
                };
                
                protected.entry(*param_id)
                    .or_insert_with(Vec::new)
                    .push(protected_weight);
                
                protected_count += 1;
            }
        }
        
        println!("  ✓ Protected {} weights ({} critical)", protected_count, critical_count);
        
        // Move to next task
        self.current_task += 1;
        
        // Reset Fisher information for next task
        *self.fisher_calc.write().unwrap() = FisherInformation::new();
    }
    
    /// Calculate protection level based on importance
    fn calculate_protection_level(&self, importance: f32) -> ProtectionLevel {
        if importance > 1.0 {
            ProtectionLevel::Critical
        } else if importance > 0.5 {
            ProtectionLevel::High
        } else if importance > 0.1 {
            ProtectionLevel::Medium
        } else if importance > 0.01 {
            ProtectionLevel::Low
        } else {
            ProtectionLevel::None
        }
    }
    
    /// Get protection level for a specific parameter
    pub fn get_protection_level(&self, param_id: ParameterId) -> ProtectionLevel {
        let protected = self.protected_weights.read().unwrap();
        
        protected.get(&param_id)
            .and_then(|weights| weights.iter().map(|w| w.protection_level).max())
            .unwrap_or(ProtectionLevel::None)
    }
    
    /// Compute gradient penalty for EWC
    pub fn compute_gradient_penalty(&self, param_id: ParameterId, 
                                   current_value: f32, gradient: f32) -> f32 {
        let protected = self.protected_weights.read().unwrap();
        
        if let Some(protected_list) = protected.get(&param_id) {
            let mut penalty = 0.0;
            
            for protected_weight in protected_list {
                // Penalize gradients that move away from protected values
                let diff = current_value - protected_weight.reference_value;
                let gradient_penalty = 2.0 * self.lambda * protected_weight.importance * diff;
                penalty += gradient_penalty;
            }
            
            gradient + penalty
        } else {
            gradient
        }
    }
    
    /// Get statistics about protected weights
    pub fn get_protection_stats(&self) -> ProtectionStats {
        let protected = self.protected_weights.read().unwrap();
        
        let mut total_protected = 0;
        let mut by_level = HashMap::new();
        let mut by_task = HashMap::new();
        
        for weights in protected.values() {
            for weight in weights {
                total_protected += 1;
                *by_level.entry(weight.protection_level).or_insert(0) += 1;
                *by_task.entry(weight.task_id).or_insert(0) += 1;
            }
        }
        
        ProtectionStats {
            total_protected_weights: total_protected,
            weights_by_protection_level: by_level,
            weights_by_task: by_task,
            current_task: self.current_task,
        }
    }
}

impl FisherInformation {
    fn new() -> Self {
        Self {
            parameter_importance: HashMap::new(),
            gradient_accumulator: HashMap::new(),
            num_samples: 0,
            use_diagonal: true, // Diagonal approximation for efficiency
        }
    }
    
    /// Update gradient statistics for Fisher calculation
    fn update_gradient(&mut self, param_id: ParameterId, gradient: f32) {
        let stats = self.gradient_accumulator.entry(param_id).or_insert(GradientStats {
            sum_gradients: 0.0,
            sum_gradients_squared: 0.0,
            count: 0,
        });
        
        stats.sum_gradients += gradient;
        stats.sum_gradients_squared += gradient * gradient;
        stats.count += 1;
        
        // Update Fisher importance (diagonal approximation)
        // Fisher ≈ E[gradient^2] for log-likelihood
        let fisher = stats.sum_gradients_squared / stats.count as f32;
        self.parameter_importance.insert(param_id, fisher);
        
        self.num_samples += 1;
    }
    
    /// Get importance score for a parameter
    fn get_importance(&self, param_id: ParameterId) -> f32 {
        self.parameter_importance.get(&param_id).copied().unwrap_or(0.0)
    }
    
    /// Get top-k most important parameters
    pub fn get_top_important(&self, k: usize) -> Vec<(ParameterId, f32)> {
        let mut importance_vec: Vec<_> = self.parameter_importance.iter()
            .map(|(&id, &imp)| (id, imp))
            .collect();
        
        importance_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        importance_vec.truncate(k);
        importance_vec
    }
}

/// Statistics about EWC protection
#[derive(Debug)]
pub struct ProtectionStats {
    pub total_protected_weights: usize,
    pub weights_by_protection_level: HashMap<ProtectionLevel, usize>,
    pub weights_by_task: HashMap<u32, usize>,
    pub current_task: u32,
}

/// Integration with memory substrate for importance-based consolidation
pub struct MemoryImportanceCalculator {
    fisher_info: Arc<RwLock<HashMap<crate::memory_substrate::MemoryId, f32>>>,
    access_frequency: Arc<RwLock<HashMap<crate::memory_substrate::MemoryId, u64>>>,
    recency_weights: Arc<RwLock<HashMap<crate::memory_substrate::MemoryId, f32>>>,
}

impl MemoryImportanceCalculator {
    pub fn new() -> Self {
        Self {
            fisher_info: Arc::new(RwLock::new(HashMap::new())),
            access_frequency: Arc::new(RwLock::new(HashMap::new())),
            recency_weights: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Calculate combined importance score for memory consolidation
    pub fn calculate_importance(&self, memory_id: crate::memory_substrate::MemoryId) -> f32 {
        let fisher = self.fisher_info.read().unwrap().get(&memory_id).copied().unwrap_or(0.0);
        let frequency = self.access_frequency.read().unwrap().get(&memory_id).copied().unwrap_or(0) as f32;
        let recency = self.recency_weights.read().unwrap().get(&memory_id).copied().unwrap_or(0.0);
        
        // Combined importance: Fisher information + access patterns
        let importance = 0.5 * fisher + 0.3 * (frequency / 100.0).min(1.0) + 0.2 * recency;
        importance
    }
    
    /// Update Fisher information for a memory pattern
    pub fn update_fisher_for_memory(&self, memory_id: crate::memory_substrate::MemoryId, 
                                   gradient_magnitude: f32) {
        let mut fisher = self.fisher_info.write().unwrap();
        let current = fisher.entry(memory_id).or_insert(0.0);
        
        // Exponential moving average
        *current = 0.9 * *current + 0.1 * gradient_magnitude * gradient_magnitude;
    }
    
    /// Update access frequency
    pub fn record_access(&self, memory_id: crate::memory_substrate::MemoryId) {
        *self.access_frequency.write().unwrap().entry(memory_id).or_insert(0) += 1;
    }
    
    /// Update recency weight
    pub fn update_recency(&self, memory_id: crate::memory_substrate::MemoryId, weight: f32) {
        self.recency_weights.write().unwrap().insert(memory_id, weight);
    }
    
    /// Get memories that should be protected
    pub fn get_protected_memories(&self, threshold: f32) -> Vec<crate::memory_substrate::MemoryId> {
        let fisher = self.fisher_info.read().unwrap();
        
        fisher.iter()
            .filter(|(_, &importance)| importance > threshold)
            .map(|(&id, _)| id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ewc_loss_calculation() {
        let mut ewc = EWCRegularizer::new(0.5);
        
        // Simulate first task
        let mut weights_task1 = HashMap::new();
        weights_task1.insert(ParameterId(1), 0.5);
        weights_task1.insert(ParameterId(2), -0.3);
        
        // Update Fisher information
        ewc.update_fisher(ParameterId(1), 0.8);
        ewc.update_fisher(ParameterId(2), 0.2);
        
        // Consolidate task 1
        ewc.consolidate_task(&weights_task1);
        
        // New weights for task 2
        let mut weights_task2 = HashMap::new();
        weights_task2.insert(ParameterId(1), 0.7); // Changed from 0.5
        weights_task2.insert(ParameterId(2), -0.3); // Unchanged
        
        let loss = ewc.calculate_ewc_loss(&weights_task2);
        assert!(loss > 0.0); // Should have penalty for changing protected weight
    }
    
    #[test]
    fn test_fisher_information_update() {
        let mut fisher = FisherInformation::new();
        
        // Simulate gradient updates
        for _ in 0..10 {
            fisher.update_gradient(ParameterId(1), 0.5);
            fisher.update_gradient(ParameterId(2), 0.1);
        }
        
        let importance1 = fisher.get_importance(ParameterId(1));
        let importance2 = fisher.get_importance(ParameterId(2));
        
        assert!(importance1 > importance2); // Larger gradients = more important
    }
    
    #[test]
    fn test_protection_levels() {
        let ewc = EWCRegularizer::new(1.0);
        
        assert_eq!(ewc.calculate_protection_level(2.0), ProtectionLevel::Critical);
        assert_eq!(ewc.calculate_protection_level(0.6), ProtectionLevel::High);
        assert_eq!(ewc.calculate_protection_level(0.05), ProtectionLevel::Low);
        assert_eq!(ewc.calculate_protection_level(0.001), ProtectionLevel::None);
    }
    
    #[test]
    fn test_memory_importance() {
        let calc = MemoryImportanceCalculator::new();
        let memory_id = crate::memory_substrate::MemoryId(1);
        
        // Update various importance factors
        calc.update_fisher_for_memory(memory_id, 0.8);
        calc.record_access(memory_id);
        calc.update_recency(memory_id, 0.7);
        
        let importance = calc.calculate_importance(memory_id);
        assert!(importance > 0.0);
        
        // Check if memory should be protected
        let protected = calc.get_protected_memories(0.1);
        assert!(protected.contains(&memory_id));
    }
}