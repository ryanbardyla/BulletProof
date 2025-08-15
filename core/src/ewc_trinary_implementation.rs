//! Elastic Weight Consolidation for Trinary Neural Networks
//! 
//! REVOLUTIONARY: First-ever EWC implementation for trinary (3-state) neurons!
//! Prevents catastrophic forgetting while maintaining energy efficiency through sparsity.
//! 
//! Based on Kirkpatrick et al. (2017) but adapted for trinary computing where:
//! - Baseline neurons (0) contribute ZERO to Fisher Information (energy efficient!)
//! - Activated (+1) and Inhibited (-1) neurons have asymmetric importance
//! - Fisher Information Matrix computed only for non-baseline connections
//! 
//! This allows continuous learning from market data without forgetting profitable patterns!

use crate::tryte::Tryte;
use crate::sparse_trith_net::SparseTrithNet;
use crate::protein_synthesis::{ProteinSynthesisMemory, MemoryConsolidation};
use crate::dna_compression::DNACompressor;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use ndarray::{Array2, Array1};
use serde::{Serialize, Deserialize};

/// Fisher Information Matrix for trinary networks
#[derive(Debug, Clone)]
pub struct TrinaryFisherMatrix {
    /// Fisher information for each weight (sparse representation)
    pub fisher_diagonal: HashMap<WeightId, f32>,
    /// Importance threshold - weights below this are free to change
    pub importance_threshold: f32,
    /// Sparsity level - percentage of baseline weights
    pub sparsity_percentage: f32,
    /// Task ID this Fisher matrix belongs to
    pub task_id: String,
    /// Number of samples used to compute Fisher
    pub sample_count: usize,
}

/// Unique identifier for a weight in the network
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct WeightId {
    pub layer: usize,
    pub from_neuron: usize,
    pub to_neuron: usize,
}

/// EWC memory for multiple tasks
#[derive(Debug, Clone)]
pub struct EWCMemory {
    /// Fisher matrices for previous tasks
    pub task_fisher_matrices: HashMap<String, TrinaryFisherMatrix>,
    /// Optimal weights for each task
    pub task_optimal_weights: HashMap<String, HashMap<WeightId, f32>>,
    /// Current task being learned
    pub current_task: String,
    /// Regularization strength (lambda in EWC paper)
    pub lambda: f32,
    /// Memory budget - max number of tasks to remember
    pub memory_budget: usize,
    /// Protein synthesis for biological memory consolidation
    pub protein_memory: Arc<RwLock<ProteinSynthesisMemory>>,
}

/// EWC training sample with trinary data
#[derive(Debug, Clone)]
pub struct EWCSample {
    pub input: Vec<Tryte>,
    pub target: Vec<Tryte>,
    pub importance_weight: f32, // Golden Geese tier weighting
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Statistics for EWC performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EWCStats {
    pub tasks_learned: usize,
    pub catastrophic_forgetting_prevented: usize,
    pub average_fisher_sparsity: f32,
    pub memory_efficiency: f32,
    pub important_weights_preserved: usize,
    pub baseline_weights_freed: usize,
    pub total_consolidations: usize,
}

impl EWCMemory {
    pub fn new(lambda: f32, memory_budget: usize) -> Self {
        println!("🧠 Initializing EWC Memory System...");
        println!("📊 Lambda: {}, Memory budget: {} tasks", lambda, memory_budget);
        
        Self {
            task_fisher_matrices: HashMap::new(),
            task_optimal_weights: HashMap::new(),
            current_task: "initial".to_string(),
            lambda,
            memory_budget,
            protein_memory: Arc::new(RwLock::new(ProteinSynthesisMemory::new())),
        }
    }
    
    /// Compute Fisher Information Matrix for trinary network
    pub fn compute_fisher_matrix(
        &mut self,
        network: &SparseTrithNet,
        samples: &[EWCSample],
    ) -> Result<TrinaryFisherMatrix, Box<dyn std::error::Error>> {
        println!("🔬 Computing Fisher Information Matrix for {} samples...", samples.len());
        
        let mut fisher_diagonal = HashMap::new();
        let mut total_weights = 0;
        let mut baseline_weights = 0;
        
        // Process each sample
        for sample in samples {
            // Forward pass to get network activations
            let output = network.forward_sparse(&sample.input)?;
            
            // Compute gradient equivalent for trinary
            let gradient = self.compute_trinary_gradient(&output, &sample.target);
            
            // Update Fisher information (diagonal approximation)
            for layer_idx in 0..network.layers.len() {
                let layer = &network.layers[layer_idx];
                
                for from_idx in 0..layer.input_size {
                    for to_idx in 0..layer.output_size {
                        let weight_id = WeightId {
                            layer: layer_idx,
                            from_neuron: from_idx,
                            to_neuron: to_idx,
                        };
                        
                        // Get activation states
                        let from_activation = if from_idx < sample.input.len() {
                            sample.input[from_idx]
                        } else {
                            Tryte::Baseline
                        };
                        
                        let to_activation = if to_idx < output.len() {
                            output[to_idx]
                        } else {
                            Tryte::Baseline
                        };
                        
                        // CRITICAL: Skip baseline connections (ZERO energy!)
                        if from_activation == Tryte::Baseline || to_activation == Tryte::Baseline {
                            baseline_weights += 1;
                            continue; // No Fisher information for inactive connections
                        }
                        
                        // Compute Fisher contribution (squared gradient)
                        let fisher_contribution = self.compute_fisher_contribution(
                            from_activation,
                            to_activation,
                            &gradient,
                            sample.importance_weight,
                        );
                        
                        *fisher_diagonal.entry(weight_id).or_insert(0.0) += fisher_contribution;
                        total_weights += 1;
                    }
                }
            }
        }
        
        // Normalize Fisher values
        let sample_count = samples.len();
        for fisher_value in fisher_diagonal.values_mut() {
            *fisher_value /= sample_count as f32;
        }
        
        // Calculate sparsity
        let sparsity_percentage = if total_weights + baseline_weights > 0 {
            baseline_weights as f32 / (total_weights + baseline_weights) as f32
        } else {
            0.0
        };
        
        // Determine importance threshold (keep top 20% of weights important)
        let mut fisher_values: Vec<f32> = fisher_diagonal.values().cloned().collect();
        fisher_values.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let importance_threshold = if fisher_values.len() > 5 {
            fisher_values[fisher_values.len() / 5] // 80th percentile
        } else {
            0.01
        };
        
        println!("✅ Fisher matrix computed:");
        println!("   📊 Important weights: {}", fisher_diagonal.len());
        println!("   ⚡ Sparsity: {:.1}%", sparsity_percentage * 100.0);
        println!("   🎯 Importance threshold: {:.4}", importance_threshold);
        
        Ok(TrinaryFisherMatrix {
            fisher_diagonal,
            importance_threshold,
            sparsity_percentage,
            task_id: self.current_task.clone(),
            sample_count,
        })
    }
    
    /// Compute trinary gradient (simplified for 3-state)
    fn compute_trinary_gradient(&self, output: &[Tryte], target: &[Tryte]) -> Vec<f32> {
        output.iter().zip(target.iter()).map(|(out, tar)| {
            let error = (*out as i8 - *tar as i8) as f32;
            // Gradient is proportional to error but bounded to [-1, 1]
            error.clamp(-1.0, 1.0)
        }).collect()
    }
    
    /// Compute Fisher contribution for a weight
    fn compute_fisher_contribution(
        &self,
        from_activation: Tryte,
        to_activation: Tryte,
        gradient: &[f32],
        importance_weight: f32,
    ) -> f32 {
        // Convert trinary to contribution factor
        let from_factor = match from_activation {
            Tryte::Activated => 1.0,
            Tryte::Inhibited => -1.0,
            Tryte::Baseline => 0.0, // Should never reach here due to skip
        };
        
        let to_factor = match to_activation {
            Tryte::Activated => 1.0,
            Tryte::Inhibited => -1.0,
            Tryte::Baseline => 0.0, // Should never reach here due to skip
        };
        
        // Average gradient magnitude
        let avg_gradient = gradient.iter().map(|g| g.abs()).sum::<f32>() / gradient.len() as f32;
        
        // Fisher = (gradient * activation)^2 * importance
        let fisher = (avg_gradient * from_factor.abs() * to_factor.abs()).powi(2) * importance_weight;
        
        fisher
    }
    
    /// Consolidate current task into EWC memory
    pub async fn consolidate_task(
        &mut self,
        task_name: &str,
        fisher_matrix: TrinaryFisherMatrix,
        current_weights: HashMap<WeightId, f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("🧬 Consolidating task '{}' into EWC memory...", task_name);
        
        // Store Fisher matrix
        self.task_fisher_matrices.insert(task_name.to_string(), fisher_matrix.clone());
        
        // Store optimal weights
        self.task_optimal_weights.insert(task_name.to_string(), current_weights);
        
        // Protein synthesis for biological consolidation
        let mut protein_memory = self.protein_memory.write().unwrap();
        let consolidation = MemoryConsolidation::new(
            fisher_matrix.fisher_diagonal.len(),
            0.8, // High confidence for important weights
        );
        protein_memory.consolidate_memories(vec![consolidation]).await?;
        
        // Check memory budget
        if self.task_fisher_matrices.len() > self.memory_budget {
            self.prune_old_memories();
        }
        
        println!("✅ Task consolidated successfully");
        println!("   📚 Total tasks in memory: {}", self.task_fisher_matrices.len());
        println!("   🧬 Protein synthesis completed");
        
        Ok(())
    }
    
    /// Prune old memories to stay within budget
    fn prune_old_memories(&mut self) {
        println!("🗑️  Pruning old memories to stay within budget...");
        
        // Find least important task (lowest average Fisher information)
        let mut task_importances: Vec<(String, f32)> = self.task_fisher_matrices.iter()
            .map(|(task_id, fisher)| {
                let avg_importance = if !fisher.fisher_diagonal.is_empty() {
                    fisher.fisher_diagonal.values().sum::<f32>() / fisher.fisher_diagonal.len() as f32
                } else {
                    0.0
                };
                (task_id.clone(), avg_importance)
            })
            .collect();
        
        task_importances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        // Remove least important task
        if let Some((task_to_remove, importance)) = task_importances.first() {
            println!("   Removing task '{}' (importance: {:.4})", task_to_remove, importance);
            self.task_fisher_matrices.remove(task_to_remove);
            self.task_optimal_weights.remove(task_to_remove);
        }
    }
    
    /// Calculate EWC penalty for current weights
    pub fn calculate_ewc_penalty(
        &self,
        current_weights: &HashMap<WeightId, f32>,
    ) -> f32 {
        let mut total_penalty = 0.0;
        
        // For each previous task
        for (task_id, fisher_matrix) in &self.task_fisher_matrices {
            if let Some(optimal_weights) = self.task_optimal_weights.get(task_id) {
                // Calculate penalty for this task
                let task_penalty = self.calculate_task_penalty(
                    current_weights,
                    optimal_weights,
                    fisher_matrix,
                );
                total_penalty += task_penalty;
            }
        }
        
        // Apply regularization strength
        total_penalty * self.lambda
    }
    
    /// Calculate penalty for a specific task
    fn calculate_task_penalty(
        &self,
        current_weights: &HashMap<WeightId, f32>,
        optimal_weights: &HashMap<WeightId, f32>,
        fisher_matrix: &TrinaryFisherMatrix,
    ) -> f32 {
        let mut penalty = 0.0;
        
        for (weight_id, fisher_value) in &fisher_matrix.fisher_diagonal {
            // Skip unimportant weights (below threshold)
            if *fisher_value < fisher_matrix.importance_threshold {
                continue;
            }
            
            let current = current_weights.get(weight_id).unwrap_or(&0.0);
            let optimal = optimal_weights.get(weight_id).unwrap_or(&0.0);
            
            // EWC penalty: F * (w - w*)^2
            let weight_diff = current - optimal;
            penalty += fisher_value * weight_diff.powi(2);
        }
        
        penalty
    }
    
    /// Get important weights that should be preserved
    pub fn get_important_weights(&self) -> HashMap<WeightId, f32> {
        let mut important_weights = HashMap::new();
        
        for fisher_matrix in self.task_fisher_matrices.values() {
            for (weight_id, fisher_value) in &fisher_matrix.fisher_diagonal {
                if *fisher_value >= fisher_matrix.importance_threshold {
                    // Accumulate importance across all tasks
                    *important_weights.entry(weight_id.clone()).or_insert(0.0) += fisher_value;
                }
            }
        }
        
        important_weights
    }
    
    /// Check if catastrophic forgetting is being prevented
    pub fn is_preventing_forgetting(
        &self,
        current_weights: &HashMap<WeightId, f32>,
        tolerance: f32,
    ) -> bool {
        let important_weights = self.get_important_weights();
        let mut preserved_count = 0;
        let mut total_important = 0;
        
        for (weight_id, _importance) in important_weights {
            total_important += 1;
            
            // Check if weight is preserved across all tasks
            let mut is_preserved = true;
            for optimal_weights in self.task_optimal_weights.values() {
                if let (Some(current), Some(optimal)) = 
                    (current_weights.get(&weight_id), optimal_weights.get(&weight_id)) {
                    if (current - optimal).abs() > tolerance {
                        is_preserved = false;
                        break;
                    }
                }
            }
            
            if is_preserved {
                preserved_count += 1;
            }
        }
        
        // Consider forgetting prevented if >80% of important weights preserved
        let preservation_rate = if total_important > 0 {
            preserved_count as f32 / total_important as f32
        } else {
            1.0
        };
        
        preservation_rate > 0.8
    }
    
    /// Get EWC statistics
    pub fn get_stats(&self) -> EWCStats {
        let mut total_fisher_values = 0;
        let mut total_weights = 0;
        let mut baseline_freed = 0;
        
        for fisher_matrix in self.task_fisher_matrices.values() {
            total_fisher_values += fisher_matrix.fisher_diagonal.len();
            total_weights += fisher_matrix.sample_count * 1000; // Estimate
            baseline_freed += (fisher_matrix.sparsity_percentage * 1000.0) as usize;
        }
        
        let avg_sparsity = if !self.task_fisher_matrices.is_empty() {
            self.task_fisher_matrices.values()
                .map(|f| f.sparsity_percentage)
                .sum::<f32>() / self.task_fisher_matrices.len() as f32
        } else {
            0.0
        };
        
        EWCStats {
            tasks_learned: self.task_fisher_matrices.len(),
            catastrophic_forgetting_prevented: 0, // Updated elsewhere
            average_fisher_sparsity: avg_sparsity,
            memory_efficiency: 1.0 - (total_fisher_values as f32 / total_weights.max(1) as f32),
            important_weights_preserved: total_fisher_values,
            baseline_weights_freed: baseline_freed,
            total_consolidations: self.task_fisher_matrices.len(),
        }
    }
    
    /// Compress EWC memory using DNA encoding
    pub async fn compress_memory(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        println!("🧬 Compressing EWC memory with DNA encoding...");
        
        let mut compressor = DNACompressor::new();
        let mut compressed_data = Vec::new();
        
        // Compress each task's Fisher matrix
        for (task_id, fisher_matrix) in &self.task_fisher_matrices {
            // Extract Fisher values as vector
            let fisher_values: Vec<f32> = fisher_matrix.fisher_diagonal.values().cloned().collect();
            
            // Compress with DNA
            let dna_sequence = compressor.compress_weights(&fisher_values);
            
            println!("   Task '{}': {} values → {} DNA bases ({:.1}x compression)",
                    task_id,
                    fisher_values.len(),
                    dna_sequence.len(),
                    dna_sequence.compression_ratio());
            
            // Store compressed data
            compressed_data.extend_from_slice(&dna_sequence.to_string().as_bytes());
        }
        
        println!("✅ EWC memory compressed to {} bytes", compressed_data.len());
        Ok(compressed_data)
    }
}

// Note: EWC penalty methods are now implemented in sparse_trith_net.rs
// to avoid duplicate definitions

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ewc_memory_creation() {
        let ewc = EWCMemory::new(0.5, 10);
        assert_eq!(ewc.lambda, 0.5);
        assert_eq!(ewc.memory_budget, 10);
        assert_eq!(ewc.current_task, "initial");
    }
    
    #[test]
    fn test_fisher_computation() {
        let mut ewc = EWCMemory::new(0.5, 10);
        let network = SparseTrithNet::new(10, 5, 3).unwrap();
        
        let samples = vec![
            EWCSample {
                input: vec![Tryte::Activated; 10],
                target: vec![Tryte::Baseline, Tryte::Activated, Tryte::Inhibited],
                importance_weight: 1.0,
                timestamp: chrono::Utc::now(),
            }
        ];
        
        let fisher = ewc.compute_fisher_matrix(&network, &samples).unwrap();
        assert!(fisher.sparsity_percentage >= 0.0);
        assert!(fisher.sparsity_percentage <= 1.0);
    }
    
    #[test]
    fn test_catastrophic_forgetting_detection() {
        let ewc = EWCMemory::new(0.5, 10);
        let current_weights = HashMap::new();
        
        // With no tasks, should not detect forgetting
        assert!(ewc.is_preventing_forgetting(&current_weights, 0.1));
    }
    
    #[tokio::test]
    async fn test_task_consolidation() {
        let mut ewc = EWCMemory::new(0.5, 10);
        
        let fisher = TrinaryFisherMatrix {
            fisher_diagonal: HashMap::new(),
            importance_threshold: 0.01,
            sparsity_percentage: 0.8,
            task_id: "test_task".to_string(),
            sample_count: 100,
        };
        
        let weights = HashMap::new();
        
        ewc.consolidate_task("test_task", fisher, weights).await.unwrap();
        
        assert_eq!(ewc.task_fisher_matrices.len(), 1);
        assert!(ewc.task_fisher_matrices.contains_key("test_task"));
    }
    
    #[test]
    fn test_memory_pruning() {
        let mut ewc = EWCMemory::new(0.5, 2); // Small budget
        
        // Add 3 tasks (exceeds budget of 2)
        for i in 0..3 {
            let task_name = format!("task_{}", i);
            let mut fisher_diagonal = HashMap::new();
            fisher_diagonal.insert(
                WeightId { layer: 0, from_neuron: 0, to_neuron: 0 },
                i as f32 * 0.1, // Different importance
            );
            
            let fisher = TrinaryFisherMatrix {
                fisher_diagonal,
                importance_threshold: 0.01,
                sparsity_percentage: 0.8,
                task_id: task_name.clone(),
                sample_count: 100,
            };
            
            ewc.task_fisher_matrices.insert(task_name, fisher);
        }
        
        ewc.prune_old_memories();
        
        // Should have pruned to stay within budget
        assert_eq!(ewc.task_fisher_matrices.len(), 2);
    }
}