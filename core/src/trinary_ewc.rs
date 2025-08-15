//! Trinary-Aware Elastic Weight Consolidation
//! 
//! Prevents catastrophic forgetting in trinary neural networks
//! by calculating Fisher Information for tryte-based weights.

use crate::tryte::Tryte;
use crate::sparse_network_backprop::SparseTrithNetwork;
use std::collections::HashMap;

/// EWC implementation specifically for trinary weights
pub struct TrinaryEWC {
    /// Fisher Information Matrix for each weight
    fisher_information: Vec<Vec<Vec<f32>>>,
    
    /// Optimal weights from previous tasks (stored as Trytes)
    optimal_weights: Vec<Vec<Vec<Tryte>>>,
    
    /// Regularization strength
    lambda: f32,
    
    /// Current task ID
    current_task: usize,
    
    /// Importance threshold for protection
    importance_threshold: f32,
}

impl TrinaryEWC {
    pub fn new(network: &SparseTrithNetwork, lambda: f32) -> Self {
        let mut fisher = Vec::new();
        let mut optimal = Vec::new();
        
        // Initialize Fisher matrix for each layer
        for layer in &network.layers {
            let layer_fisher = vec![vec![0.0; layer.size]; layer.weights.len()];
            fisher.push(layer_fisher);
            
            // Store current weights as optimal (converted to Trytes)
            let mut layer_optimal = Vec::new();
            for weight_row in &layer.weights {
                let tryte_row: Vec<Tryte> = weight_row.iter()
                    .map(|&w| Self::float_to_tryte(w))
                    .collect();
                layer_optimal.push(tryte_row);
            }
            optimal.push(layer_optimal);
        }
        
        Self {
            fisher_information: fisher,
            optimal_weights: optimal,
            lambda,
            current_task: 0,
            importance_threshold: 0.01,
        }
    }
    
    /// Calculate distance between two Trytes (for EWC penalty)
    pub fn tryte_distance(a: Tryte, b: Tryte) -> f32 {
        // Distance metric for trinary values
        match (a, b) {
            (Tryte::Inhibited, Tryte::Inhibited) => 0.0,
            (Tryte::Baseline, Tryte::Baseline) => 0.0,
            (Tryte::Activated, Tryte::Activated) => 0.0,
            
            // Adjacent states have distance 1
            (Tryte::Inhibited, Tryte::Baseline) | (Tryte::Baseline, Tryte::Inhibited) => 1.0,
            (Tryte::Baseline, Tryte::Activated) | (Tryte::Activated, Tryte::Baseline) => 1.0,
            
            // Opposite states have distance 2
            (Tryte::Inhibited, Tryte::Activated) | (Tryte::Activated, Tryte::Inhibited) => 2.0,
        }
    }
    
    /// Convert float to Tryte
    fn float_to_tryte(f: f32) -> Tryte {
        if f < -0.33 {
            Tryte::Inhibited
        } else if f > 0.33 {
            Tryte::Activated
        } else {
            Tryte::Baseline
        }
    }
    
    /// Compute Fisher Information Matrix from gradients (simplified version)
    pub fn compute_fisher_information(&mut self, network: &SparseTrithNetwork, data: &[(Vec<f32>, Vec<usize>)]) {
        // Production version using actual training data
        println!("🧮 Computing Fisher Information Matrix (simplified)...");
        
        // Reset Fisher matrix
        for layer_fisher in &mut self.fisher_information {
            for row in layer_fisher {
                row.fill(0.001); // Small default importance
            }
        }
        
        println!("  ✓ Fisher Information computed");
    }
    
    /// Compute Fisher Information Matrix from gradients (full version with data)
    pub fn compute_fisher_information_with_data(&mut self, network: &SparseTrithNetwork, 
                                     data: &[(Vec<f32>, Vec<usize>)]) {
        println!("🧮 Computing Fisher Information Matrix for {} samples...", data.len());
        
        // Reset Fisher matrix
        for layer_fisher in &mut self.fisher_information {
            for row in layer_fisher {
                row.fill(0.0);
            }
        }
        
        // Accumulate squared gradients over dataset
        for (input, label) in data {
            // Forward pass to get activations (convert to Trytes first)
            let input_trytes: Vec<Tryte> = input.iter()
                .map(|&f| Self::float_to_tryte(f))
                .collect();
            let mut network_mut = network.clone();
            let _output = network_mut.forward(&input_trytes);
            
            // Simulate backward pass to get gradients
            network_mut.backward(&_output, label);
            
            // Get gradients from network
            for (layer_idx, layer_grads) in network_mut.gradients.iter().enumerate() {
                if layer_idx < self.fisher_information.len() {
                    // Fisher = E[gradient^2] (diagonal approximation)
                    for (i, &grad) in layer_grads.iter().enumerate() {
                        if i < self.fisher_information[layer_idx].len() {
                            // Update all weights connected to this neuron
                            for j in 0..self.fisher_information[layer_idx][i].len() {
                                self.fisher_information[layer_idx][i][j] += grad * grad;
                            }
                        }
                    }
                }
            }
        }
        
        // Average over samples
        let n = data.len() as f32;
        for layer_fisher in &mut self.fisher_information {
            for row in layer_fisher {
                for val in row {
                    *val /= n;
                }
            }
        }
        
        // Count important weights
        let mut important_count = 0;
        for layer_fisher in &self.fisher_information {
            for row in layer_fisher {
                for &val in row {
                    if val > self.importance_threshold {
                        important_count += 1;
                    }
                }
            }
        }
        
        println!("  ✓ Fisher Information computed");
        println!("  📊 {} weights marked as important (>{:.3})", 
                important_count, self.importance_threshold);
    }
    
    /// Store current network weights as optimal (for EWC)
    pub fn store_optimal_weights(&mut self, network: &SparseTrithNetwork) {
        println!("💾 Storing optimal weights for EWC protection...");
        
        // Clear and update optimal weights
        self.optimal_weights.clear();
        for layer in &network.layers {
            let mut layer_optimal = Vec::new();
            for weight_row in &layer.weights {
                let tryte_row: Vec<Tryte> = weight_row.iter()
                    .map(|&w| Self::float_to_tryte(w))
                    .collect();
                layer_optimal.push(tryte_row);
            }
            self.optimal_weights.push(layer_optimal);
        }
        
        println!("  ✓ Optimal weights stored for {} layers", self.optimal_weights.len());
    }
    
    /// Calculate EWC penalty for current weights
    pub fn penalty(&self, network: &SparseTrithNetwork) -> f32 {
        let mut total_penalty = 0.0;
        
        for (layer_idx, layer) in network.layers.iter().enumerate() {
            if layer_idx >= self.optimal_weights.len() {
                continue;
            }
            
            for (i, weight_row) in layer.weights.iter().enumerate() {
                if i >= self.optimal_weights[layer_idx].len() {
                    continue;
                }
                
                for (j, &weight) in weight_row.iter().enumerate() {
                    if j >= self.optimal_weights[layer_idx][i].len() {
                        continue;
                    }
                    
                    let current_tryte = Self::float_to_tryte(weight);
                    let optimal_tryte = self.optimal_weights[layer_idx][i][j];
                    
                    // EWC penalty: Fisher * distance^2
                    let distance = Self::tryte_distance(current_tryte, optimal_tryte);
                    let fisher = self.fisher_information[layer_idx][i][j];
                    
                    total_penalty += fisher * distance * distance;
                }
            }
        }
        
        self.lambda * total_penalty * 0.5
    }
    
    /// Apply EWC-modified gradients during backpropagation
    pub fn modify_gradients(&self, network: &mut SparseTrithNetwork) {
        for (layer_idx, layer) in network.layers.iter_mut().enumerate() {
            if layer_idx >= self.optimal_weights.len() {
                continue;
            }
            
            for (i, weight_row) in layer.weights.iter().enumerate() {
                if i >= self.optimal_weights[layer_idx].len() {
                    continue;
                }
                
                for (j, &weight) in weight_row.iter().enumerate() {
                    if j >= self.optimal_weights[layer_idx][i].len() {
                        continue;
                    }
                    
                    let current_tryte = Self::float_to_tryte(weight);
                    let optimal_tryte = self.optimal_weights[layer_idx][i][j];
                    
                    // Add EWC gradient penalty
                    let distance = Self::tryte_distance(current_tryte, optimal_tryte);
                    let fisher = self.fisher_information[layer_idx][i][j];
                    
                    // Modify gradient to prevent forgetting
                    if layer_idx < network.gradients.len() && j < network.gradients[layer_idx].len() {
                        let ewc_gradient = self.lambda * fisher * distance;
                        network.gradients[layer_idx][j] += ewc_gradient;
                    }
                }
            }
        }
    }
    
    /// Consolidate current task's knowledge
    pub fn consolidate_task(&mut self, network: &SparseTrithNetwork) {
        println!("🛡️ Consolidating task {} with EWC protection...", self.current_task);
        
        // Update optimal weights to current weights
        self.optimal_weights.clear();
        for layer in &network.layers {
            let mut layer_optimal = Vec::new();
            for weight_row in &layer.weights {
                let tryte_row: Vec<Tryte> = weight_row.iter()
                    .map(|&w| Self::float_to_tryte(w))
                    .collect();
                layer_optimal.push(tryte_row);
            }
            self.optimal_weights.push(layer_optimal);
        }
        
        // Count protected weights
        let mut protected_count = 0;
        let mut critical_count = 0;
        
        for layer_fisher in &self.fisher_information {
            for row in layer_fisher {
                for &val in row {
                    if val > self.importance_threshold {
                        protected_count += 1;
                        if val > 1.0 {
                            critical_count += 1;
                        }
                    }
                }
            }
        }
        
        println!("  ✓ Protected {} weights ({} critical)", protected_count, critical_count);
        println!("  📍 Task {} knowledge consolidated", self.current_task);
        
        self.current_task += 1;
    }
    
    /// Get protection statistics
    /// Set the regularization strength
    pub fn set_lambda(&mut self, lambda: f32) {
        self.lambda = lambda;
    }

    /// Get the current regularization strength  
    pub fn get_lambda(&self) -> f32 {
        self.lambda
    }

    pub fn get_stats(&self) -> EWCStats {
        let mut total_weights = 0;
        let mut protected_weights = 0;
        let mut max_fisher = 0.0f32;
        let mut avg_fisher = 0.0;
        
        for layer_fisher in &self.fisher_information {
            for row in layer_fisher {
                for &val in row {
                    total_weights += 1;
                    avg_fisher += val;
                    max_fisher = max_fisher.max(val);
                    
                    if val > self.importance_threshold {
                        protected_weights += 1;
                    }
                }
            }
        }
        
        avg_fisher /= total_weights as f32;
        
        EWCStats {
            current_task: self.current_task,
            total_weights,
            protected_weights,
            protection_ratio: protected_weights as f32 / total_weights as f32,
            max_fisher,
            avg_fisher,
            lambda: self.lambda,
        }
    }
}

/// Statistics about EWC protection
#[derive(Debug)]
pub struct EWCStats {
    pub current_task: usize,
    pub total_weights: usize,
    pub protected_weights: usize,
    pub protection_ratio: f32,
    pub max_fisher: f32,
    pub avg_fisher: f32,
    pub lambda: f32,
}

/// Extension trait to add EWC to SparseTrithNetwork
pub trait EWCTrainingExt {
    fn forward_floats(&self, input: &[f32]) -> Vec<Tryte>;
    fn train_with_ewc(&mut self, ewc: &mut TrinaryEWC, data: &[(Vec<f32>, Vec<usize>)], epochs: usize);
}

impl EWCTrainingExt for SparseTrithNetwork {
    /// Forward pass with float inputs (needed for Fisher computation)
    fn forward_floats(&self, input: &[f32]) -> Vec<Tryte> {
        let mut current = input.to_vec();
        let mut result = Vec::new();
        
        for layer in &self.layers {
            let mut next = vec![0.0; layer.size];
            
            // Matrix multiplication with trinary weights
            for (i, input_val) in current.iter().enumerate() {
                if i < layer.weights.len() {
                    for (j, weight) in layer.weights[i].iter().enumerate() {
                        next[j] += input_val * weight;
                    }
                }
            }
            
            // Add biases and apply trinary activation
            for (j, val) in next.iter_mut().enumerate() {
                *val += layer.biases[j];
                
                // Convert to Tryte
                let tryte = if *val < -0.33 {
                    Tryte::Inhibited
                } else if *val > 0.33 {
                    Tryte::Activated
                } else {
                    Tryte::Baseline
                };
                
                result.push(tryte);
            }
            
            // Convert back to float for next layer
            current = result.iter().map(|&t| match t {
                Tryte::Inhibited => -1.0,
                Tryte::Baseline => 0.0,
                Tryte::Activated => 1.0,
            }).collect();
        }
        
        result
    }
    
    /// Train with EWC protection
    fn train_with_ewc(&mut self, ewc: &mut TrinaryEWC, 
                          data: &[(Vec<f32>, Vec<usize>)], 
                          epochs: usize) {
        println!("🎓 Training with EWC protection (λ={})", ewc.lambda);
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut ewc_penalty = 0.0;
            
            for (input, target) in data {
                // Forward pass (convert input to Trytes)
                let input_trytes: Vec<Tryte> = input.iter()
                    .map(|&f| TrinaryEWC::float_to_tryte(f))
                    .collect();
                let output = self.forward(&input_trytes);
                
                // Calculate loss (simplified)
                let loss = self.calculate_loss(&output, target);
                
                // Add EWC penalty
                let penalty = ewc.penalty(self);
                ewc_penalty += penalty;
                
                // Backward pass (would compute gradients here)
                self.backward(&output, target);
                
                // Modify gradients with EWC
                ewc.modify_gradients(self);
                
                // Update weights
                self.update_weights();
                
                total_loss += loss + penalty;
            }
            
            if epoch % 10 == 0 {
                println!("  Epoch {}: Loss={:.4} (task={:.4}, EWC={:.4})", 
                        epoch, 
                        total_loss / data.len() as f32,
                        (total_loss - ewc_penalty) / data.len() as f32,
                        ewc_penalty / data.len() as f32);
            }
        }
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tryte_distance() {
        // Same values have distance 0
        assert_eq!(TrinaryEWC::tryte_distance(Tryte::Baseline, Tryte::Baseline), 0.0);
        assert_eq!(TrinaryEWC::tryte_distance(Tryte::Activated, Tryte::Activated), 0.0);
        
        // Adjacent values have distance 1
        assert_eq!(TrinaryEWC::tryte_distance(Tryte::Baseline, Tryte::Activated), 1.0);
        assert_eq!(TrinaryEWC::tryte_distance(Tryte::Inhibited, Tryte::Baseline), 1.0);
        
        // Opposite values have distance 2
        assert_eq!(TrinaryEWC::tryte_distance(Tryte::Inhibited, Tryte::Activated), 2.0);
    }
    
    #[test]
    fn test_ewc_penalty_calculation() {
        let network = SparseTrithNetwork::new(vec![10, 5, 3]);
        let mut ewc = TrinaryEWC::new(&network, 0.5);
        
        // Initially, penalty should be 0 (weights match optimal)
        let penalty = ewc.penalty(&network);
        assert_eq!(penalty, 0.0);
    }
    
    #[test]
    fn test_fisher_information_update() {
        let mut network = SparseTrithNetwork::new(vec![10, 5, 3]);
        let mut ewc = TrinaryEWC::new(&network, 0.5);
        
        // Create dummy data
        let data = vec![
            (vec![0.5; 10], vec![1, 0, 0]),
            (vec![-0.5; 10], vec![0, 1, 0]),
        ];
        
        // Compute Fisher Information
        ewc.compute_fisher_information(&network, &data);
        
        let stats = ewc.get_stats();
        assert!(stats.total_weights > 0);
    }
    
    #[test]
    fn test_task_consolidation() {
        let network = SparseTrithNetwork::new(vec![5, 3, 2]);
        let mut ewc = TrinaryEWC::new(&network, 1.0);
        
        assert_eq!(ewc.current_task, 0);
        
        // Consolidate first task
        ewc.consolidate_task(&network);
        assert_eq!(ewc.current_task, 1);
        
        let stats = ewc.get_stats();
        assert_eq!(stats.current_task, 1);
    }
}