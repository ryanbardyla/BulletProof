//! Continual Learning Engine with Memory Protection
//! 
//! Implements lifelong learning that preserves knowledge across multiple tasks
//! using EWC, meta-learning, and biological memory consolidation.

use crate::trinary_ewc::{TrinaryEWC, EWCStats};
use crate::sparse_network_backprop::SparseTrithNetwork;
use crate::tryte::Tryte;
use crate::protein_synthesis::{ProteinSynthesisNeuron, ProteinType};
use std::collections::HashMap;
use anyhow::{Result, anyhow};

/// Task data for continual learning
#[derive(Clone, Debug)]
pub struct TaskData {
    pub task_id: usize,
    pub name: String,
    pub data: Vec<(Vec<f32>, Vec<usize>)>,
    pub epochs: usize,
    pub importance: f32,
    pub domain: String,  // e.g., "vision", "nlp", "control"
}

/// Performance metrics for a task
#[derive(Debug, Clone)]
pub struct TaskPerformance {
    pub task_id: usize,
    pub accuracy: f32,
    pub loss: f32,
    pub retention_score: f32,  // How well previous tasks are retained
    pub learning_speed: f32,   // Epochs needed to reach threshold
    pub memory_efficiency: f32, // Protected weights / total weights
}

/// Continual learning strategy
#[derive(Debug, Clone)]
pub enum LearningStrategy {
    /// Standard EWC with fixed lambda
    StandardEWC { lambda: f32 },
    /// Adaptive EWC with dynamic lambda based on task similarity
    AdaptiveEWC { base_lambda: f32, adaptation_rate: f32 },
    /// Experience replay with EWC
    ExperienceReplay { buffer_size: usize, replay_ratio: f32 },
    /// Meta-learning for fast adaptation
    MetaLearning { inner_lr: f32, outer_lr: f32 },
    /// Biological-inspired with protein synthesis
    BiologicalConsolidation { protein_threshold: f32, decay_rate: f32 },
}

/// Continual Learning Engine - The Core System
pub struct ContinualLearner {
    /// The neural network being trained
    pub network: SparseTrithNetwork,
    
    /// EWC memory protection system
    pub ewc: TrinaryEWC,
    
    /// Protein synthesis for biological memory
    pub protein_neurons: Vec<ProteinSynthesisNeuron>,
    
    /// Learning strategy
    pub strategy: LearningStrategy,
    
    /// Task history and performance
    pub task_history: Vec<TaskData>,
    pub performance_history: Vec<TaskPerformance>,
    
    /// Experience replay buffer
    pub experience_buffer: Vec<(Vec<f32>, Vec<usize>, usize)>, // input, label, task_id
    
    /// Task similarity matrix for adaptation
    pub task_similarity: HashMap<(usize, usize), f32>,
    
    /// Current task state
    pub current_task_id: usize,
    pub is_first_task: bool,
    
    /// Memory consolidation settings
    pub consolidation_threshold: f32,
    pub forgetting_threshold: f32,
    pub max_tasks: usize,
}

impl ContinualLearner {
    pub fn new(network_architecture: Vec<usize>, strategy: LearningStrategy) -> Self {
        let network = SparseTrithNetwork::new(network_architecture);
        let ewc = TrinaryEWC::new(&network, 1000.0); // High initial lambda
        let protein_neurons = (0..1000).map(|_| ProteinSynthesisNeuron::new()).collect();
        
        Self {
            network,
            ewc,
            protein_neurons,
            strategy,
            task_history: Vec::new(),
            performance_history: Vec::new(),
            experience_buffer: Vec::new(),
            task_similarity: HashMap::new(),
            current_task_id: 0,
            is_first_task: true,
            consolidation_threshold: 0.85, // 85% accuracy to consolidate
            forgetting_threshold: 0.7,     // <70% accuracy triggers intervention
            max_tasks: 50,                  // Maximum tasks to remember
        }
    }
    
    /// Train on a new task with memory protection
    pub fn train_with_memory_protection(&mut self, task: TaskData) -> Result<TaskPerformance> {
        println!("🧠 Starting continual learning on task: {}", task.name);
        println!("   Strategy: {:?}", self.strategy);
        
        // Update task state
        self.current_task_id = task.task_id;
        
        // Compute task similarity if not first task
        if !self.is_first_task {
            self.compute_task_similarity(&task)?;
            self.adapt_strategy_for_task(&task)?;
        }
        
        // Store previous task data for Fisher computation
        if !self.is_first_task {
            let previous_task = self.task_history.last().unwrap();
            println!("🧮 Computing Fisher Information from previous task: {}", previous_task.name);
            self.ewc.compute_fisher_information(&self.network, &previous_task.data);
            self.ewc.consolidate_task(&self.network);
        }
        
        // Initialize performance tracking
        let mut best_accuracy = 0.0;
        let mut epochs_to_threshold = task.epochs;
        let start_time = std::time::Instant::now();
        
        // Training loop with memory protection
        for epoch in 0..task.epochs {
            let mut epoch_loss = 0.0;
            let mut correct_predictions = 0;
            let mut total_predictions = 0;
            
            // Experience replay if enabled
            if let LearningStrategy::ExperienceReplay { replay_ratio, .. } = &self.strategy {
                self.replay_previous_experiences(*replay_ratio)?;
            }
            
            // Train on current task
            for (input, target) in &task.data {
                // Convert to trytes
                let input_trytes: Vec<Tryte> = input.iter()
                    .map(|&f| self.float_to_tryte(f))
                    .collect();
                
                // Forward pass
                let output = self.network.forward(&input_trytes);
                
                // Calculate base loss
                let task_loss = self.network.calculate_loss(&output, target);
                
                // Calculate EWC penalty (memory protection)
                let ewc_penalty = if self.is_first_task { 
                    0.0 
                } else { 
                    self.ewc.penalty(&self.network) 
                };
                
                // Calculate protein synthesis penalty (biological memory)
                let protein_penalty = self.calculate_protein_penalty(&input_trytes, &output);
                
                // Total loss with memory protection
                let total_loss = task_loss + ewc_penalty + protein_penalty;
                epoch_loss += total_loss;
                
                // Backward pass with EWC gradient modification
                self.network.backward(&output, target);
                if !self.is_first_task {
                    self.ewc.modify_gradients(&mut self.network);
                }
                
                // Update weights
                self.network.update_weights();
                
                // Track accuracy
                let predicted_class = self.get_predicted_class(&output);
                let target_class = self.get_target_class(target);
                if predicted_class == target_class {
                    correct_predictions += 1;
                }
                total_predictions += 1;
                
                // Add to experience buffer
                if let LearningStrategy::ExperienceReplay { buffer_size, .. } = &self.strategy {
                    self.add_to_experience_buffer(input.clone(), target.clone(), task.task_id, *buffer_size);
                }
            }
            
            // Calculate epoch metrics
            let epoch_accuracy = correct_predictions as f32 / total_predictions as f32;
            let avg_loss = epoch_loss / task.data.len() as f32;
            
            if epoch_accuracy > best_accuracy {
                best_accuracy = epoch_accuracy;
            }
            
            // Check if we reached learning threshold
            if epoch_accuracy > self.consolidation_threshold && epochs_to_threshold == task.epochs {
                epochs_to_threshold = epoch + 1;
            }
            
            // Update protein synthesis based on performance
            self.update_protein_synthesis(epoch_accuracy, task.importance);
            
            // Periodic reporting
            if epoch % 20 == 0 {
                let ewc_stats = self.ewc.get_stats();
                let current_ewc_penalty = if self.is_first_task { 0.0 } else { self.ewc.penalty(&self.network) };
                println!("  Epoch {}: Acc={:.3}, Loss={:.4}, EWC_penalty={:.4}, Protected={}/{}",
                        epoch, epoch_accuracy, avg_loss, current_ewc_penalty,
                        ewc_stats.protected_weights, ewc_stats.total_weights);
                
                // Check for catastrophic forgetting
                if epoch > 0 && epoch % 50 == 0 {
                    self.check_for_forgetting()?;
                }
            }
        }
        
        // Final performance evaluation
        let final_performance = self.evaluate_task_performance(&task)?;
        let learning_speed = epochs_to_threshold as f32 / task.epochs as f32;
        let memory_efficiency = self.ewc.get_stats().protection_ratio;
        
        // Test retention on previous tasks
        let retention_score = if self.is_first_task {
            1.0
        } else {
            self.test_retention_on_previous_tasks()?
        };
        
        let performance = TaskPerformance {
            task_id: task.task_id,
            accuracy: final_performance,
            loss: 0.0, // Would calculate final loss
            retention_score,
            learning_speed,
            memory_efficiency,
        };
        
        // Consolidate memory if performance is good
        if final_performance > self.consolidation_threshold {
            self.consolidate_task_memory(&task)?;
            println!("✅ Task {} consolidated with {:.1}% accuracy", task.name, final_performance * 100.0);
        } else {
            println!("⚠️  Task {} performance below threshold: {:.1}%", task.name, final_performance * 100.0);
        }
        
        // Update learning state
        self.task_history.push(task);
        self.performance_history.push(performance.clone());
        self.is_first_task = false;
        
        // Manage memory capacity
        if self.task_history.len() > self.max_tasks {
            self.prune_old_memories()?;
        }
        
        println!("🏆 Task completed in {:.2}s: Acc={:.1}%, Retention={:.1}%, Memory={:.1}%",
                start_time.elapsed().as_secs_f32(),
                final_performance * 100.0,
                retention_score * 100.0,
                memory_efficiency * 100.0);
        
        Ok(performance)
    }
    
    /// Adapt learning strategy based on task similarity
    fn adapt_strategy_for_task(&mut self, task: &TaskData) -> Result<()> {
        if let Some(previous_task) = self.task_history.last() {
            let similarity = self.task_similarity
                .get(&(previous_task.task_id, task.task_id))
                .copied()
                .unwrap_or(0.0);
            
            match &mut self.strategy {
                LearningStrategy::AdaptiveEWC { base_lambda, adaptation_rate } => {
                    // More similar tasks need less protection, dissimilar tasks need more
                    let new_lambda = *base_lambda * (1.0 + (1.0 - similarity) * *adaptation_rate);
                    self.ewc.set_lambda(new_lambda);
                    println!("   📊 Adapted EWC λ={:.1} (similarity={:.3})", new_lambda, similarity);
                }
                LearningStrategy::BiologicalConsolidation { protein_threshold, .. } => {
                    // Adjust protein threshold based on similarity
                    *protein_threshold = 0.5 + (1.0 - similarity) * 0.3;
                    println!("   🧬 Adapted protein threshold={:.3}", protein_threshold);
                }
                _ => {}
            }
        }
        Ok(())
    }
    
    /// Compute similarity between tasks using activation patterns
    fn compute_task_similarity(&mut self, new_task: &TaskData) -> Result<()> {
        if let Some(previous_task_data) = self.task_history.last().map(|t| (t.task_id, t.data.clone())) {
            // Sample activations from both tasks
            let prev_activations = self.sample_task_activations(&previous_task_data.1)?;
            let new_activations = self.sample_task_activations(&new_task.data)?;
            
            // Compute cosine similarity
            let similarity = self.cosine_similarity(&prev_activations, &new_activations);
            
            self.task_similarity.insert(
                (previous_task_data.0, new_task.task_id), 
                similarity
            );
            
            println!("   📈 Task similarity: {:.3}", similarity);
        }
        Ok(())
    }
    
    /// Sample activation patterns from task data
    fn sample_task_activations(&mut self, data: &[(Vec<f32>, Vec<usize>)]) -> Result<Vec<f32>> {
        let mut activations = Vec::new();
        
        // Sample up to 100 examples
        let sample_size = data.len().min(100);
        for i in 0..sample_size {
            let (input, _) = &data[i];
            let input_trytes: Vec<Tryte> = input.iter()
                .map(|&f| self.float_to_tryte(f))
                .collect();
            
            let output = self.network.forward(&input_trytes);
            
            // Convert trytes to float activations
            for tryte in output {
                activations.push(match tryte {
                    Tryte::Inhibited => -1.0,
                    Tryte::Baseline => 0.0,
                    Tryte::Activated => 1.0,
                });
            }
        }
        
        Ok(activations)
    }
    
    /// Compute cosine similarity between activation vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
    
    /// Experience replay from previous tasks
    fn replay_previous_experiences(&mut self, replay_ratio: f32) -> Result<()> {
        if self.experience_buffer.is_empty() {
            return Ok(());
        }
        
        let replay_count = (self.experience_buffer.len() as f32 * replay_ratio) as usize;
        
        for _ in 0..replay_count {
            let idx = fastrand::usize(..self.experience_buffer.len());
            let (input, target, _task_id) = &self.experience_buffer[idx];
            
            let input_trytes: Vec<Tryte> = input.iter()
                .map(|&f| self.float_to_tryte(f))
                .collect();
            
            let output = self.network.forward(&input_trytes);
            let loss = self.network.calculate_loss(&output, target);
            let ewc_penalty = self.ewc.penalty(&self.network);
            
            self.network.backward(&output, target);
            self.ewc.modify_gradients(&mut self.network);
            self.network.update_weights();
        }
        
        Ok(())
    }
    
    /// Add experience to replay buffer
    fn add_to_experience_buffer(&mut self, input: Vec<f32>, target: Vec<usize>, 
                               task_id: usize, buffer_size: usize) {
        self.experience_buffer.push((input, target, task_id));
        
        // Maintain buffer size
        if self.experience_buffer.len() > buffer_size {
            // Remove oldest experiences
            self.experience_buffer.remove(0);
        }
    }
    
    /// Calculate protein synthesis penalty for biological memory
    fn calculate_protein_penalty(&mut self, input: &[Tryte], output: &[Tryte]) -> f32 {
        if let LearningStrategy::BiologicalConsolidation { protein_threshold, decay_rate } = &self.strategy {
            // Calculate activation strength
            let activation_strength = output.iter()
                .map(|&t| match t {
                    Tryte::Activated => 1.0,
                    Tryte::Inhibited => 0.5,
                    Tryte::Baseline => 0.0,
                })
                .sum::<f32>() / output.len() as f32;
            
            if activation_strength > *protein_threshold {
                // Strong activation triggers protein synthesis in neurons
                let input_strength = input.iter().map(|&t| match t {
                    Tryte::Activated => 1.0,
                    Tryte::Inhibited => -1.0,
                    Tryte::Baseline => 0.0,
                }).sum::<f32>().abs();
                
                // Process through available protein neurons
                let neuron_idx = self.current_task_id % self.protein_neurons.len();
                if neuron_idx < self.protein_neurons.len() {
                    self.protein_neurons[neuron_idx].process_with_proteins(input_strength, 1);
                }
                
                // Return penalty proportional to protein synthesis cost
                return activation_strength * 0.1;
            }
        }
        
        0.0
    }
    
    /// Update protein synthesis based on learning performance
    fn update_protein_synthesis(&mut self, accuracy: f32, task_importance: f32) {
        if accuracy > self.consolidation_threshold {
            // Good performance triggers memory consolidation in protein neurons
            let consolidation_strength = accuracy * task_importance;
            let neuron_idx = self.current_task_id % self.protein_neurons.len();
            if neuron_idx < self.protein_neurons.len() {
                // Strengthen protein synthesis in this neuron
                self.protein_neurons[neuron_idx].process_with_proteins(consolidation_strength, 1);
            }
        }
    }
    
    /// Check for catastrophic forgetting on previous tasks
    fn check_for_forgetting(&mut self) -> Result<()> {
        let retention_score = self.test_retention_on_previous_tasks()?;
        
        if retention_score < self.forgetting_threshold {
            println!("🚨 Catastrophic forgetting detected! Retention: {:.1}%", retention_score * 100.0);
            
            // Trigger remediation
            match &mut self.strategy {
                LearningStrategy::StandardEWC { lambda } |
                LearningStrategy::AdaptiveEWC { base_lambda: lambda, .. } => {
                    *lambda *= 2.0; // Increase regularization
                    println!("   🛡️  Increased EWC λ to {}", lambda);
                }
                _ => {}
            }
            
            // Replay critical memories
            if !self.experience_buffer.is_empty() {
                println!("   🔄 Replaying critical memories...");
                self.replay_previous_experiences(0.5)?; // Replay 50% of buffer
            }
        }
        
        Ok(())
    }
    
    /// Test retention on all previous tasks
    fn test_retention_on_previous_tasks(&mut self) -> Result<f32> {
        if self.task_history.is_empty() {
            return Ok(1.0);
        }
        
        let mut total_accuracy = 0.0;
        let mut task_count = 0;
        
        let tasks = self.task_history.clone();
        for task in &tasks {
            let accuracy = self.evaluate_task_performance(task)?;
            total_accuracy += accuracy;
            task_count += 1;
        }
        
        Ok(total_accuracy / task_count as f32)
    }
    
    /// Evaluate performance on a specific task
    fn evaluate_task_performance(&mut self, task: &TaskData) -> Result<f32> {
        let mut correct = 0;
        let mut total = 0;
        
        // Test on a subset of task data
        let test_size = task.data.len().min(100);
        for i in 0..test_size {
            let (input, target) = &task.data[i];
            let input_trytes: Vec<Tryte> = input.iter()
                .map(|&f| self.float_to_tryte(f))
                .collect();
            
            let output = self.network.forward(&input_trytes);
            
            let predicted_class = self.get_predicted_class(&output);
            let target_class = self.get_target_class(target);
            
            if predicted_class == target_class {
                correct += 1;
            }
            total += 1;
        }
        
        Ok(correct as f32 / total as f32)
    }
    
    /// Consolidate task memory for long-term storage
    fn consolidate_task_memory(&mut self, task: &TaskData) -> Result<()> {
        println!("🧠 Consolidating memory for task: {}", task.name);
        
        // Update EWC with final Fisher information
        self.ewc.consolidate_task(&self.network);
        
        // Protein synthesis consolidation - strengthen associated neurons
        let neuron_idx = task.task_id % self.protein_neurons.len();
        if neuron_idx < self.protein_neurons.len() {
            let consolidation_strength = task.importance * 2.0; // Strong consolidation
            self.protein_neurons[neuron_idx].process_with_proteins(consolidation_strength, 1);
        }
        
        // Store critical examples in experience buffer
        if let LearningStrategy::ExperienceReplay { .. } = &self.strategy {
            // Add most challenging examples to buffer
            for (input, target) in &task.data {
                if fastrand::f32() < 0.1 { // Store 10% of examples
                    self.experience_buffer.push((input.clone(), target.clone(), task.task_id));
                }
            }
        }
        
        println!("   ✅ Memory consolidated for task {}", task.task_id);
        Ok(())
    }
    
    /// Prune old memories when capacity is exceeded
    fn prune_old_memories(&mut self) -> Result<()> {
        println!("🗂️  Pruning old memories (capacity exceeded)...");
        
        // Keep most recent and most important tasks
        self.task_history.sort_by(|a, b| {
            let importance_cmp = b.importance.partial_cmp(&a.importance).unwrap();
            if importance_cmp == std::cmp::Ordering::Equal {
                b.task_id.cmp(&a.task_id) // More recent if equal importance
            } else {
                importance_cmp
            }
        });
        
        // Keep top N tasks
        let keep_count = (self.max_tasks * 3) / 4; // Keep 75%
        self.task_history.truncate(keep_count);
        self.performance_history.truncate(keep_count);
        
        // Prune experience buffer accordingly
        let valid_task_ids: std::collections::HashSet<_> = 
            self.task_history.iter().map(|t| t.task_id).collect();
        
        self.experience_buffer.retain(|(_, _, task_id)| {
            valid_task_ids.contains(task_id)
        });
        
        println!("   ✅ Pruned to {} tasks", self.task_history.len());
        Ok(())
    }
    
    /// Get predicted class from network output
    fn get_predicted_class(&self, output: &[Tryte]) -> usize {
        output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let a_val = match a {
                    Tryte::Activated => 1.0,
                    Tryte::Baseline => 0.0,
                    Tryte::Inhibited => -1.0,
                };
                let b_val = match b {
                    Tryte::Activated => 1.0,
                    Tryte::Baseline => 0.0,
                    Tryte::Inhibited => -1.0,
                };
                a_val.partial_cmp(&b_val).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
    
    /// Get target class from one-hot encoded target
    fn get_target_class(&self, target: &[usize]) -> usize {
        target.iter()
            .enumerate()
            .find(|(_, &val)| val == 1)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
    
    /// Convert float to Tryte
    fn float_to_tryte(&self, f: f32) -> Tryte {
        if f < -0.33 {
            Tryte::Inhibited
        } else if f > 0.33 {
            Tryte::Activated
        } else {
            Tryte::Baseline
        }
    }
    
    /// Get comprehensive learning statistics
    pub fn get_learning_stats(&self) -> LearningStats {
        let total_accuracy = if self.performance_history.is_empty() {
            0.0
        } else {
            self.performance_history.iter().map(|p| p.accuracy).sum::<f32>() 
                / self.performance_history.len() as f32
        };
        
        let avg_retention = if self.performance_history.is_empty() {
            1.0
        } else {
            self.performance_history.iter().map(|p| p.retention_score).sum::<f32>() 
                / self.performance_history.len() as f32
        };
        
        let memory_usage = self.ewc.get_stats().protection_ratio;
        
        LearningStats {
            total_tasks_learned: self.task_history.len(),
            average_accuracy: total_accuracy,
            average_retention: avg_retention,
            memory_efficiency: memory_usage,
            experience_buffer_size: self.experience_buffer.len(),
            protected_weights: self.ewc.get_stats().protected_weights,
            total_weights: self.ewc.get_stats().total_weights,
            current_lambda: self.ewc.get_lambda(),
        }
    }
}

/// Comprehensive learning statistics
#[derive(Debug)]
pub struct LearningStats {
    pub total_tasks_learned: usize,
    pub average_accuracy: f32,
    pub average_retention: f32,
    pub memory_efficiency: f32,
    pub experience_buffer_size: usize,
    pub protected_weights: usize,
    pub total_weights: usize,
    pub current_lambda: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_continual_learner_creation() {
        let strategy = LearningStrategy::StandardEWC { lambda: 1000.0 };
        let learner = ContinualLearner::new(vec![10, 5, 3], strategy);
        
        assert_eq!(learner.current_task_id, 0);
        assert!(learner.is_first_task);
        assert_eq!(learner.task_history.len(), 0);
    }
    
    #[test]
    fn test_task_similarity_computation() {
        let mut learner = ContinualLearner::new(
            vec![4, 3, 2], 
            LearningStrategy::AdaptiveEWC { base_lambda: 1000.0, adaptation_rate: 0.5 }
        );
        
        // Create similar tasks
        let task1 = TaskData {
            task_id: 1,
            name: "Task1".to_string(),
            data: vec![
                (vec![1.0, 0.0, 0.0, 0.0], vec![1, 0]),
                (vec![0.0, 1.0, 0.0, 0.0], vec![0, 1]),
            ],
            epochs: 10,
            importance: 1.0,
            domain: "test".to_string(),
        };
        
        let task2 = TaskData {
            task_id: 2,
            name: "Task2".to_string(),
            data: vec![
                (vec![1.0, 1.0, 0.0, 0.0], vec![1, 0]),
                (vec![0.0, 0.0, 1.0, 0.0], vec![0, 1]),
            ],
            epochs: 10,
            importance: 1.0,
            domain: "test".to_string(),
        };
        
        // Train on first task
        learner.task_history.push(task1);
        learner.is_first_task = false;
        
        // Compute similarity with second task
        let result = learner.compute_task_similarity(&task2);
        assert!(result.is_ok());
    }
}