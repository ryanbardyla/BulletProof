// 🧠 ELASTIC WEIGHT CONSOLIDATION & META-LEARNING
// Compile-time support to prevent catastrophic forgetting!

use std::collections::HashMap;

/// Fisher Information Matrix for EWC
/// Tracks importance of each weight for previous tasks
#[derive(Debug, Clone)]
pub struct FisherMatrix {
    pub weights: HashMap<String, Vec<f64>>,
    pub importance: HashMap<String, f64>,
}

/// Meta-Learning parameters for fast adaptation
#[derive(Debug, Clone)]
pub struct MetaLearner {
    pub inner_lr: f64,      // Inner loop learning rate
    pub outer_lr: f64,      // Outer loop learning rate  
    pub adaptation_steps: usize,
    pub meta_batch_size: usize,
}

/// EWC compile-time attributes
#[derive(Debug, Clone)]
pub struct EWCAttribute {
    pub lambda: f64,        // EWC regularization strength
    pub preserve: bool,     // Preserve these weights
    pub task_id: usize,     // Task this weight belongs to
}

impl FisherMatrix {
    pub fn new() -> Self {
        FisherMatrix {
            weights: HashMap::new(),
            importance: HashMap::new(),
        }
    }
    
    /// Calculate Fisher Information for a weight
    pub fn calculate_importance(&mut self, weight_name: &str, gradients: &[f64]) {
        // Fisher Information = E[gradient^2]
        let fisher_value = gradients.iter()
            .map(|g| g * g)
            .sum::<f64>() / gradients.len() as f64;
            
        self.importance.insert(weight_name.to_string(), fisher_value);
    }
    
    /// Generate EWC loss term at compile time
    pub fn generate_ewc_loss_code(&self, current_weights: &HashMap<String, f64>, lambda: f64) -> String {
        let mut code = String::new();
        
        code.push_str("// EWC Regularization Term\n");
        code.push_str("let ewc_loss = 0;\n");
        
        for (name, importance) in &self.importance {
            if let Some(prev_weight) = self.weights.get(name) {
                if let Some(curr_weight) = current_weights.get(name) {
                    // L_EWC = λ/2 * Σ F_i * (θ_i - θ*_i)^2
                    code.push_str(&format!(
                        "ewc_loss = ewc_loss + {} * {} * pow({} - {}, 2);\n",
                        lambda / 2.0,
                        importance,
                        curr_weight,
                        prev_weight[0] // Assuming single value for simplicity
                    ));
                }
            }
        }
        
        code
    }
}

impl MetaLearner {
    pub fn new() -> Self {
        MetaLearner {
            inner_lr: 0.01,
            outer_lr: 0.001,
            adaptation_steps: 5,
            meta_batch_size: 4,
        }
    }
    
    /// Generate MAML (Model-Agnostic Meta-Learning) code
    pub fn generate_maml_code(&self) -> String {
        let mut code = String::new();
        
        code.push_str("// MAML Meta-Learning\n");
        code.push_str("organism MetaLearningNetwork {\n");
        code.push_str("    // Meta-parameters (learned across tasks)\n");
        code.push_str(&format!("    gene inner_lr = {}\n", self.inner_lr));
        code.push_str(&format!("    gene outer_lr = {}\n", self.outer_lr));
        code.push_str("\n");
        
        code.push_str("    fn meta_train(tasks) {\n");
        code.push_str("        let meta_gradients = []\n");
        code.push_str("\n");
        
        code.push_str("        // Outer loop (meta-update)\n");
        code.push_str(&format!("        for task_batch = 0 {{\n")); // Simplified
        code.push_str("            let task_losses = []\n");
        code.push_str("\n");
        
        code.push_str("            // Inner loop (task-specific adaptation)\n");
        code.push_str(&format!("            for step = 0 {{\n")); // adaptation_steps
        code.push_str("                // Fast adaptation with inner_lr\n");
        code.push_str("                let adapted_weights = weights\n");
        code.push_str(&format!("                for i = 0 {{\n")); // inner adaptation
        code.push_str("                    let gradient = compute_gradient(task_data)\n");
        code.push_str(&format!("                    adapted_weights = adapted_weights - {} * gradient\n", self.inner_lr));
        code.push_str("                }\n");
        code.push_str("\n");
        
        code.push_str("                // Evaluate on query set\n");
        code.push_str("                let query_loss = evaluate(adapted_weights, query_data)\n");
        code.push_str("                task_losses.push(query_loss)\n");
        code.push_str("            }\n");
        code.push_str("\n");
        
        code.push_str("            // Meta-gradient computation\n");
        code.push_str("            let meta_grad = compute_meta_gradient(task_losses)\n");
        code.push_str("            meta_gradients.push(meta_grad)\n");
        code.push_str("        }\n");
        code.push_str("\n");
        
        code.push_str("        // Meta-update with outer_lr\n");
        code.push_str(&format!("        weights = weights - {} * average(meta_gradients)\n", self.outer_lr));
        code.push_str("    }\n");
        
        code.push_str("\n");
        code.push_str("    fn adapt_to_new_task(task_data) {\n");
        code.push_str("        // Few-shot learning: adapt quickly with just a few examples\n");
        code.push_str("        let adapted = weights\n");
        code.push_str(&format!("        for i = 0 {{\n"));
        code.push_str(&format!("            adapted = adapted - {} * compute_gradient(task_data)\n", self.inner_lr));
        code.push_str("        }\n");
        code.push_str("        return adapted\n");
        code.push_str("    }\n");
        
        code.push_str("}\n");
        
        code
    }
}

/// Compile-time directive processor for EWC and Meta-Learning
pub struct CompileTimeML {
    pub ewc_enabled: bool,
    pub meta_learning_enabled: bool,
    pub fisher_matrix: FisherMatrix,
    pub meta_learner: MetaLearner,
}

impl CompileTimeML {
    pub fn new() -> Self {
        CompileTimeML {
            ewc_enabled: true,
            meta_learning_enabled: true,
            fisher_matrix: FisherMatrix::new(),
            meta_learner: MetaLearner::new(),
        }
    }
    
    /// Process compile-time directives like @ewc and @meta
    pub fn process_directive(&mut self, directive: &str) -> Option<String> {
        if directive.starts_with("@ewc") {
            // Enable EWC for this network
            self.ewc_enabled = true;
            
            // Parse lambda if provided: @ewc(lambda=0.5)
            if let Some(lambda_str) = directive.split("lambda=").nth(1) {
                let lambda: f64 = lambda_str.split(')').next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.5);
                
                return Some(format!("// EWC enabled with lambda={}\n", lambda));
            }
            
            Some("// EWC enabled\n".to_string())
        } else if directive.starts_with("@meta") {
            // Enable meta-learning
            self.meta_learning_enabled = true;
            
            // Generate MAML code
            Some(self.meta_learner.generate_maml_code())
        } else if directive.starts_with("@preserve") {
            // Mark weights as important (should not be forgotten)
            Some("// Weights marked for preservation\n".to_string())
        } else {
            None
        }
    }
    
    /// Generate compile-time optimized code
    pub fn generate_optimized_network(&self, base_code: &str) -> String {
        let mut optimized = String::new();
        
        // Add EWC support if enabled
        if self.ewc_enabled {
            optimized.push_str("// EWC-Protected Neural Network\n");
            optimized.push_str("// This network remembers previous tasks!\n\n");
            
            // Inject Fisher Information tracking
            optimized.push_str("protein FisherInfo {\n");
            optimized.push_str("    importance: Array,\n");
            optimized.push_str("    prev_weights: Array\n");
            optimized.push_str("}\n\n");
        }
        
        // Add meta-learning if enabled
        if self.meta_learning_enabled {
            optimized.push_str("// Meta-Learning Enabled\n");
            optimized.push_str("// Fast adaptation to new tasks!\n\n");
            optimized.push_str(&self.meta_learner.generate_maml_code());
            optimized.push_str("\n");
        }
        
        // Add the base network code
        optimized.push_str(base_code);
        
        // Add EWC loss calculation
        if self.ewc_enabled {
            optimized.push_str("\n// EWC Loss Calculation\n");
            optimized.push_str("fn calculate_total_loss(task_loss) {\n");
            optimized.push_str("    let ewc_loss = calculate_ewc_penalty()\n");
            optimized.push_str("    return task_loss + ewc_loss\n");
            optimized.push_str("}\n");
        }
        
        optimized
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fisher_matrix() {
        let mut fisher = FisherMatrix::new();
        let gradients = vec![0.1, -0.2, 0.15, -0.05];
        
        fisher.calculate_importance("weight1", &gradients);
        
        assert!(fisher.importance.contains_key("weight1"));
        assert!(fisher.importance["weight1"] > 0.0);
    }
    
    #[test]
    fn test_meta_learner_code_gen() {
        let meta = MetaLearner::new();
        let code = meta.generate_maml_code();
        
        assert!(code.contains("MAML"));
        assert!(code.contains("meta_train"));
        assert!(code.contains("adapt_to_new_task"));
    }
}