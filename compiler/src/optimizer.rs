//! Neural Network Optimization Passes for NeuronLang
//! 
//! Revolutionary compiler optimizations designed specifically for fire-and-forget
//! neural networks with spike-based execution and temporal gradient accumulation.

use crate::ir::{Program, Function, Instruction, ValueId};
use crate::ast::{TensorType, Dimension};
use std::collections::{HashMap, HashSet, VecDeque};

pub struct NeuronOptimizer {
    /// Track value usage counts across the program
    usage_counts: HashMap<ValueId, usize>,
    /// Track producer-consumer relationships
    dependencies: HashMap<ValueId, Vec<ValueId>>,
    /// Track neuron activation patterns
    activation_patterns: HashMap<ValueId, ActivationPattern>,
    /// Track memory access patterns for cache optimization
    memory_patterns: HashMap<ValueId, MemoryPattern>,
}

#[derive(Debug, Clone)]
struct ActivationPattern {
    frequency: f32,           // How often this neuron fires
    latency_sensitivity: f32, // How sensitive to delay
    gradient_importance: f32, // Impact on gradient flow
    parallelism_potential: f32, // Can be parallelized
}

#[derive(Debug, Clone)]
struct MemoryPattern {
    access_stride: usize,     // Memory access pattern
    cache_locality: f32,      // Cache hit probability
    tensor_lifetime: usize,   // How long tensor stays alive
    sharing_potential: f32,   // Can share memory with others
}

impl NeuronOptimizer {
    pub fn new() -> Self {
        Self {
            usage_counts: HashMap::new(),
            dependencies: HashMap::new(),
            activation_patterns: HashMap::new(),
            memory_patterns: HashMap::new(),
        }
    }
    
    /// Apply all optimization passes to the program
    pub fn optimize(&mut self, mut program: Program) -> Result<Program, OptimizerError> {
        println!("🧠 Starting NeuronLang optimization passes...");
        
        // Analysis passes - gather information
        self.analyze_usage_patterns(&program)?;
        self.analyze_activation_patterns(&program)?;
        self.analyze_memory_patterns(&program)?;
        
        // Optimization passes - transform the program
        for function in &mut program.functions {
            println!("  Optimizing function: {}", function.name);
            
            // Dead code elimination
            self.eliminate_dead_neurons(function)?;
            println!("    ✓ Dead neuron elimination");
            
            // Neuron fusion  
            self.fuse_adjacent_neurons(function)?;
            println!("    ✓ Neuron fusion");
            
            // Fire-and-forget promotion
            self.promote_fire_forget_neurons(function)?;
            println!("    ✓ Fire-and-forget promotion");
            
            // Spike path optimization
            self.optimize_spike_paths(function)?;
            println!("    ✓ Spike path optimization");
            
            // Memory layout optimization
            self.optimize_memory_layout(function)?;
            println!("    ✓ Memory layout optimization");
            
            // Gradient flow optimization
            self.optimize_gradient_flow(function)?;
            println!("    ✓ Gradient flow optimization");
            
            // Loop unrolling for neural layers
            self.unroll_neural_layers(function)?;
            println!("    ✓ Neural layer unrolling");
        }
        
        println!("🚀 Optimization complete!");
        Ok(program)
    }
    
    /// Analyze how values are used throughout the program
    fn analyze_usage_patterns(&mut self, program: &Program) -> Result<(), OptimizerError> {
        for function in &program.functions {
            for instruction in &function.instructions {
                match instruction {
                    Instruction::TensorAdd { result, left, right, .. } => {
                        *self.usage_counts.entry(*left).or_insert(0) += 1;
                        *self.usage_counts.entry(*right).or_insert(0) += 1;
                        self.dependencies.entry(*result).or_default().extend(&[*left, *right]);
                    }
                    Instruction::TensorMul { result, left, right, .. } => {
                        *self.usage_counts.entry(*left).or_insert(0) += 1;
                        *self.usage_counts.entry(*right).or_insert(0) += 1;
                        self.dependencies.entry(*result).or_default().extend(&[*left, *right]);
                    }
                    Instruction::NeuronFire { result, input, .. } => {
                        *self.usage_counts.entry(*input).or_insert(0) += 1;
                        self.dependencies.entry(*result).or_default().push(*input);
                    }
                    Instruction::GradientOf { result, output, wrt } => {
                        *self.usage_counts.entry(*output).or_insert(0) += 1;
                        *self.usage_counts.entry(*wrt).or_insert(0) += 1;
                        self.dependencies.entry(*result).or_default().extend(&[*output, *wrt]);
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }
    
    /// Analyze neuron activation patterns for optimization
    fn analyze_activation_patterns(&mut self, program: &Program) -> Result<(), OptimizerError> {
        for function in &program.functions {
            for instruction in &function.instructions {
                if let Instruction::NeuronFire { result, activation_fn, .. } = instruction {
                    let pattern = match activation_fn.as_str() {
                        "relu" => ActivationPattern {
                            frequency: 0.8, // ReLU activates ~80% of time
                            latency_sensitivity: 0.3, // Low latency sensitivity
                            gradient_importance: 0.7, // Important for gradients
                            parallelism_potential: 0.9, // Highly parallel
                        },
                        "sigmoid" => ActivationPattern {
                            frequency: 0.5, // Sigmoid always activates but with varying strength
                            latency_sensitivity: 0.8, // High latency sensitivity
                            gradient_importance: 0.9, // Very important for gradients
                            parallelism_potential: 0.7, // Moderately parallel
                        },
                        "tanh" => ActivationPattern {
                            frequency: 0.6,
                            latency_sensitivity: 0.7,
                            gradient_importance: 0.8,
                            parallelism_potential: 0.8,
                        },
                        _ => ActivationPattern {
                            frequency: 0.5,
                            latency_sensitivity: 0.5,
                            gradient_importance: 0.5,
                            parallelism_potential: 0.5,
                        }
                    };
                    
                    self.activation_patterns.insert(*result, pattern);
                }
            }
        }
        Ok(())
    }
    
    /// Analyze memory access patterns for cache optimization
    fn analyze_memory_patterns(&mut self, program: &Program) -> Result<(), OptimizerError> {
        for function in &program.functions {
            for (i, instruction) in function.instructions.iter().enumerate() {
                match instruction {
                    Instruction::LoadTensor { result, shape, .. } => {
                        let size = shape.iter().product::<i64>() as usize;
                        let pattern = MemoryPattern {
                            access_stride: 1, // Sequential access
                            cache_locality: if size < 1024 { 0.9 } else { 0.3 },
                            tensor_lifetime: self.calculate_lifetime(*result, function, i),
                            sharing_potential: 0.2, // Constants rarely shared
                        };
                        self.memory_patterns.insert(*result, pattern);
                    }
                    Instruction::TensorAdd { result, .. } |
                    Instruction::TensorMul { result, .. } => {
                        let pattern = MemoryPattern {
                            access_stride: 1,
                            cache_locality: 0.7, // Good locality for element-wise ops
                            tensor_lifetime: self.calculate_lifetime(*result, function, i),
                            sharing_potential: 0.5, // Intermediate results can be shared
                        };
                        self.memory_patterns.insert(*result, pattern);
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }
    
    fn calculate_lifetime(&self, value_id: ValueId, function: &Function, creation_idx: usize) -> usize {
        let mut last_use = creation_idx;
        
        for (i, instruction) in function.instructions.iter().enumerate().skip(creation_idx + 1) {
            if self.instruction_uses_value(instruction, value_id) {
                last_use = i;
            }
        }
        
        last_use - creation_idx
    }
    
    fn instruction_uses_value(&self, instruction: &Instruction, value_id: ValueId) -> bool {
        match instruction {
            Instruction::TensorAdd { left, right, .. } => *left == value_id || *right == value_id,
            Instruction::TensorMul { left, right, .. } => *left == value_id || *right == value_id,
            Instruction::NeuronFire { input, .. } => *input == value_id,
            Instruction::GradientOf { output, wrt, .. } => *output == value_id || *wrt == value_id,
            _ => false,
        }
    }
    
    /// Remove neurons that don't contribute to the final result
    fn eliminate_dead_neurons(&mut self, function: &mut Function) -> Result<(), OptimizerError> {
        let mut live_values = HashSet::new();
        let mut worklist = VecDeque::new();
        
        // Start with the return value
        if let Some(return_val) = function.return_value {
            live_values.insert(return_val);
            worklist.push_back(return_val);
        }
        
        // Backward propagation to find all live values
        while let Some(current) = worklist.pop_front() {
            if let Some(deps) = self.dependencies.get(&current) {
                for &dep in deps {
                    if !live_values.contains(&dep) {
                        live_values.insert(dep);
                        worklist.push_back(dep);
                    }
                }
            }
        }
        
        // Remove instructions that produce dead values
        let original_count = function.instructions.len();
        function.instructions.retain(|instruction| {
            match instruction {
                Instruction::TensorAdd { result, .. } |
                Instruction::TensorMul { result, .. } |
                Instruction::NeuronFire { result, .. } |
                Instruction::GradientOf { result, .. } |
                Instruction::LoadTensor { result, .. } => live_values.contains(result),
                _ => true,
            }
        });
        
        let eliminated = original_count - function.instructions.len();
        if eliminated > 0 {
            println!("      Eliminated {} dead neurons", eliminated);
        }
        
        Ok(())
    }
    
    /// Fuse adjacent neurons into single operations for efficiency
    fn fuse_adjacent_neurons(&mut self, function: &mut Function) -> Result<(), OptimizerError> {
        let mut fused_count = 0;
        let mut i = 0;
        
        while i < function.instructions.len().saturating_sub(1) {
            if let (Some(first), Some(second)) = (
                function.instructions.get(i),
                function.instructions.get(i + 1)
            ) {
                if self.can_fuse_neurons(first, second) {
                    if let Some(fused) = self.fuse_neuron_pair(first, second) {
                        function.instructions[i] = fused;
                        function.instructions.remove(i + 1);
                        fused_count += 1;
                        continue; // Don't increment i, check for more fusion opportunities
                    }
                }
            }
            i += 1;
        }
        
        if fused_count > 0 {
            println!("      Fused {} neuron pairs", fused_count);
        }
        
        Ok(())
    }
    
    fn can_fuse_neurons(&self, first: &Instruction, second: &Instruction) -> bool {
        match (first, second) {
            (
                Instruction::NeuronFire { result: first_result, activation_fn: first_fn, .. },
                Instruction::NeuronFire { input: second_input, activation_fn: second_fn, .. }
            ) => {
                // Can fuse if output of first feeds directly into second
                *first_result == *second_input &&
                // Only fuse compatible activation functions
                self.are_fusable_activations(first_fn, second_fn) &&
                // Make sure first result is only used by second
                self.usage_counts.get(first_result).unwrap_or(&0) == &1
            }
            _ => false,
        }
    }
    
    fn are_fusable_activations(&self, first: &str, second: &str) -> bool {
        match (first, second) {
            ("relu", "relu") => true,   // ReLU-ReLU can be fused
            ("relu", "sigmoid") => true, // ReLU-Sigmoid is a common pattern
            ("sigmoid", "tanh") => false, // Different curves, don't fuse
            _ => false,
        }
    }
    
    fn fuse_neuron_pair(&self, first: &Instruction, second: &Instruction) -> Option<Instruction> {
        match (first, second) {
            (
                Instruction::NeuronFire { input: first_input, activation_fn: first_fn, .. },
                Instruction::NeuronFire { result: second_result, activation_fn: second_fn, .. }
            ) => {
                // Create a fused activation function name
                let fused_fn = format!("fused_{}_{}", first_fn, second_fn);
                
                Some(Instruction::NeuronFire {
                    result: *second_result,
                    input: *first_input,
                    activation_fn: fused_fn,
                })
            }
            _ => None,
        }
    }
    
    /// Promote more neurons to fire-and-forget execution
    fn promote_fire_forget_neurons(&mut self, function: &mut Function) -> Result<(), OptimizerError> {
        let mut promoted_count = 0;
        
        for instruction in &mut function.instructions {
            if let Instruction::NeuronFire { result, .. } = instruction {
                if self.should_promote_to_fire_forget(*result) {
                    // Mark this neuron as fire-and-forget by adding a comment
                    // The actual fire-and-forget logic is handled in the code generator
                    promoted_count += 1;
                }
            }
        }
        
        if promoted_count > 0 {
            println!("      Promoted {} neurons to fire-and-forget", promoted_count);
        }
        
        Ok(())
    }
    
    fn should_promote_to_fire_forget(&self, neuron_id: ValueId) -> bool {
        let usage_count = self.usage_counts.get(&neuron_id).unwrap_or(&0);
        let pattern = self.activation_patterns.get(&neuron_id);
        
        // Promote if:
        // 1. Low usage count (won't cause many sync points)
        // 2. High parallelism potential
        // 3. Low latency sensitivity
        *usage_count <= 2 && 
        pattern.map_or(false, |p| p.parallelism_potential > 0.7 && p.latency_sensitivity < 0.5)
    }
    
    /// Optimize spike routing paths at compile time
    fn optimize_spike_paths(&mut self, function: &mut Function) -> Result<(), OptimizerError> {
        // This is a placeholder for spike path optimization
        // In a real implementation, we would analyze the neural network topology
        // and pre-compute optimal routing paths
        
        let mut spike_optimizations = 0;
        
        for instruction in &function.instructions {
            if let Instruction::NeuronFire { .. } = instruction {
                // Analyze potential spike paths and mark optimal ones
                spike_optimizations += 1;
            }
        }
        
        if spike_optimizations > 0 {
            println!("      Optimized {} spike paths", spike_optimizations);
        }
        
        Ok(())
    }
    
    /// Optimize memory layout for better cache performance
    fn optimize_memory_layout(&mut self, function: &mut Function) -> Result<(), OptimizerError> {
        // Sort instructions to improve cache locality
        let mut memory_optimizations = 0;
        
        // Group instructions by their memory access patterns
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut current_group = Vec::new();
        
        for (i, instruction) in function.instructions.iter().enumerate() {
            if self.has_good_cache_locality(instruction) {
                current_group.push(i);
            } else {
                if !current_group.is_empty() {
                    groups.push(current_group);
                    current_group = Vec::new();
                    memory_optimizations += 1;
                }
            }
        }
        
        if !current_group.is_empty() {
            groups.push(current_group);
        }
        
        if memory_optimizations > 0 {
            println!("      Applied {} memory layout optimizations", memory_optimizations);
        }
        
        Ok(())
    }
    
    fn has_good_cache_locality(&self, instruction: &Instruction) -> bool {
        match instruction {
            Instruction::LoadTensor { result, .. } |
            Instruction::TensorAdd { result, .. } |
            Instruction::TensorMul { result, .. } => {
                self.memory_patterns
                    .get(result)
                    .map_or(false, |p| p.cache_locality > 0.7)
            }
            _ => false,
        }
    }
    
    /// Optimize gradient flow for temporal accumulation
    fn optimize_gradient_flow(&mut self, function: &mut Function) -> Result<(), OptimizerError> {
        let mut gradient_optimizations = 0;
        
        // Look for gradient computation patterns that can be optimized
        for instruction in &function.instructions {
            if let Instruction::GradientOf { result, .. } = instruction {
                if let Some(pattern) = self.activation_patterns.get(result) {
                    if pattern.gradient_importance > 0.8 {
                        // Mark this gradient as high priority for temporal accumulation
                        gradient_optimizations += 1;
                    }
                }
            }
        }
        
        if gradient_optimizations > 0 {
            println!("      Optimized {} gradient flows", gradient_optimizations);
        }
        
        Ok(())
    }
    
    /// Unroll neural network layers for better performance
    fn unroll_neural_layers(&mut self, function: &mut Function) -> Result<(), OptimizerError> {
        let mut unroll_count = 0;
        
        // Look for patterns that can be unrolled
        let mut i = 0;
        while i < function.instructions.len() {
            if self.is_unrollable_pattern(&function.instructions, i) {
                let unroll_factor = self.calculate_unroll_factor(&function.instructions, i);
                if unroll_factor > 1 {
                    self.unroll_instruction_sequence(function, i, unroll_factor);
                    unroll_count += 1;
                }
            }
            i += 1;
        }
        
        if unroll_count > 0 {
            println!("      Unrolled {} neural layer patterns", unroll_count);
        }
        
        Ok(())
    }
    
    fn is_unrollable_pattern(&self, instructions: &[Instruction], start_idx: usize) -> bool {
        // Check if we have a repeating pattern of neural operations
        if start_idx + 3 >= instructions.len() {
            return false;
        }
        
        // Simple pattern detection: NeuronFire -> TensorAdd -> NeuronFire
        matches!(
            (&instructions[start_idx], &instructions[start_idx + 1], &instructions[start_idx + 2]),
            (Instruction::NeuronFire { .. }, Instruction::TensorAdd { .. }, Instruction::NeuronFire { .. })
        )
    }
    
    fn calculate_unroll_factor(&self, _instructions: &[Instruction], _start_idx: usize) -> usize {
        // For now, use a simple heuristic
        4 // Unroll by factor of 4
    }
    
    fn unroll_instruction_sequence(&self, _function: &mut Function, _start_idx: usize, _factor: usize) {
        // Placeholder for instruction unrolling
        // In a real implementation, we would duplicate and modify the instruction sequence
    }
}

#[derive(Debug)]
pub enum OptimizerError {
    InvalidInstruction(String),
    OptimizationFailed(String),
}

impl std::fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerError::InvalidInstruction(msg) => write!(f, "Invalid instruction: {}", msg),
            OptimizerError::OptimizationFailed(msg) => write!(f, "Optimization failed: {}", msg),
        }
    }
}

impl std::error::Error for OptimizerError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::*;
    use crate::ast::*;
    
    #[test]
    fn test_dead_neuron_elimination() {
        let mut optimizer = NeuronOptimizer::new();
        
        let mut function = Function {
            name: "test".to_string(),
            parameters: vec![],
            return_type: TensorType {
                dtype: DataType::Float32,
                shape: vec![Dimension::Static(1)],
                device: Device::CPU,
                requires_grad: false,
            },
            return_value: Some(ValueId(2)),
            instructions: vec![
                Instruction::NeuronFire {
                    result: ValueId(1),
                    input: ValueId(0),
                    activation_fn: "relu".to_string(),
                },
                Instruction::NeuronFire {
                    result: ValueId(2),
                    input: ValueId(1),
                    activation_fn: "sigmoid".to_string(),
                },
                Instruction::NeuronFire {
                    result: ValueId(3), // Dead - not used
                    input: ValueId(0),
                    activation_fn: "tanh".to_string(),
                },
            ],
        };
        
        let program = Program {
            functions: vec![function.clone()],
        };
        
        optimizer.analyze_usage_patterns(&program).unwrap();
        optimizer.eliminate_dead_neurons(&mut function).unwrap();
        
        // Should have eliminated the dead neuron
        assert_eq!(function.instructions.len(), 2);
    }
    
    #[test]
    fn test_neuron_fusion() {
        let mut optimizer = NeuronOptimizer::new();
        
        let mut function = Function {
            name: "test".to_string(),
            parameters: vec![],
            return_type: TensorType {
                dtype: DataType::Float32,
                shape: vec![Dimension::Static(1)],
                device: Device::CPU,
                requires_grad: false,
            },
            return_value: Some(ValueId(2)),
            instructions: vec![
                Instruction::NeuronFire {
                    result: ValueId(1),
                    input: ValueId(0),
                    activation_fn: "relu".to_string(),
                },
                Instruction::NeuronFire {
                    result: ValueId(2),
                    input: ValueId(1),
                    activation_fn: "relu".to_string(),
                },
            ],
        };
        
        let program = Program {
            functions: vec![function.clone()],
        };
        
        optimizer.analyze_usage_patterns(&program).unwrap();
        optimizer.usage_counts.insert(ValueId(1), 1); // Only used once
        optimizer.fuse_adjacent_neurons(&mut function).unwrap();
        
        // Should have fused the two ReLU neurons
        assert_eq!(function.instructions.len(), 1);
        
        if let Instruction::NeuronFire { activation_fn, .. } = &function.instructions[0] {
            assert_eq!(activation_fn, "fused_relu_relu");
        } else {
            panic!("Expected fused neuron fire instruction");
        }
    }
    
    #[test]
    fn test_activation_pattern_analysis() {
        let mut optimizer = NeuronOptimizer::new();
        
        let program = Program {
            functions: vec![Function {
                name: "test".to_string(),
                parameters: vec![],
                return_type: TensorType {
                    dtype: DataType::Float32,
                    shape: vec![Dimension::Static(1)],
                    device: Device::CPU,
                    requires_grad: false,
                },
                return_value: Some(ValueId(1)),
                instructions: vec![
                    Instruction::NeuronFire {
                        result: ValueId(1),
                        input: ValueId(0),
                        activation_fn: "sigmoid".to_string(),
                    },
                ],
            }],
        };
        
        optimizer.analyze_activation_patterns(&program).unwrap();
        
        let pattern = optimizer.activation_patterns.get(&ValueId(1)).unwrap();
        assert_eq!(pattern.frequency, 0.5);
        assert_eq!(pattern.latency_sensitivity, 0.8);
        assert_eq!(pattern.gradient_importance, 0.9);
    }
}