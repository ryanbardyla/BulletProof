// NeuronIR - Revolutionary Intermediate Representation for AI
// Preserves tensor shapes, automatic differentiation, and parallelism

use crate::ast;
use crate::types::Type;
use std::collections::HashMap;
use std::fmt;

/// NeuronIR Module - compilation unit
#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub functions: Vec<Function>,
    pub globals: Vec<Global>,
    pub metadata: Metadata,
}

/// Function in IR
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<Value>,
    pub return_type: Type,
    pub blocks: Vec<BasicBlock>,
    pub is_differentiable: bool,
    pub is_kernel: bool, // GPU kernel function
}

/// Basic block - sequence of instructions with single entry/exit
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub label: String,
    pub instructions: Vec<Instruction>,
    pub terminator: Terminator,
}

/// IR Value - can be tensor, scalar, or reference
#[derive(Debug, Clone)]
pub struct Value {
    pub id: ValueId,
    pub typ: Type,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

/// IR Instructions - tensor-aware operations
#[derive(Debug, Clone)]
pub enum Instruction {
    // Tensor creation
    TensorLiteral {
        result: Value,
        data: Vec<f64>,
        shape: Vec<usize>,
    },
    
    TensorZeros {
        result: Value,
        shape: Vec<usize>,
        dtype: ast::DataType,
    },
    
    TensorOnes {
        result: Value,
        shape: Vec<usize>,
        dtype: ast::DataType,
    },
    
    TensorRandom {
        result: Value,
        shape: Vec<usize>,
        dtype: ast::DataType,
        distribution: Distribution,
    },
    
    // Tensor operations - preserve shapes!
    TensorAdd {
        result: Value,
        left: Value,
        right: Value,
        broadcast: bool,
    },
    
    TensorSub {
        result: Value,
        left: Value,
        right: Value,
        broadcast: bool,
    },
    
    TensorMul {
        result: Value,
        left: Value,
        right: Value,
        broadcast: bool,
    },
    
    TensorDiv {
        result: Value,
        left: Value,
        right: Value,
        broadcast: bool,
    },
    
    // Matrix operations
    MatMul {
        result: Value,
        left: Value,
        right: Value,
        transpose_left: bool,
        transpose_right: bool,
    },
    
    // Shape operations
    Reshape {
        result: Value,
        input: Value,
        new_shape: Vec<ShapeDim>,
    },
    
    Transpose {
        result: Value,
        input: Value,
        dims: Vec<usize>,
    },
    
    Broadcast {
        result: Value,
        input: Value,
        target_shape: Vec<ShapeDim>,
    },
    
    // Reduction operations
    ReduceSum {
        result: Value,
        input: Value,
        axes: Vec<usize>,
        keepdims: bool,
    },
    
    ReduceMean {
        result: Value,
        input: Value,
        axes: Vec<usize>,
        keepdims: bool,
    },
    
    ReduceMax {
        result: Value,
        input: Value,
        axes: Vec<usize>,
        keepdims: bool,
    },
    
    // Neural network operations
    Linear {
        result: Value,
        input: Value,
        weight: Value,
        bias: Option<Value>,
    },
    
    Conv2d {
        result: Value,
        input: Value,
        kernel: Value,
        stride: (usize, usize),
        padding: (usize, usize),
    },
    
    BatchNorm {
        result: Value,
        input: Value,
        mean: Value,
        variance: Value,
        gamma: Value,
        beta: Value,
        epsilon: f64,
    },
    
    // Activation functions
    Activation {
        result: Value,
        input: Value,
        function: ActivationFunction,
    },
    
    // Dropout
    Dropout {
        result: Value,
        input: Value,
        rate: f64,
        training: bool,
    },
    
    // Gradient operations - AUTOMATIC DIFFERENTIATION!
    GradientOf {
        result: Value,
        output: Value,
        wrt: Value, // with respect to
    },
    
    AccumulateGrad {
        result: Value,
        grad: Value,
        accumulated: Value,
    },
    
    // Memory operations
    Load {
        result: Value,
        address: Value,
    },
    
    Store {
        value: Value,
        address: Value,
    },
    
    Alloc {
        result: Value,
        size: usize,
        alignment: usize,
    },
    
    // Control flow helpers
    Phi {
        result: Value,
        incoming: Vec<(Value, String)>, // (value, from_block)
    },
    
    // Parallel operations
    ParallelMap {
        result: Value,
        input: Value,
        function: String,
        num_threads: Option<usize>,
    },
    
    ParallelReduce {
        result: Value,
        input: Value,
        function: String,
        init: Value,
        num_threads: Option<usize>,
    },
    
    // Device operations
    ToDevice {
        result: Value,
        input: Value,
        device: ast::Device,
    },
    
    // Kernel launch (GPU)
    LaunchKernel {
        kernel: String,
        grid_dim: (usize, usize, usize),
        block_dim: (usize, usize, usize),
        args: Vec<Value>,
        shared_mem: usize,
    },
    
    // Function call
    Call {
        result: Option<Value>,
        function: String,
        args: Vec<Value>,
    },
}

/// Shape dimension - can be static or dynamic
#[derive(Debug, Clone)]
pub enum ShapeDim {
    Static(usize),
    Dynamic(String), // Named dynamic dimension
    Inferred,        // To be inferred
}

/// Distribution for random tensors
#[derive(Debug, Clone)]
pub enum Distribution {
    Uniform { low: f64, high: f64 },
    Normal { mean: f64, std: f64 },
    Xavier,
    Kaiming,
}

/// Activation functions
#[derive(Debug, Clone)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    SiLU,
    Softmax { axis: usize },
    LogSoftmax { axis: usize },
    LeakyReLU { negative_slope: f64 },
}

/// Block terminator
#[derive(Debug, Clone)]
pub enum Terminator {
    Return {
        value: Option<Value>,
    },
    
    Branch {
        target: String,
    },
    
    ConditionalBranch {
        condition: Value,
        true_target: String,
        false_target: String,
    },
    
    Switch {
        value: Value,
        default: String,
        cases: Vec<(i64, String)>,
    },
    
    Unreachable,
}

/// Global variable
#[derive(Debug, Clone)]
pub struct Global {
    pub name: String,
    pub typ: Type,
    pub initializer: Option<Value>,
    pub is_mutable: bool,
}

/// Module metadata
#[derive(Debug, Clone)]
pub struct Metadata {
    pub target_device: ast::Device,
    pub optimization_level: u8,
    pub enable_autodiff: bool,
    pub enable_fusion: bool,
}

/// IR Builder - constructs IR from AST
pub struct IRBuilder {
    current_function: Option<Function>,
    current_block: Option<BasicBlock>,
    value_counter: u32,
    functions: Vec<Function>,
    globals: Vec<Global>,
}

impl IRBuilder {
    pub fn new() -> Self {
        Self {
            current_function: None,
            current_block: None,
            value_counter: 0,
            functions: Vec::new(),
            globals: Vec::new(),
        }
    }
    
    /// Create a new value
    pub fn new_value(&mut self, typ: Type, name: Option<String>) -> Value {
        let id = ValueId(self.value_counter);
        self.value_counter += 1;
        Value { id, typ, name }
    }
    
    /// Add instruction to current block
    pub fn add_instruction(&mut self, inst: Instruction) {
        if let Some(block) = &mut self.current_block {
            block.instructions.push(inst);
        }
    }
    
    /// Create tensor addition with broadcasting
    pub fn build_tensor_add(&mut self, left: Value, right: Value) -> Value {
        let result_type = self.infer_broadcast_type(&left.typ, &right.typ);
        let result = self.new_value(result_type, None);
        
        self.add_instruction(Instruction::TensorAdd {
            result: result.clone(),
            left,
            right,
            broadcast: true,
        });
        
        result
    }
    
    /// Create matrix multiplication
    pub fn build_matmul(&mut self, left: Value, right: Value) -> Value {
        let result_type = self.infer_matmul_type(&left.typ, &right.typ);
        let result = self.new_value(result_type, None);
        
        self.add_instruction(Instruction::MatMul {
            result: result.clone(),
            left,
            right,
            transpose_left: false,
            transpose_right: false,
        });
        
        result
    }
    
    /// Create activation function
    pub fn build_activation(&mut self, input: Value, func: ActivationFunction) -> Value {
        let result = self.new_value(input.typ.clone(), None);
        
        self.add_instruction(Instruction::Activation {
            result: result.clone(),
            input,
            function: func,
        });
        
        result
    }
    
    /// Create gradient computation
    pub fn build_gradient(&mut self, output: Value, wrt: Value) -> Value {
        let result = self.new_value(wrt.typ.clone(), Some("grad".to_string()));
        
        self.add_instruction(Instruction::GradientOf {
            result: result.clone(),
            output,
            wrt,
        });
        
        result
    }
    
    /// Infer type for broadcasting
    fn infer_broadcast_type(&self, t1: &Type, t2: &Type) -> Type {
        // Simplified - real implementation would use full type system
        t1.clone()
    }
    
    /// Infer type for matrix multiplication
    fn infer_matmul_type(&self, t1: &Type, t2: &Type) -> Type {
        // Simplified - real implementation would compute output shape
        t1.clone()
    }
    
    /// Build a complete module
    pub fn build_module(self, name: String) -> Module {
        Module {
            name,
            functions: self.functions,
            globals: self.globals,
            metadata: Metadata {
                target_device: ast::Device::Auto,
                optimization_level: 2,
                enable_autodiff: true,
                enable_fusion: true,
            },
        }
    }
}

/// IR Printer - human-readable format
impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "; NeuronIR Module: {}", self.name)?;
        writeln!(f, "; Target: {:?}", self.metadata.target_device)?;
        writeln!(f)?;
        
        // Print globals
        for global in &self.globals {
            writeln!(f, "@{} = global {} {}", 
                global.name, 
                if global.is_mutable { "mut" } else { "" },
                global.typ)?;
        }
        
        if !self.globals.is_empty() {
            writeln!(f)?;
        }
        
        // Print functions
        for func in &self.functions {
            write!(f, "{}", func)?;
            writeln!(f)?;
        }
        
        Ok(())
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "define ")?;
        if self.is_differentiable {
            write!(f, "@differentiable ")?;
        }
        if self.is_kernel {
            write!(f, "@kernel ")?;
        }
        
        write!(f, "{} @{}(", self.return_type, self.name)?;
        
        for (i, param) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{} %{}", param.typ, param.id.0)?;
            if let Some(name) = &param.name {
                write!(f, " ; {}", name)?;
            }
        }
        
        writeln!(f, ") {{")?;
        
        for block in &self.blocks {
            write!(f, "{}", block)?;
        }
        
        writeln!(f, "}}")
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{}:", self.label)?;
        
        for inst in &self.instructions {
            writeln!(f, "  {}", inst)?;
        }
        
        writeln!(f, "  {}", self.terminator)
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Instruction::TensorAdd { result, left, right, broadcast } => {
                write!(f, "%{} = tensor.add %{}, %{}", 
                    result.id.0, left.id.0, right.id.0)?;
                if *broadcast {
                    write!(f, " ; broadcast")?;
                }
                Ok(())
            }
            Instruction::MatMul { result, left, right, .. } => {
                write!(f, "%{} = tensor.matmul %{}, %{}", 
                    result.id.0, left.id.0, right.id.0)
            }
            Instruction::Activation { result, input, function } => {
                write!(f, "%{} = tensor.{:?} %{}", 
                    result.id.0, function, input.id.0)
            }
            Instruction::GradientOf { result, output, wrt } => {
                write!(f, "%{} = gradient %{} wrt %{}", 
                    result.id.0, output.id.0, wrt.id.0)
            }
            Instruction::Call { result, function, args } => {
                if let Some(r) = result {
                    write!(f, "%{} = ", r.id.0)?;
                }
                write!(f, "call @{}(", function)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "%{}", arg.id.0)?;
                }
                write!(f, ")")
            }
            _ => write!(f, "; {:?}", self), // Placeholder for other instructions
        }
    }
}

impl fmt::Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Terminator::Return { value } => {
                write!(f, "ret")?;
                if let Some(v) = value {
                    write!(f, " %{}", v.id.0)?;
                }
                Ok(())
            }
            Terminator::Branch { target } => {
                write!(f, "br {}", target)
            }
            Terminator::ConditionalBranch { condition, true_target, false_target } => {
                write!(f, "br %{}, {}, {}", condition.id.0, true_target, false_target)
            }
            _ => write!(f, "; {:?}", self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ir_builder() {
        let mut builder = IRBuilder::new();
        
        // Create some tensor values
        let input = builder.new_value(
            Type::Tensor(crate::ast::TensorType {
                dtype: ast::DataType::F32,
                shape: vec![ast::Dimension::Static(32), ast::Dimension::Static(784)],
                device: ast::Device::Auto,
                requires_grad: true,
            }),
            Some("input".to_string())
        );
        
        let weight = builder.new_value(
            Type::Tensor(crate::ast::TensorType {
                dtype: ast::DataType::F32,
                shape: vec![ast::Dimension::Static(784), ast::Dimension::Static(256)],
                device: ast::Device::Auto,
                requires_grad: true,
            }),
            Some("weight".to_string())
        );
        
        // Build operations
        let hidden = builder.build_matmul(input.clone(), weight.clone());
        let activated = builder.build_activation(hidden, ActivationFunction::ReLU);
        let grad = builder.build_gradient(activated, input);
        
        // Check we created the right instructions
        assert_eq!(builder.value_counter, 6); // input, weight, hidden, activated, grad
    }
}