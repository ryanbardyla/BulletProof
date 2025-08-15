// Abstract Syntax Tree for NeuronLang
// This defines the structure of our revolutionary language

use std::fmt;

// BIOLOGICAL TYPES for NeuronLang

/// Trinary values
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TryteValue {
    Inhibited,  // -1
    Baseline,   // 0
    Activated,  // +1
}

/// Protein types in neural system
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProteinType {
    CREB,       // Transcription factor
    PKA,        // Protein Kinase A
    MAPK,       // Mitogen-activated protein kinase
    CaMKII,     // Calcium/calmodulin-dependent kinase
    BDNF,       // Brain-derived neurotrophic factor
    Arc,        // Activity-regulated cytoskeleton
    PSD95,      // Postsynaptic density protein
    Synaptophysin,
}

/// Memory formation phases
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryPhase {
    ShortTerm,      // < 1 hour
    EarlyLTP,       // 1-3 hours
    LateLTP,        // 3+ hours with protein synthesis
    Consolidated,   // Permanent
    Reconsolidating, // Memory being updated
}

/// Tensor shape dimension - can be static or dynamic
#[derive(Debug, Clone, PartialEq)]
pub enum Dimension {
    /// Static dimension with known size
    Static(usize),
    /// Dynamic dimension (represented as ? in syntax)
    Dynamic,
    /// Named dimension (e.g., 'batch', 'seq_len')
    Named(String),
}

/// Tensor type with element type and shape
#[derive(Debug, Clone, PartialEq)]
pub struct TensorType {
    pub dtype: DataType,
    pub shape: Vec<Dimension>,
    pub device: Device,
    pub requires_grad: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    F32,
    F64,
    I32,
    I64,
    Bool,
    Tryte,     // Trinary type (-1, 0, +1)
    Protein,   // Protein concentration
    Auto,      // Infer from context
}

#[derive(Debug, Clone, PartialEq)]
pub enum Device {
    CPU,
    CUDA(Option<usize>), // Optional device index
    Auto,
}

/// Type in NeuronLang
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Tensor(TensorType),
    Scalar(DataType),
    Function(Box<FunctionType>),
    Model,
    Optimizer,
    Loss,
    Void,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionType {
    pub params: Vec<Type>,
    pub return_type: Box<Type>,
    pub is_differentiable: bool,
}

/// Top-level program structure
#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<Item>,
}

/// Top-level items (functions, models, etc.)
#[derive(Debug, Clone)]
pub enum Item {
    Function(Function),
    Model(Model),
    TrainBlock(TrainBlock),
    Import(Import),
    Global(GlobalVar),
}

/// Function definition
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<Parameter>,
    pub return_type: Type,
    pub body: Block,
    pub decorators: Vec<Decorator>,
}

/// Function parameter
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub typ: Type,
    pub default: Option<Expression>,
}

/// Decorator (e.g., @differentiable, @parallel)
#[derive(Debug, Clone, PartialEq)]
pub enum Decorator {
    Differentiable,
    Parallel,
    Device(Device),
    Jit,
    Kernel,
    Distributed { nodes: usize },
    Custom(String, Vec<Expression>),
}

/// Model definition
#[derive(Debug, Clone)]
pub struct Model {
    pub name: String,
    pub layers: Vec<Layer>,
    pub forward: Option<Function>,
}

/// Neural network layer
#[derive(Debug, Clone)]
pub struct Layer {
    pub name: String,
    pub layer_type: LayerType,
}

#[derive(Debug, Clone)]
pub enum LayerType {
    Linear { in_features: usize, out_features: usize },
    Conv2d { in_channels: usize, out_channels: usize, kernel_size: usize },
    Attention { heads: usize, dim: usize },
    Custom(String, Vec<Expression>),
}

/// Training configuration block
#[derive(Debug, Clone)]
pub struct TrainBlock {
    pub model: String,
    pub dataset: String,
    pub config: TrainConfig,
}

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub optimizer: OptimizerConfig,
    pub loss: LossFunction,
    pub metrics: Vec<String>,
    pub epochs: usize,
    pub batch_size: Option<usize>,
    pub device: Device,
}

#[derive(Debug, Clone)]
pub enum OptimizerConfig {
    Adam { lr: f64, betas: (f64, f64) },
    SGD { lr: f64, momentum: Option<f64> },
    Custom(String, Vec<(String, Expression)>),
}

#[derive(Debug, Clone)]
pub enum LossFunction {
    CrossEntropy,
    MSE,
    Custom(String),
}

/// Import statement
#[derive(Debug, Clone)]
pub struct Import {
    pub module: String,
    pub items: Vec<String>,
    pub alias: Option<String>,
}

/// Global variable
#[derive(Debug, Clone)]
pub struct GlobalVar {
    pub name: String,
    pub typ: Type,
    pub value: Expression,
    pub is_mutable: bool,
}

/// Statement in a block
#[derive(Debug, Clone)]
pub enum Statement {
    Let(LetBinding),
    Assignment(Assignment),
    Expression(Expression),
    Return(Option<Expression>),
    If(IfStatement),
    For(ForLoop),
    ParallelFor(ParallelFor),
    Match(MatchStatement),
    Break,
    Continue,
}

/// Let binding
#[derive(Debug, Clone)]
pub struct LetBinding {
    pub name: String,
    pub typ: Option<Type>,
    pub value: Expression,
    pub is_mutable: bool,
}

/// Assignment
#[derive(Debug, Clone)]
pub struct Assignment {
    pub target: Expression,
    pub value: Expression,
}

/// If statement
#[derive(Debug, Clone)]
pub struct IfStatement {
    pub condition: Expression,
    pub then_branch: Block,
    pub else_branch: Option<Block>,
}

/// For loop
#[derive(Debug, Clone)]
pub struct ForLoop {
    pub var: String,
    pub iter: Expression,
    pub body: Block,
}

/// Parallel for loop
#[derive(Debug, Clone)]
pub struct ParallelFor {
    pub var: String,
    pub iter: Expression,
    pub body: Block,
    pub num_threads: Option<usize>,
}

/// Match statement
#[derive(Debug, Clone)]
pub struct MatchStatement {
    pub expr: Expression,
    pub arms: Vec<MatchArm>,
}

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expression>,
    pub body: Expression,
}

/// Pattern for pattern matching
#[derive(Debug, Clone)]
pub enum Pattern {
    Wildcard,
    Literal(Literal),
    Variable(String),
    TensorShape(Vec<Dimension>),
    Tuple(Vec<Pattern>),
}

/// Block of statements
#[derive(Debug, Clone)]
pub struct Block {
    pub statements: Vec<Statement>,
}

/// Expression
#[derive(Debug, Clone)]
pub enum Expression {
    // Literals
    Literal(Literal),
    
    // Variables
    Identifier(String),
    
    // Tensor operations
    TensorLiteral {
        data: Vec<f64>,
        shape: Vec<usize>,
    },
    
    // Binary operations
    BinaryOp {
        left: Box<Expression>,
        op: BinaryOperator,
        right: Box<Expression>,
    },
    
    // Unary operations
    UnaryOp {
        op: UnaryOperator,
        operand: Box<Expression>,
    },
    
    // Pipeline operator (|>)
    Pipeline {
        input: Box<Expression>,
        stages: Vec<Expression>,
    },
    
    // Function call
    Call {
        func: Box<Expression>,
        args: Vec<Expression>,
    },
    
    // Field access
    Field {
        object: Box<Expression>,
        field: String,
    },
    
    // Index access
    Index {
        object: Box<Expression>,
        index: Box<Expression>,
    },
    
    // BIOLOGICAL EXPRESSIONS - Revolutionary!
    
    // Tryte literal
    TryteLiteral(TryteValue),
    
    // Neuron definition
    NeuronExpr {
        id: String,
        state: TryteValue,
        threshold: f32,
        proteins: Vec<(ProteinType, f32)>,
    },
    
    // Synapse connection
    SynapseExpr {
        from: Box<Expression>,
        to: Box<Expression>,
        weight: f32,
        plasticity: TryteValue,
    },
    
    // Protein synthesis
    ProteinSynthesis {
        protein: ProteinType,
        concentration: f32,
        trigger: Box<Expression>,
    },
    
    // Spike propagation
    Spike {
        neuron: Box<Expression>,
        strength: f32,
    },
    
    // Memory consolidation
    Consolidation {
        pattern: Box<Expression>,
        phase: MemoryPhase,
    },
    
    // Sparse computation
    SparseCompute {
        network: Box<Expression>,
        skip_baseline: bool,
    },
    
    // Lambda
    Lambda {
        params: Vec<String>,
        body: Box<Expression>,
    },
    
    // Tensor comprehension
    TensorComp {
        expr: Box<Expression>,
        var: String,
        iter: Box<Expression>,
        filter: Option<Box<Expression>>,
    },
    
    // Gradient
    Gradient {
        expr: Box<Expression>,
        wrt: Box<Expression>,
    },
    
    // Block expression
    Block(Block),
}

#[derive(Debug, Clone)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    // Arithmetic
    Add, Sub, Mul, Div, Mod, Pow,
    
    // Tensor operations
    MatMul, // @
    
    // Comparison
    Eq, Ne, Lt, Le, Gt, Ge,
    
    // Logical
    And, Or,
    
    // Bitwise
    BitAnd, BitOr, BitXor, Shl, Shr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Neg, // -
    Not, // !
    BitNot, // ~
}

// Display implementations for pretty printing
impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::Tensor(t) => {
                write!(f, "tensor<{}, [", t.dtype)?;
                for (i, dim) in t.shape.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", dim)?;
                }
                write!(f, "]>")
            }
            Type::Scalar(dt) => write!(f, "{}", dt),
            Type::Function(ft) => {
                write!(f, "fn(")?;
                for (i, param) in ft.params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", param)?;
                }
                write!(f, ") -> {}", ft.return_type)
            }
            Type::Model => write!(f, "model"),
            Type::Optimizer => write!(f, "optimizer"),
            Type::Loss => write!(f, "loss"),
            Type::Void => write!(f, "void"),
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DataType::F32 => write!(f, "f32"),
            DataType::F64 => write!(f, "f64"),
            DataType::I32 => write!(f, "i32"),
            DataType::I64 => write!(f, "i64"),
            DataType::Bool => write!(f, "bool"),
            DataType::Auto => write!(f, "auto"),
        }
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Dimension::Static(n) => write!(f, "{}", n),
            Dimension::Dynamic => write!(f, "?"),
            Dimension::Named(name) => write!(f, "{}", name),
        }
    }
}