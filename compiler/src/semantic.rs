//! Semantic Analysis for NeuronLang
//! 
//! Validates biological constraints and type safety for neural operations

use crate::ast::{*, self};
use crate::error::CompilerError;
use std::collections::HashMap;

pub struct SemanticAnalyzer {
    /// Symbol table for variables
    symbols: HashMap<String, Type>,
    
    /// Protein concentrations (for validation)
    proteins: HashMap<ProteinType, f32>,
    
    /// Current scope depth
    scope_depth: usize,
    
    /// Errors collected during analysis
    errors: Vec<CompilerError>,
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        Self {
            symbols: HashMap::new(),
            proteins: HashMap::new(),
            scope_depth: 0,
            errors: Vec::new(),
        }
    }
    
    pub fn analyze(&mut self, program: Program) -> Result<Program, CompilerError> {
        // Initialize biological constants
        self.init_biological_constants();
        
        // Analyze each item in the program
        for item in &program.items {
            self.analyze_item(item)?;
        }
        
        if !self.errors.is_empty() {
            return Err(self.errors[0].clone());
        }
        
        Ok(program)
    }
    
    fn init_biological_constants(&mut self) {
        // Set baseline protein concentrations
        self.proteins.insert(ProteinType::CREB, 0.1);
        self.proteins.insert(ProteinType::PKA, 0.2);
        self.proteins.insert(ProteinType::MAPK, 0.15);
        self.proteins.insert(ProteinType::CaMKII, 0.3);
        self.proteins.insert(ProteinType::Arc, 0.05);
        self.proteins.insert(ProteinType::BDNF, 0.1);
    }
    
    fn analyze_item(&mut self, item: &Item) -> Result<(), CompilerError> {
        match item {
            Item::Function(func) => self.analyze_function(func),
            Item::Let(let_stmt) => self.analyze_let(let_stmt),
            Item::Model(model) => self.analyze_model(model),
            _ => Ok(()),
        }
    }
    
    fn analyze_function(&mut self, func: &Function) -> Result<(), CompilerError> {
        // Enter new scope
        self.scope_depth += 1;
        
        // Add parameters to symbol table
        for param in &func.params {
            self.symbols.insert(param.name.clone(), param.param_type.clone());
        }
        
        // Analyze function body
        for stmt in &func.body {
            self.analyze_statement(stmt)?;
        }
        
        // Exit scope
        self.scope_depth -= 1;
        
        Ok(())
    }
    
    fn analyze_statement(&mut self, stmt: &Statement) -> Result<(), CompilerError> {
        match stmt {
            Statement::Expression(expr) => self.analyze_expression(expr),
            Statement::Let(let_stmt) => self.analyze_let(let_stmt),
            Statement::Return(expr) => {
                if let Some(e) = expr {
                    self.analyze_expression(e)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
    
    fn analyze_let(&mut self, let_stmt: &LetStatement) -> Result<(), CompilerError> {
        // Analyze the value expression
        let value_type = self.infer_type(&let_stmt.value)?;
        
        // Check type compatibility if explicit type given
        if let Some(ref declared_type) = let_stmt.var_type {
            if !self.types_compatible(declared_type, &value_type) {
                return Err(CompilerError::TypeMismatch(
                    format!("{:?}", declared_type),
                    format!("{:?}", value_type),
                    0, // TODO: Add position tracking
                ));
            }
        }
        
        // Add to symbol table
        self.symbols.insert(let_stmt.name.clone(), value_type);
        
        Ok(())
    }
    
    fn analyze_expression(&mut self, expr: &Expression) -> Result<(), CompilerError> {
        match expr {
            Expression::TryteLiteral(tryte) => {
                // Tryte values are always valid
                Ok(())
            }
            
            Expression::NeuronExpr { state, threshold, proteins, .. } => {
                // Validate threshold
                if *threshold < 0.0 || *threshold > 2.0 {
                    return Err(CompilerError::InvalidSyntax(
                        "Neuron threshold must be between 0 and 2".to_string(),
                        0,
                    ));
                }
                
                // Validate protein concentrations
                for (protein, concentration) in proteins {
                    if *concentration < 0.0 || *concentration > 1.0 {
                        return Err(CompilerError::InvalidSyntax(
                            format!("Protein concentration must be between 0 and 1"),
                            0,
                        ));
                    }
                    
                    // Check CREB activation threshold
                    if *protein == ProteinType::CREB && *concentration > 0.7 {
                        // This triggers protein synthesis - valid but noteworthy
                        // Could add a warning system here
                    }
                }
                
                Ok(())
            }
            
            Expression::SynapseExpr { weight, plasticity, .. } => {
                // Validate weight bounds
                if weight.abs() > 2.0 {
                    return Err(CompilerError::InvalidSyntax(
                        "Synapse weight must be between -2 and 2".to_string(),
                        0,
                    ));
                }
                
                Ok(())
            }
            
            Expression::ProteinSynthesis { protein, concentration, trigger } => {
                // Validate protein synthesis trigger
                self.analyze_expression(trigger)?;
                
                // Check biological constraints
                if *protein == ProteinType::CREB && *concentration < 0.7 {
                    // Warning: CREB below activation threshold
                    // Protein synthesis won't occur
                }
                
                Ok(())
            }
            
            Expression::Consolidation { pattern, phase } => {
                self.analyze_expression(pattern)?;
                
                // Validate memory phase transitions
                match phase {
                    MemoryPhase::LateLTP | MemoryPhase::Consolidated => {
                        // These require protein synthesis
                        // Check if CREB is activated
                        if self.proteins.get(&ProteinType::CREB).unwrap_or(&0.0) < &0.7 {
                            // Warning: Late-phase LTP requires CREB activation
                        }
                    }
                    _ => {}
                }
                
                Ok(())
            }
            
            Expression::SparseCompute { network, skip_baseline } => {
                self.analyze_expression(network)?;
                
                // Sparse computation is always valid
                // It's an optimization, not a requirement
                Ok(())
            }
            
            Expression::BinaryOp { left, op, right } => {
                self.analyze_expression(left)?;
                self.analyze_expression(right)?;
                
                // Type check the operation
                let left_type = self.infer_type(left)?;
                let right_type = self.infer_type(right)?;
                
                // Special rules for Tryte operations
                if matches!(left_type, Type::Scalar(DataType::Tryte)) {
                    self.validate_tryte_operation(op)?;
                }
                
                Ok(())
            }
            
            _ => {
                // Default handling for other expressions
                Ok(())
            }
        }
    }
    
    fn infer_type(&self, expr: &Expression) -> Result<Type, CompilerError> {
        match expr {
            Expression::TryteLiteral(_) => Ok(Type::Scalar(DataType::Tryte)),
            
            Expression::Literal(lit) => match lit {
                Literal::Float(_) => Ok(Type::Scalar(DataType::F32)),
                Literal::Integer(_) => Ok(Type::Scalar(DataType::I32)),
                Literal::Boolean(_) => Ok(Type::Scalar(DataType::Bool)),
                Literal::String(_) => Ok(Type::String),
            },
            
            Expression::NeuronExpr { .. } => Ok(Type::Neuron),
            
            Expression::SynapseExpr { .. } => Ok(Type::Synapse),
            
            Expression::ProteinSynthesis { .. } => Ok(Type::Scalar(DataType::Protein)),
            
            Expression::Identifier(name) => {
                self.symbols.get(name)
                    .cloned()
                    .ok_or_else(|| CompilerError::UndefinedVariable(name.clone(), 0))
            }
            
            _ => Ok(Type::Void), // TODO: Implement for other expression types
        }
    }
    
    fn types_compatible(&self, t1: &Type, t2: &Type) -> bool {
        match (t1, t2) {
            (Type::Scalar(d1), Type::Scalar(d2)) => {
                match (d1, d2) {
                    (DataType::Auto, _) | (_, DataType::Auto) => true,
                    (DataType::Tryte, DataType::Tryte) => true,
                    (DataType::Protein, DataType::Protein) => true,
                    (DataType::F32, DataType::F64) | (DataType::F64, DataType::F32) => true,
                    (DataType::I32, DataType::I64) | (DataType::I64, DataType::I32) => true,
                    _ => d1 == d2,
                }
            }
            (Type::Neuron, Type::Neuron) => true,
            (Type::Synapse, Type::Synapse) => true,
            _ => false,
        }
    }
    
    fn validate_tryte_operation(&self, op: &BinaryOperator) -> Result<(), CompilerError> {
        match op {
            // Valid Tryte operations
            BinaryOperator::Add | 
            BinaryOperator::Multiply |
            BinaryOperator::Equal |
            BinaryOperator::NotEqual => Ok(()),
            
            // Invalid operations on Trytes
            BinaryOperator::Divide => {
                Err(CompilerError::InvalidTryteOperation(
                    "Division not defined for Trytes".to_string(),
                    0,
                ))
            }
            
            _ => Ok(()),
        }
    }
    
    fn analyze_model(&mut self, model: &Model) -> Result<(), CompilerError> {
        // Analyze each layer
        for layer in &model.layers {
            self.analyze_layer(layer)?;
        }
        
        // Analyze forward function if present
        if let Some(ref forward) = model.forward {
            self.analyze_function(forward)?;
        }
        
        Ok(())
    }
    
    fn analyze_layer(&mut self, layer: &Layer) -> Result<(), CompilerError> {
        match &layer.layer_type {
            LayerType::Linear { in_features, out_features } => {
                if *in_features == 0 || *out_features == 0 {
                    return Err(CompilerError::InvalidSyntax(
                        "Layer dimensions must be positive".to_string(),
                        0,
                    ));
                }
            }
            _ => {}
        }
        Ok(())
    }
}

// Add biological types to the Type enum
impl Type {
    pub const NEURON: Type = Type::Neuron;
    pub const SYNAPSE: Type = Type::Synapse;
}

// Extend the Type enum in ast.rs
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Tensor(TensorType),
    Scalar(DataType),
    Function(Box<FunctionType>),
    Model,
    Optimizer,
    String,
    Void,
    
    // Biological types
    Neuron,
    Synapse,
    Network,
}