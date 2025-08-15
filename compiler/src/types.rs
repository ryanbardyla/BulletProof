// NeuronLang Type System - Revolutionary compile-time tensor shape checking
// No language has this level of tensor type safety!

use crate::ast::{Type, TensorType, DataType, Dimension, Device};
use std::collections::HashMap;
use std::fmt;

/// Type environment for tracking variable types
#[derive(Debug, Clone)]
pub struct TypeEnv {
    /// Stack of scopes (for nested blocks)
    scopes: Vec<HashMap<String, Type>>,
    /// Named dimension constraints
    dimension_constraints: HashMap<String, DimensionConstraint>,
}

/// Constraint on a named dimension
#[derive(Debug, Clone)]
pub enum DimensionConstraint {
    /// Exact size
    Exact(usize),
    /// Range of valid sizes
    Range(usize, usize),
    /// Must be divisible by N
    Divisible(usize),
    /// Must match another dimension
    SameAs(String),
    /// Any size (unconstrained)
    Any,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            dimension_constraints: HashMap::new(),
        }
    }
    
    /// Enter a new scope
    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }
    
    /// Exit current scope
    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }
    
    /// Add a variable to current scope
    pub fn insert(&mut self, name: String, typ: Type) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, typ);
        }
    }
    
    /// Look up a variable (searches all scopes)
    pub fn get(&self, name: &str) -> Option<&Type> {
        for scope in self.scopes.iter().rev() {
            if let Some(typ) = scope.get(name) {
                return Some(typ);
            }
        }
        None
    }
    
    /// Add a dimension constraint
    pub fn add_constraint(&mut self, dim: String, constraint: DimensionConstraint) {
        self.dimension_constraints.insert(dim, constraint);
    }
}

/// The core type checker
pub struct TypeChecker {
    env: TypeEnv,
    errors: Vec<TypeError>,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            env: TypeEnv::new(),
            errors: Vec::new(),
        }
    }
    
    /// Unify two types (make them compatible)
    pub fn unify(&mut self, t1: &Type, t2: &Type) -> Result<Type, TypeError> {
        match (t1, t2) {
            // Identical types
            (Type::Scalar(d1), Type::Scalar(d2)) if d1 == d2 => Ok(t1.clone()),
            
            // Tensor unification - the magic happens here!
            (Type::Tensor(tensor1), Type::Tensor(tensor2)) => {
                self.unify_tensors(tensor1, tensor2)
            }
            
            // Auto type inference
            (Type::Scalar(DataType::Auto), other) | (other, Type::Scalar(DataType::Auto)) => {
                Ok(other.clone())
            }
            
            // Function types
            (Type::Function(f1), Type::Function(f2)) => {
                self.unify_functions(f1, f2)
            }
            
            // Incompatible types
            _ => Err(TypeError::TypeMismatch {
                expected: t1.clone(),
                found: t2.clone(),
            }),
        }
    }
    
    /// Unify two tensor types - this is where the magic happens!
    fn unify_tensors(&mut self, t1: &TensorType, t2: &TensorType) -> Result<Type, TypeError> {
        // Check data types
        let dtype = self.unify_dtypes(t1.dtype.clone(), t2.dtype.clone())?;
        
        // Check shapes - this is the revolutionary part
        let shape = self.unify_shapes(&t1.shape, &t2.shape)?;
        
        // Unify devices
        let device = self.unify_devices(t1.device.clone(), t2.device.clone())?;
        
        // Gradient requirement
        let requires_grad = t1.requires_grad || t2.requires_grad;
        
        Ok(Type::Tensor(TensorType {
            dtype,
            shape,
            device,
            requires_grad,
        }))
    }
    
    /// Unify tensor shapes with broadcasting rules
    fn unify_shapes(&mut self, shape1: &[Dimension], shape2: &[Dimension]) -> Result<Vec<Dimension>, TypeError> {
        // Handle broadcasting - dimensions align from the right
        let len1 = shape1.len();
        let len2 = shape2.len();
        let max_len = len1.max(len2);
        
        let mut unified_shape = Vec::with_capacity(max_len);
        
        // Pad shorter shape with 1s on the left (broadcasting rule)
        let padded1 = self.pad_shape(shape1, max_len);
        let padded2 = self.pad_shape(shape2, max_len);
        
        // Unify each dimension
        for (d1, d2) in padded1.iter().zip(padded2.iter()) {
            unified_shape.push(self.unify_dimensions(d1, d2)?);
        }
        
        Ok(unified_shape)
    }
    
    /// Unify two dimensions
    fn unify_dimensions(&mut self, d1: &Dimension, d2: &Dimension) -> Result<Dimension, TypeError> {
        match (d1, d2) {
            // Both static and equal
            (Dimension::Static(n1), Dimension::Static(n2)) if n1 == n2 => {
                Ok(Dimension::Static(*n1))
            }
            
            // Broadcasting: 1 broadcasts to any size
            (Dimension::Static(1), other) | (other, Dimension::Static(1)) => {
                Ok(other.clone())
            }
            
            // Dynamic dimension unifies with anything
            (Dimension::Dynamic, other) | (other, Dimension::Dynamic) => {
                Ok(other.clone())
            }
            
            // Named dimensions
            (Dimension::Named(name1), Dimension::Named(name2)) if name1 == name2 => {
                Ok(Dimension::Named(name1.clone()))
            }
            
            // Named dimension with constraint
            (Dimension::Named(name), Dimension::Static(size)) |
            (Dimension::Static(size), Dimension::Named(name)) => {
                // Add constraint that this named dimension must be this size
                self.env.add_constraint(name.clone(), DimensionConstraint::Exact(*size));
                Ok(Dimension::Static(*size))
            }
            
            // Incompatible dimensions
            _ => Err(TypeError::ShapeMismatch {
                dim1: d1.clone(),
                dim2: d2.clone(),
            }),
        }
    }
    
    /// Pad shape with 1s for broadcasting
    fn pad_shape(&self, shape: &[Dimension], target_len: usize) -> Vec<Dimension> {
        let mut padded = vec![Dimension::Static(1); target_len - shape.len()];
        padded.extend_from_slice(shape);
        padded
    }
    
    /// Unify data types
    fn unify_dtypes(&self, d1: DataType, d2: DataType) -> Result<DataType, TypeError> {
        match (d1, d2) {
            (DataType::Auto, other) | (other, DataType::Auto) => Ok(other),
            (d1, d2) if d1 == d2 => Ok(d1),
            _ => Err(TypeError::DataTypeMismatch { d1, d2 }),
        }
    }
    
    /// Unify devices
    fn unify_devices(&self, dev1: Device, dev2: Device) -> Result<Device, TypeError> {
        match (dev1, dev2) {
            (Device::Auto, other) | (other, Device::Auto) => Ok(other),
            (d1, d2) if d1 == d2 => Ok(d1),
            _ => Err(TypeError::DeviceMismatch { dev1, dev2 }),
        }
    }
    
    /// Unify function types
    fn unify_functions(&mut self, f1: &crate::ast::FunctionType, f2: &crate::ast::FunctionType) 
        -> Result<Type, TypeError> {
        // Check parameter count
        if f1.params.len() != f2.params.len() {
            return Err(TypeError::ArityMismatch {
                expected: f1.params.len(),
                found: f2.params.len(),
            });
        }
        
        // Unify parameter types
        let mut unified_params = Vec::new();
        for (p1, p2) in f1.params.iter().zip(f2.params.iter()) {
            unified_params.push(self.unify(p1, p2)?);
        }
        
        // Unify return types
        let unified_return = self.unify(&f1.return_type, &f2.return_type)?;
        
        Ok(Type::Function(Box::new(crate::ast::FunctionType {
            params: unified_params,
            return_type: Box::new(unified_return),
            is_differentiable: f1.is_differentiable && f2.is_differentiable,
        })))
    }
    
    /// Infer type of a binary operation
    pub fn infer_binop(&mut self, op: &crate::ast::BinaryOperator, 
                       left: &Type, right: &Type) -> Result<Type, TypeError> {
        use crate::ast::BinaryOperator::*;
        
        match op {
            // Arithmetic operations
            Add | Sub | Mul | Div => {
                // These work element-wise on tensors with broadcasting
                self.unify(left, right)
            }
            
            // Matrix multiplication - special shape rules!
            MatMul => {
                match (left, right) {
                    (Type::Tensor(t1), Type::Tensor(t2)) => {
                        self.check_matmul_shapes(&t1.shape, &t2.shape)?;
                        
                        // Result shape: [..., m, k] @ [..., k, n] = [..., m, n]
                        let mut result_shape = Vec::new();
                        
                        // Broadcast batch dimensions
                        let batch_dims = t1.shape.len().max(t2.shape.len()) - 2;
                        for i in 0..batch_dims {
                            let d1 = t1.shape.get(i).cloned().unwrap_or(Dimension::Static(1));
                            let d2 = t2.shape.get(i).cloned().unwrap_or(Dimension::Static(1));
                            result_shape.push(self.unify_dimensions(&d1, &d2)?);
                        }
                        
                        // Add output dimensions [m, n]
                        if let Some(m) = t1.shape.iter().rev().nth(1) {
                            result_shape.push(m.clone());
                        }
                        if let Some(n) = t2.shape.last() {
                            result_shape.push(n.clone());
                        }
                        
                        Ok(Type::Tensor(TensorType {
                            dtype: self.unify_dtypes(t1.dtype.clone(), t2.dtype.clone())?,
                            shape: result_shape,
                            device: self.unify_devices(t1.device.clone(), t2.device.clone())?,
                            requires_grad: t1.requires_grad || t2.requires_grad,
                        }))
                    }
                    _ => Err(TypeError::InvalidOperation {
                        op: "matmul".to_string(),
                        typ: left.clone(),
                    }),
                }
            }
            
            // Comparison operations return bool
            Eq | Ne | Lt | Le | Gt | Ge => {
                self.unify(left, right)?;
                Ok(Type::Scalar(DataType::Bool))
            }
            
            // Logical operations
            And | Or => {
                self.unify(left, &Type::Scalar(DataType::Bool))?;
                self.unify(right, &Type::Scalar(DataType::Bool))?;
                Ok(Type::Scalar(DataType::Bool))
            }
            
            _ => Err(TypeError::InvalidOperation {
                op: format!("{:?}", op),
                typ: left.clone(),
            }),
        }
    }
    
    /// Check if shapes are compatible for matrix multiplication
    fn check_matmul_shapes(&self, shape1: &[Dimension], shape2: &[Dimension]) -> Result<(), TypeError> {
        // Need at least 2D tensors
        if shape1.len() < 2 || shape2.len() < 2 {
            return Err(TypeError::InvalidMatMul {
                shape1: shape1.to_vec(),
                shape2: shape2.to_vec(),
            });
        }
        
        // Check inner dimensions match: [..., m, k] @ [..., k, n]
        let k1 = &shape1[shape1.len() - 1];
        let k2 = &shape2[shape2.len() - 2];
        
        match (k1, k2) {
            (Dimension::Static(n1), Dimension::Static(n2)) if n1 != n2 => {
                return Err(TypeError::InvalidMatMul {
                    shape1: shape1.to_vec(),
                    shape2: shape2.to_vec(),
                });
            }
            _ => {} // Dynamic or matching dimensions are OK
        }
        
        Ok(())
    }
    
    /// Infer type of a function call
    pub fn infer_call(&mut self, func_type: &Type, args: &[Type]) -> Result<Type, TypeError> {
        match func_type {
            Type::Function(f) => {
                // Check arity
                if f.params.len() != args.len() {
                    return Err(TypeError::ArityMismatch {
                        expected: f.params.len(),
                        found: args.len(),
                    });
                }
                
                // Check each argument
                for (param, arg) in f.params.iter().zip(args.iter()) {
                    self.unify(param, arg)?;
                }
                
                Ok(*f.return_type.clone())
            }
            _ => Err(TypeError::NotCallable {
                typ: func_type.clone(),
            }),
        }
    }
}

/// Type errors
#[derive(Debug, Clone)]
pub enum TypeError {
    TypeMismatch {
        expected: Type,
        found: Type,
    },
    ShapeMismatch {
        dim1: Dimension,
        dim2: Dimension,
    },
    DataTypeMismatch {
        d1: DataType,
        d2: DataType,
    },
    DeviceMismatch {
        dev1: Device,
        dev2: Device,
    },
    InvalidMatMul {
        shape1: Vec<Dimension>,
        shape2: Vec<Dimension>,
    },
    ArityMismatch {
        expected: usize,
        found: usize,
    },
    NotCallable {
        typ: Type,
    },
    InvalidOperation {
        op: String,
        typ: Type,
    },
    UnboundVariable {
        name: String,
    },
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TypeError::TypeMismatch { expected, found } => {
                write!(f, "Type mismatch: expected {}, found {}", expected, found)
            }
            TypeError::ShapeMismatch { dim1, dim2 } => {
                write!(f, "Shape mismatch: dimension {} is incompatible with {}", dim1, dim2)
            }
            TypeError::DataTypeMismatch { d1, d2 } => {
                write!(f, "Data type mismatch: {} vs {}", d1, d2)
            }
            TypeError::DeviceMismatch { dev1, dev2 } => {
                write!(f, "Device mismatch: {:?} vs {:?}", dev1, dev2)
            }
            TypeError::InvalidMatMul { shape1, shape2 } => {
                write!(f, "Invalid matrix multiplication: {:?} @ {:?}", shape1, shape2)
            }
            TypeError::ArityMismatch { expected, found } => {
                write!(f, "Wrong number of arguments: expected {}, found {}", expected, found)
            }
            TypeError::NotCallable { typ } => {
                write!(f, "Type {} is not callable", typ)
            }
            TypeError::InvalidOperation { op, typ } => {
                write!(f, "Invalid operation {} for type {}", op, typ)
            }
            TypeError::UnboundVariable { name } => {
                write!(f, "Unbound variable: {}", name)
            }
        }
    }
}

impl std::error::Error for TypeError {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shape_unification() {
        let mut checker = TypeChecker::new();
        
        // Test broadcasting: [32, 784] and [784, 10] should fail for element-wise ops
        let shape1 = vec![Dimension::Static(32), Dimension::Static(784)];
        let shape2 = vec![Dimension::Static(784), Dimension::Static(10)];
        
        assert!(checker.unify_shapes(&shape1, &shape2).is_err());
        
        // Test broadcasting: [32, 1] and [32, 10] should work (broadcasts to [32, 10])
        let shape1 = vec![Dimension::Static(32), Dimension::Static(1)];
        let shape2 = vec![Dimension::Static(32), Dimension::Static(10)];
        
        let result = checker.unify_shapes(&shape1, &shape2).unwrap();
        assert_eq!(result, vec![Dimension::Static(32), Dimension::Static(10)]);
        
        // Test dynamic dimensions: [?, 784] unifies with [32, 784]
        let shape1 = vec![Dimension::Dynamic, Dimension::Static(784)];
        let shape2 = vec![Dimension::Static(32), Dimension::Static(784)];
        
        let result = checker.unify_shapes(&shape1, &shape2).unwrap();
        assert_eq!(result, vec![Dimension::Static(32), Dimension::Static(784)]);
    }
    
    #[test]
    fn test_matmul_shapes() {
        let mut checker = TypeChecker::new();
        
        // Valid matmul: [32, 784] @ [784, 10] = [32, 10]
        let t1 = Type::Tensor(TensorType {
            dtype: DataType::F32,
            shape: vec![Dimension::Static(32), Dimension::Static(784)],
            device: Device::Auto,
            requires_grad: false,
        });
        
        let t2 = Type::Tensor(TensorType {
            dtype: DataType::F32,
            shape: vec![Dimension::Static(784), Dimension::Static(10)],
            device: Device::Auto,
            requires_grad: false,
        });
        
        let result = checker.infer_binop(&crate::ast::BinaryOperator::MatMul, &t1, &t2).unwrap();
        
        if let Type::Tensor(t) = result {
            assert_eq!(t.shape, vec![Dimension::Static(32), Dimension::Static(10)]);
        } else {
            panic!("Expected tensor type");
        }
    }
}