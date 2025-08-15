//! Compiler error handling

use std::fmt;

#[derive(Debug, Clone)]
pub enum CompilerError {
    // Lexer errors
    UnexpectedCharacter(char, usize),
    UnterminatedString(usize),
    InvalidTryteValue(String, usize),
    InvalidProteinType(String, usize),
    
    // Parser errors
    UnexpectedToken(String, usize),
    MissingToken(String, usize),
    InvalidSyntax(String, usize),
    
    // Semantic errors
    UndefinedVariable(String, usize),
    TypeMismatch(String, String, usize),
    InvalidTryteOperation(String, usize),
    ProteinNotDeclared(String, usize),
    
    // Code generation errors
    NotImplemented(String),
    InternalError(String),
}

impl fmt::Display for CompilerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CompilerError::UnexpectedCharacter(ch, pos) => {
                write!(f, "Unexpected character '{}' at position {}", ch, pos)
            }
            CompilerError::UnterminatedString(pos) => {
                write!(f, "Unterminated string starting at position {}", pos)
            }
            CompilerError::InvalidTryteValue(val, pos) => {
                write!(f, "Invalid Tryte value '{}' at position {}. Must be -1, 0, or +1", val, pos)
            }
            CompilerError::InvalidProteinType(name, pos) => {
                write!(f, "Unknown protein type '{}' at position {}", name, pos)
            }
            CompilerError::UnexpectedToken(token, pos) => {
                write!(f, "Unexpected token '{}' at position {}", token, pos)
            }
            CompilerError::MissingToken(expected, pos) => {
                write!(f, "Expected '{}' at position {}", expected, pos)
            }
            CompilerError::InvalidSyntax(msg, pos) => {
                write!(f, "Invalid syntax at position {}: {}", pos, msg)
            }
            CompilerError::UndefinedVariable(name, pos) => {
                write!(f, "Undefined variable '{}' at position {}", name, pos)
            }
            CompilerError::TypeMismatch(expected, found, pos) => {
                write!(f, "Type mismatch at position {}: expected {}, found {}", pos, expected, found)
            }
            CompilerError::InvalidTryteOperation(op, pos) => {
                write!(f, "Invalid operation '{}' on Trytes at position {}", op, pos)
            }
            CompilerError::ProteinNotDeclared(name, pos) => {
                write!(f, "Protein '{}' used but not declared at position {}", name, pos)
            }
            CompilerError::NotImplemented(feature) => {
                write!(f, "Feature not yet implemented: {}", feature)
            }
            CompilerError::InternalError(msg) => {
                write!(f, "Internal compiler error: {}", msg)
            }
        }
    }
}

impl std::error::Error for CompilerError {}