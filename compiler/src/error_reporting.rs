// 🎯 ERROR REPORTING WITH LINE NUMBERS
// Enhanced error reporting with source location tracking

use crate::source_location::{SourceLocation, SourceSpan, LocatedError};
use std::fmt;

pub struct ErrorReporter {
    source_text: String,
    errors: Vec<LocatedError>,
}

impl ErrorReporter {
    pub fn new(source_text: String) -> Self {
        ErrorReporter {
            source_text,
            errors: Vec::new(),
        }
    }
    
    pub fn report_error(&mut self, error: LocatedError) {
        self.errors.push(error);
    }
    
    pub fn report_error_at(&mut self, message: String, location: SourceLocation) {
        let mut error = LocatedError::new(message, location);
        
        // Add source line for context
        if let Some(line) = self.get_source_line(location.line) {
            error = error.with_source_line(line);
        }
        
        self.errors.push(error);
    }
    
    pub fn report_error_span(&mut self, message: String, span: SourceSpan) {
        let mut error = LocatedError::with_span(message, span);
        
        // Add source line for context
        if let Some(line) = self.get_source_line(span.start.line) {
            error = error.with_source_line(line);
        }
        
        self.errors.push(error);
    }
    
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
    
    pub fn get_errors(&self) -> &[LocatedError] {
        &self.errors
    }
    
    pub fn clear_errors(&mut self) {
        self.errors.clear();
    }
    
    fn get_source_line(&self, line_number: usize) -> Option<String> {
        self.source_text.lines().nth(line_number.saturating_sub(1)).map(|s| s.to_string())
    }
    
    pub fn format_errors(&self) -> String {
        let mut output = String::new();
        
        for error in &self.errors {
            output.push_str(&format!("{}\n", error));
        }
        
        if self.errors.len() > 1 {
            output.push_str(&format!("\n{} error(s) found\n", self.errors.len()));
        }
        
        output
    }
    
    pub fn print_errors(&self) {
        if !self.errors.is_empty() {
            eprintln!("{}", self.format_errors());
        }
    }
}

// Compilation result with enhanced error reporting
pub enum CompilationResult<T> {
    Success(T),
    Failure(Vec<LocatedError>),
}

impl<T> CompilationResult<T> {
    pub fn is_success(&self) -> bool {
        matches!(self, CompilationResult::Success(_))
    }
    
    pub fn is_failure(&self) -> bool {
        matches!(self, CompilationResult::Failure(_))
    }
    
    pub fn unwrap(self) -> T {
        match self {
            CompilationResult::Success(value) => value,
            CompilationResult::Failure(errors) => {
                panic!("Called unwrap on a Failure: {:?}", errors);
            }
        }
    }
    
    pub fn errors(&self) -> Option<&[LocatedError]> {
        match self {
            CompilationResult::Failure(errors) => Some(errors),
            _ => None,
        }
    }
}

// Helper macro for creating located errors
#[macro_export]
macro_rules! located_error {
    ($location:expr, $($arg:tt)*) => {
        LocatedError::new(format!($($arg)*), $location)
    };
}

// Helper macro for creating span errors
#[macro_export]
macro_rules! span_error {
    ($span:expr, $($arg:tt)*) => {
        LocatedError::with_span(format!($($arg)*), $span)
    };
}