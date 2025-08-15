// 📍 SOURCE LOCATION TRACKING
// Provides line and column tracking for better error messages

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceLocation {
    pub line: usize,
    pub column: usize,
    pub position: usize, // Absolute position in file
}

impl SourceLocation {
    pub fn new() -> Self {
        SourceLocation {
            line: 1,
            column: 1,
            position: 0,
        }
    }
    
    pub fn advance(&mut self, ch: char) {
        self.position += 1;
        if ch == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
    }
    
    pub fn clone_at(&self) -> Self {
        *self
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocatedToken {
    pub token: crate::minimal_lexer::Token,
    pub location: SourceLocation,
}

impl LocatedToken {
    pub fn new(token: crate::minimal_lexer::Token, location: SourceLocation) -> Self {
        LocatedToken { token, location }
    }
}

// Span for tracking ranges in source code
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceSpan {
    pub start: SourceLocation,
    pub end: SourceLocation,
}

impl SourceSpan {
    pub fn new(start: SourceLocation, end: SourceLocation) -> Self {
        SourceSpan { start, end }
    }
    
    pub fn single(location: SourceLocation) -> Self {
        SourceSpan {
            start: location,
            end: location,
        }
    }
}

impl fmt::Display for SourceSpan {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.start.line == self.end.line {
            write!(f, "{}:{}-{}", self.start.line, self.start.column, self.end.column)
        } else {
            write!(f, "{}-{}", self.start, self.end)
        }
    }
}

// Enhanced error type with location information
#[derive(Debug)]
pub struct LocatedError {
    pub message: String,
    pub location: Option<SourceLocation>,
    pub span: Option<SourceSpan>,
    pub source_line: Option<String>,
}

impl LocatedError {
    pub fn new(message: String, location: SourceLocation) -> Self {
        LocatedError {
            message,
            location: Some(location),
            span: None,
            source_line: None,
        }
    }
    
    pub fn with_span(message: String, span: SourceSpan) -> Self {
        LocatedError {
            message,
            location: Some(span.start),
            span: Some(span),
            source_line: None,
        }
    }
    
    pub fn with_source_line(mut self, line: String) -> Self {
        self.source_line = Some(line);
        self
    }
}

impl fmt::Display for LocatedError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(loc) = self.location {
            write!(f, "Error at {}: {}", loc, self.message)?;
        } else if let Some(span) = self.span {
            write!(f, "Error at {}: {}", span, self.message)?;
        } else {
            write!(f, "Error: {}", self.message)?;
        }
        
        // Show the source line if available
        if let Some(ref source_line) = self.source_line {
            writeln!(f)?;
            writeln!(f, "  |")?;
            writeln!(f, "  | {}", source_line)?;
            
            // Draw pointer to error location
            if let Some(loc) = self.location {
                write!(f, "  | ")?;
                for _ in 0..loc.column.saturating_sub(1) {
                    write!(f, " ")?;
                }
                writeln!(f, "^-- here")?;
            }
        }
        
        Ok(())
    }
}

impl std::error::Error for LocatedError {}