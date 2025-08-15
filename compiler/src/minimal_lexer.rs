// 🧬 MINIMAL NEURONLANG LEXER
// The smallest possible lexer that can tokenize NeuronLang

use std::str::Chars;
use std::iter::Peekable;
use crate::source_location::{SourceLocation, LocatedToken};

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords (biological)
    Organism,
    Cell,
    Neuron,
    Protein,
    Gene,
    Layer,      // Neural network layer
    Evolve,
    Mutate,
    Express,
    Synthesize,
    
    // Neural Network operations
    Network,
    Forward,
    Backward,
    Activation,
    Weights,
    Bias,
    Gradient,
    Learn,
    Train,
    
    // Redis operations (REAL!)
    RedisConnect,
    RedisGet,
    RedisSet,
    RedisPublish,
    RedisSubscribe,
    
    // Keywords (computational)
    Function,
    Struct,
    Enum,
    Let,
    If,
    Else,
    Loop,
    While,
    For,
    In,
    Return,
    Break,
    Continue,
    Breakpoint,  // Debug breakpoint
    Match,
    Lambda,      // lambda keyword for closures
    Import,      // import keyword for modules
    Export,      // export keyword for exposing items
    Module,      // module keyword for module definitions
    
    // 🗑️ Garbage collection keywords  
    GcAlloc,     // gc_alloc for managed allocation
    GcFree,      // gc_free for explicit deallocation
    GcCollect,   // gc_collect to trigger collection
    
    // Trinary literals
    Positive,    // +1
    Negative,    // -1
    Baseline,    // 0
    
    // Identifiers and literals
    Identifier(String),
    Number(f64),
    String(String),
    
    // Operators
    Pipe,        // |>
    Arrow,       // ->
    FatArrow,    // =>
    Plus,
    Minus,
    Star,
    Slash,
    Equal,
    EqualEqual,
    NotEqual,
    Less,
    Greater,
    LessEqual,    // <=
    GreaterEqual, // >=
    
    // Delimiters
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Comma,
    Semicolon,
    Colon,
    Dot,
    
    // Special
    Eof,
    Newline,
}

pub struct Lexer<'a> {
    input: Peekable<Chars<'a>>,
    current_char: Option<char>,
    location: SourceLocation,
    source_text: &'a str,  // Keep reference to source for error reporting
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut lexer = Lexer {
            input: input.chars().peekable(),
            current_char: None,
            location: SourceLocation::new(),
            source_text: input,
        };
        lexer.advance();
        lexer
    }
    
    fn advance(&mut self) {
        if let Some(ch) = self.current_char {
            self.location.advance(ch);
        }
        self.current_char = self.input.next();
    }
    
    fn peek(&mut self) -> Option<&char> {
        self.input.peek()
    }
    
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current_char {
            if ch.is_whitespace() && ch != '\n' {
                self.advance();
            } else {
                break;
            }
        }
    }
    
    fn skip_comment(&mut self) {
        if self.current_char == Some('/') && self.peek() == Some(&'/') {
            while self.current_char.is_some() && self.current_char != Some('\n') {
                self.advance();
            }
        }
    }
    
    fn read_identifier(&mut self) -> String {
        let mut result = String::new();
        
        while let Some(ch) = self.current_char {
            if ch.is_alphanumeric() || ch == '_' {
                result.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        result
    }
    
    fn read_number(&mut self) -> f64 {
        let mut result = String::new();
        let mut has_dot = false;
        
        // If we start with a dot, add it
        if self.current_char == Some('.') {
            result.push('.');
            has_dot = true;
            self.advance();
        }
        
        while let Some(ch) = self.current_char {
            if ch.is_numeric() {
                result.push(ch);
                self.advance();
            } else if ch == '.' && !has_dot {
                // Allow one dot for decimal numbers
                result.push(ch);
                has_dot = true;
                self.advance();
            } else {
                break;
            }
        }
        
        result.parse().unwrap_or(0.0)
    }
    
    fn read_string(&mut self) -> String {
        let mut result = String::new();
        self.advance(); // Skip opening quote
        
        while let Some(ch) = self.current_char {
            if ch == '"' {
                self.advance(); // Skip closing quote
                break;
            } else if ch == '\\' {
                self.advance();
                if let Some(escaped) = self.current_char {
                    match escaped {
                        'n' => result.push('\n'),
                        't' => result.push('\t'),
                        '\\' => result.push('\\'),
                        '"' => result.push('"'),
                        _ => result.push(escaped),
                    }
                    self.advance();
                }
            } else {
                result.push(ch);
                self.advance();
            }
        }
        
        result
    }
    
    pub fn next_located_token(&mut self) -> LocatedToken {
        let start_location = self.location.clone_at();
        let token = self.next_token_internal();
        LocatedToken::new(token, start_location)
    }
    
    pub fn next_token(&mut self) -> Token {
        self.next_token_internal()
    }
    
    fn next_token_internal(&mut self) -> Token {
        self.skip_whitespace();
        self.skip_comment();
        
        match self.current_char {
            None => Token::Eof,
            Some('\n') => {
                self.advance();
                Token::Newline
            }
            Some('"') => Token::String(self.read_string()),
            Some('+') => {
                self.advance();
                if self.current_char == Some('1') {
                    self.advance();
                    Token::Positive
                } else {
                    Token::Plus
                }
            }
            Some('-') => {
                self.advance();
                if self.current_char == Some('1') {
                    self.advance();
                    Token::Negative
                } else if self.current_char == Some('>') {
                    self.advance();
                    Token::Arrow
                } else {
                    Token::Minus
                }
            }
            Some('0') => {
                self.advance();
                Token::Baseline
            }
            Some('|') => {
                self.advance();
                if self.current_char == Some('>') {
                    self.advance();
                    Token::Pipe
                } else {
                    Token::Identifier("|".to_string())
                }
            }
            Some('=') => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Token::EqualEqual
                } else if self.current_char == Some('>') {
                    self.advance();
                    Token::FatArrow
                } else {
                    Token::Equal
                }
            }
            Some('!') => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Token::NotEqual
                } else {
                    Token::Identifier("!".to_string())
                }
            }
            Some('<') => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Token::LessEqual
                } else {
                    Token::Less
                }
            }
            Some('>') => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Token::GreaterEqual
                } else {
                    Token::Greater
                }
            }
            Some('(') => {
                self.advance();
                Token::LeftParen
            }
            Some(')') => {
                self.advance();
                Token::RightParen
            }
            Some('{') => {
                self.advance();
                Token::LeftBrace
            }
            Some('}') => {
                self.advance();
                Token::RightBrace
            }
            Some('[') => {
                self.advance();
                Token::LeftBracket
            }
            Some(']') => {
                self.advance();
                Token::RightBracket
            }
            Some(',') => {
                self.advance();
                Token::Comma
            }
            Some(';') => {
                self.advance();
                Token::Semicolon
            }
            Some(':') => {
                self.advance();
                Token::Colon
            }
            Some('.') => {
                // Check if this is a decimal number (next char is digit)
                if let Some(next_ch) = self.peek() {
                    if next_ch.is_numeric() {
                        // This is a decimal number starting with dot (like .5)
                        Token::Number(self.read_number())
                    } else {
                        // This is a regular dot token
                        self.advance();
                        Token::Dot
                    }
                } else {
                    // No next character, so it's just a dot
                    self.advance();
                    Token::Dot
                }
            }
            Some('*') => {
                self.advance();
                Token::Star
            }
            Some('/') => {
                self.advance();
                Token::Slash
            }
            Some(ch) if ch.is_alphabetic() || ch == '_' => {
                let ident = self.read_identifier();
                match ident.as_str() {
                    "organism" => Token::Organism,
                    "cell" => Token::Cell,
                    "neuron" => Token::Neuron,
                    "protein" => Token::Protein,
                    "gene" => Token::Gene,
                    "layer" => Token::Layer,
                    "evolve" => Token::Evolve,
                    "mutate" => Token::Mutate,
                    "express" => Token::Express,
                    "synthesize" => Token::Synthesize,
                    "network" => Token::Network,
                    "forward" => Token::Forward,
                    "backward" => Token::Backward,
                    "activation" => Token::Activation,
                    "weights" => Token::Weights,
                    "bias" => Token::Bias,
                    "gradient" => Token::Gradient,
                    "learn" => Token::Learn,
                    "train" => Token::Train,
                    "function" | "fn" => Token::Function,
                    "struct" => Token::Struct,
                    "enum" => Token::Enum,
                    "let" => Token::Let,
                    "if" => Token::If,
                    "else" => Token::Else,
                    "loop" => Token::Loop,
                    "while" => Token::While,
                    "for" => Token::For,
                    "in" => Token::In,
                    "return" => Token::Return,
                    "break" => Token::Break,
                    "continue" => Token::Continue,
                    "breakpoint" => Token::Breakpoint,
                    "match" => Token::Match,
                    "lambda" => Token::Lambda,
                    "import" => Token::Import,
                    "export" => Token::Export,
                    "module" => Token::Module,
                    "gc_alloc" => Token::GcAlloc,
                    "gc_free" => Token::GcFree,
                    "gc_collect" => Token::GcCollect,
                    "redis_connect" => Token::RedisConnect,
                    "redis_get" => Token::RedisGet,
                    "redis_set" => Token::RedisSet,
                    "redis_publish" => Token::RedisPublish,
                    "redis_subscribe" => Token::RedisSubscribe,
                    _ => Token::Identifier(ident),
                }
            }
            Some(ch) if ch.is_numeric() => {
                Token::Number(self.read_number())
            }
            Some(ch) => {
                self.advance();
                Token::Identifier(ch.to_string())
            }
        }
    }
    
    pub fn tokenize_with_locations(&mut self) -> Vec<LocatedToken> {
        let mut tokens = Vec::new();
        
        loop {
            let located_token = self.next_located_token();
            let is_eof = located_token.token == Token::Eof;
            tokens.push(located_token);
            if is_eof {
                break;
            }
        }
        
        tokens
    }
    
    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        
        loop {
            let token = self.next_token();
            if token == Token::Eof {
                tokens.push(token);
                break;
            } else if token != Token::Newline {
                tokens.push(token);
            }
        }
        
        tokens
    }
    
    pub fn get_source_line(&self, line_number: usize) -> Option<String> {
        self.source_text.lines().nth(line_number.saturating_sub(1)).map(|s| s.to_string())
    }
    
    pub fn get_source_text(&self) -> &str {
        self.source_text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_tokens() {
        let input = "organism cell { }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        
        assert_eq!(tokens, vec![
            Token::Organism,
            Token::Cell,
            Token::LeftBrace,
            Token::RightBrace,
            Token::Eof,
        ]);
    }
    
    #[test]
    fn test_trinary_values() {
        let input = "+1 -1 0";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        
        assert_eq!(tokens, vec![
            Token::Positive,
            Token::Negative,
            Token::Baseline,
            Token::Eof,
        ]);
    }
    
    #[test]
    fn test_pipe_operator() {
        let input = "data |> process |> output";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        
        assert_eq!(tokens, vec![
            Token::Identifier("data".to_string()),
            Token::Pipe,
            Token::Identifier("process".to_string()),
            Token::Pipe,
            Token::Identifier("output".to_string()),
            Token::Eof,
        ]);
    }
    
    #[test]
    fn test_function_definition() {
        let input = "fn metabolize(glucose) -> ATP";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        
        assert_eq!(tokens, vec![
            Token::Function,
            Token::Identifier("metabolize".to_string()),
            Token::LeftParen,
            Token::Identifier("glucose".to_string()),
            Token::RightParen,
            Token::Arrow,
            Token::Identifier("ATP".to_string()),
            Token::Eof,
        ]);
    }
}