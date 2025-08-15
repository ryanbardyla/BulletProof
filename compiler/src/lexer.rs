//! NeuronLang Lexer - Tokenizes .nl source files
//! 
//! The first step in compiling NeuronLang: breaking source code into tokens.
//! Handles trinary literals, temporal units, and biological keywords.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    TryteLiteral(TryteValue),
    IntegerLiteral(i32),
    FloatLiteral(f32),
    StringLiteral(String),
    BooleanLiteral(bool),
    VoltageLiteral(f32), // in mV
    TimeLiteral(f32, TimeUnit),
    
    // Keywords - Neural
    Brain,
    Layer,
    Neuron,
    Synapse,
    Spiking,
    Lif,
    Izhikevich,
    Trinary,
    
    // Keywords - Actions
    Fire,
    Forget,
    Rest,
    Propagate,
    Consolidate,
    Synthesize,
    Compress,
    
    // Keywords - Memory & Learning
    Ewc,
    Stdp,
    Hebbian,
    Meta,
    Learn,
    Adapt,
    Pattern,
    Temporal,
    
    // Keywords - Proteins
    Creb,
    Bdnf,
    Arc,
    PkmZeta,
    
    // Keywords - Trading
    Trade,
    Buy,
    Sell,
    Close,
    HyperLiquid,
    Redis,
    
    // Keywords - Control Flow
    If,
    Else,
    Match,
    For,
    While,
    When,
    Then,
    Return,
    
    // Keywords - Types
    Tryte,
    Spike,
    Bool,
    I32,
    F32,
    String,
    
    // Identifiers and Operators
    Identifier(String),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Power,
    Equal,
    EqualEqual,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    And,
    Or,
    Xor,
    Not,
    
    // Punctuation
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Comma,
    Dot,
    Colon,
    Semicolon,
    Arrow,
    FatArrow,
    Bang,
    At,
    
    // Special
    Eof,
    Newline,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TryteValue {
    Inhibited,  // -1
    Baseline,   // 0
    Activated,  // +1
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeUnit {
    Microseconds,
    Milliseconds,
    Seconds,
    Minutes,
    Hours,
}

pub struct Lexer {
    input: Vec<char>,
    position: usize,
    current_char: Option<char>,
    line: usize,
    column: usize,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        let chars: Vec<char> = input.chars().collect();
        let current_char = if chars.is_empty() { None } else { Some(chars[0]) };
        
        Lexer {
            input: chars,
            position: 0,
            current_char,
            line: 1,
            column: 1,
        }
    }
    
    fn advance(&mut self) {
        self.position += 1;
        if self.position >= self.input.len() {
            self.current_char = None;
        } else {
            self.current_char = Some(self.input[self.position]);
            self.column += 1;
        }
        
        if self.current_char == Some('\n') {
            self.line += 1;
            self.column = 1;
        }
    }
    
    fn peek(&self, offset: usize) -> Option<char> {
        let pos = self.position + offset;
        if pos < self.input.len() {
            Some(self.input[pos])
        } else {
            None
        }
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
        if self.current_char == Some('/') && self.peek(1) == Some('/') {
            // Line comment
            while self.current_char.is_some() && self.current_char != Some('\n') {
                self.advance();
            }
        } else if self.current_char == Some('/') && self.peek(1) == Some('*') {
            // Block comment
            self.advance(); // skip /
            self.advance(); // skip *
            while self.current_char.is_some() {
                if self.current_char == Some('*') && self.peek(1) == Some('/') {
                    self.advance(); // skip *
                    self.advance(); // skip /
                    break;
                }
                self.advance();
            }
        }
    }
    
    fn read_number(&mut self) -> Token {
        let mut num_str = String::new();
        let mut is_float = false;
        let mut is_negative = false;
        
        // Handle negative numbers
        if self.current_char == Some('-') {
            is_negative = true;
            num_str.push('-');
            self.advance();
        }
        
        // Check for tryte literals first
        if self.current_char == Some('1') && !self.peek(1).map_or(false, |c| c.is_ascii_digit() || c == '.') {
            self.advance();
            return if is_negative {
                Token::TryteLiteral(TryteValue::Inhibited)
            } else {
                Token::TryteLiteral(TryteValue::Activated)
            };
        }
        
        if self.current_char == Some('0') && !self.peek(1).map_or(false, |c| c.is_ascii_digit() || c == '.') {
            self.advance();
            return Token::TryteLiteral(TryteValue::Baseline);
        }
        
        // Read the number
        while let Some(ch) = self.current_char {
            if ch.is_ascii_digit() {
                num_str.push(ch);
                self.advance();
            } else if ch == '.' && !is_float {
                is_float = true;
                num_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        // Check for units (mV, ms, us, s, etc.)
        if self.current_char == Some('m') {
            if self.peek(1) == Some('V') {
                self.advance(); // skip m
                self.advance(); // skip V
                let voltage: f32 = num_str.parse().unwrap_or(0.0);
                return Token::VoltageLiteral(voltage);
            } else if self.peek(1) == Some('s') {
                self.advance(); // skip m
                self.advance(); // skip s
                let time: f32 = num_str.parse().unwrap_or(0.0);
                return Token::TimeLiteral(time, TimeUnit::Milliseconds);
            }
        } else if self.current_char == Some('u') && self.peek(1) == Some('s') {
            self.advance(); // skip u
            self.advance(); // skip s
            let time: f32 = num_str.parse().unwrap_or(0.0);
            return Token::TimeLiteral(time, TimeUnit::Microseconds);
        } else if self.current_char == Some('s') && !self.peek(1).map_or(false, |c| c.is_alphabetic()) {
            self.advance(); // skip s
            let time: f32 = num_str.parse().unwrap_or(0.0);
            return Token::TimeLiteral(time, TimeUnit::Seconds);
        }
        
        // Parse as regular number
        if is_float {
            Token::FloatLiteral(num_str.parse().unwrap_or(0.0))
        } else {
            Token::IntegerLiteral(num_str.parse().unwrap_or(0))
        }
    }
    
    fn read_identifier(&mut self) -> Token {
        let mut ident = String::new();
        
        while let Some(ch) = self.current_char {
            if ch.is_alphanumeric() || ch == '_' {
                ident.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        // Check for keywords
        match ident.as_str() {
            // Neural keywords
            "brain" => Token::Brain,
            "layer" => Token::Layer,
            "neuron" => Token::Neuron,
            "synapse" => Token::Synapse,
            "spiking" => Token::Spiking,
            "lif" => Token::Lif,
            "izhikevich" => Token::Izhikevich,
            "trinary" => Token::Trinary,
            
            // Action keywords
            "fire" => Token::Fire,
            "forget" => Token::Forget,
            "rest" => Token::Rest,
            "propagate" => Token::Propagate,
            "consolidate" => Token::Consolidate,
            "synthesize" => Token::Synthesize,
            "compress" => Token::Compress,
            
            // Learning keywords
            "ewc" => Token::Ewc,
            "stdp" => Token::Stdp,
            "hebbian" => Token::Hebbian,
            "meta" => Token::Meta,
            "learn" => Token::Learn,
            "adapt" => Token::Adapt,
            "pattern" => Token::Pattern,
            "temporal" => Token::Temporal,
            
            // Protein keywords
            "CREB" => Token::Creb,
            "BDNF" => Token::Bdnf,
            "Arc" => Token::Arc,
            "PKMzeta" => Token::PkmZeta,
            
            // Trading keywords
            "trade" => Token::Trade,
            "buy" => Token::Buy,
            "sell" => Token::Sell,
            "close" => Token::Close,
            "hyperliquid" => Token::HyperLiquid,
            "redis" => Token::Redis,
            
            // Control flow
            "if" => Token::If,
            "else" => Token::Else,
            "match" => Token::Match,
            "for" => Token::For,
            "while" => Token::While,
            "when" => Token::When,
            "then" => Token::Then,
            "return" => Token::Return,
            
            // Types
            "tryte" => Token::Tryte,
            "spike" => Token::Spike,
            "bool" => Token::Bool,
            "i32" => Token::I32,
            "f32" => Token::F32,
            "string" => Token::String,
            
            // Boolean literals
            "true" => Token::BooleanLiteral(true),
            "false" => Token::BooleanLiteral(false),
            
            // Tryte literals
            "Inhibited" => Token::TryteLiteral(TryteValue::Inhibited),
            "Baseline" => Token::TryteLiteral(TryteValue::Baseline),
            "Activated" => Token::TryteLiteral(TryteValue::Activated),
            
            // Logical operators
            "and" => Token::And,
            "or" => Token::Or,
            "xor" => Token::Xor,
            "not" => Token::Not,
            
            // Default to identifier
            _ => Token::Identifier(ident),
        }
    }
    
    fn read_string(&mut self) -> Token {
        let mut string = String::new();
        self.advance(); // skip opening quote
        
        while let Some(ch) = self.current_char {
            if ch == '"' {
                self.advance(); // skip closing quote
                break;
            } else if ch == '\\' {
                self.advance();
                if let Some(escaped) = self.current_char {
                    match escaped {
                        'n' => string.push('\n'),
                        't' => string.push('\t'),
                        'r' => string.push('\r'),
                        '\\' => string.push('\\'),
                        '"' => string.push('"'),
                        _ => {
                            string.push('\\');
                            string.push(escaped);
                        }
                    }
                    self.advance();
                }
            } else {
                string.push(ch);
                self.advance();
            }
        }
        
        Token::StringLiteral(string)
    }
    
    pub fn next_token(&mut self) -> Token {
        loop {
            self.skip_whitespace();
            
            if self.current_char.is_none() {
                return Token::Eof;
            }
            
            // Skip comments
            if self.current_char == Some('/') && 
               (self.peek(1) == Some('/') || self.peek(1) == Some('*')) {
                self.skip_comment();
                continue;
            }
            
            break;
        }
        
        let ch = match self.current_char {
            Some(c) => c,
            None => return Token::Eof,
        };
        
        // Numbers (including negative)
        if ch.is_ascii_digit() || (ch == '-' && self.peek(1).map_or(false, |c| c.is_ascii_digit())) {
            return self.read_number();
        }
        
        // Identifiers and keywords
        if ch.is_alphabetic() || ch == '_' {
            return self.read_identifier();
        }
        
        // Strings
        if ch == '"' {
            return self.read_string();
        }
        
        // Single character tokens and operators
        let token = match ch {
            '+' => {
                self.advance();
                if self.current_char == Some('1') {
                    self.advance();
                    Token::TryteLiteral(TryteValue::Activated)
                } else {
                    Token::Plus
                }
            }
            '-' => {
                self.advance();
                if self.current_char == Some('1') && !self.peek(1).map_or(false, |c| c.is_ascii_digit()) {
                    self.advance();
                    Token::TryteLiteral(TryteValue::Inhibited)
                } else if self.current_char == Some('>') {
                    self.advance();
                    Token::Arrow
                } else {
                    Token::Minus
                }
            }
            '*' => {
                self.advance();
                if self.current_char == Some('*') {
                    self.advance();
                    Token::Power
                } else {
                    Token::Star
                }
            }
            '/' => {
                self.advance();
                Token::Slash
            }
            '%' => {
                self.advance();
                Token::Percent
            }
            '=' => {
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
            '!' => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Token::NotEqual
                } else {
                    Token::Bang
                }
            }
            '<' => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Token::LessEqual
                } else {
                    Token::Less
                }
            }
            '>' => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Token::GreaterEqual
                } else {
                    Token::Greater
                }
            }
            '(' => {
                self.advance();
                Token::LeftParen
            }
            ')' => {
                self.advance();
                Token::RightParen
            }
            '{' => {
                self.advance();
                Token::LeftBrace
            }
            '}' => {
                self.advance();
                Token::RightBrace
            }
            '[' => {
                self.advance();
                Token::LeftBracket
            }
            ']' => {
                self.advance();
                Token::RightBracket
            }
            ',' => {
                self.advance();
                Token::Comma
            }
            '.' => {
                self.advance();
                Token::Dot
            }
            ':' => {
                self.advance();
                Token::Colon
            }
            ';' => {
                self.advance();
                Token::Semicolon
            }
            '@' => {
                self.advance();
                Token::At
            }
            '\n' => {
                self.advance();
                Token::Newline
            }
            _ => {
                self.advance();
                Token::Identifier(ch.to_string()) // Unknown character as identifier
            }
        };
        
        token
    }
    
    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        
        loop {
            let token = self.next_token();
            if token == Token::Eof {
                tokens.push(token);
                break;
            }
            // Skip newlines for now (can keep them if needed for grammar)
            if token != Token::Newline {
                tokens.push(token);
            }
        }
        
        tokens
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::TryteLiteral(v) => write!(f, "Tryte({:?})", v),
            Token::IntegerLiteral(n) => write!(f, "Int({})", n),
            Token::FloatLiteral(n) => write!(f, "Float({})", n),
            Token::StringLiteral(s) => write!(f, "String(\"{}\")", s),
            Token::BooleanLiteral(b) => write!(f, "Bool({})", b),
            Token::VoltageLiteral(v) => write!(f, "Voltage({}mV)", v),
            Token::TimeLiteral(t, u) => write!(f, "Time({}{:?})", t, u),
            Token::Identifier(s) => write!(f, "Id({})", s),
            _ => write!(f, "{:?}", self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tryte_literals() {
        let mut lexer = Lexer::new("-1 0 +1 Inhibited Baseline Activated");
        let tokens = lexer.tokenize();
        
        assert_eq!(tokens[0], Token::TryteLiteral(TryteValue::Inhibited));
        assert_eq!(tokens[1], Token::TryteLiteral(TryteValue::Baseline));
        assert_eq!(tokens[2], Token::TryteLiteral(TryteValue::Activated));
        assert_eq!(tokens[3], Token::TryteLiteral(TryteValue::Inhibited));
        assert_eq!(tokens[4], Token::TryteLiteral(TryteValue::Baseline));
        assert_eq!(tokens[5], Token::TryteLiteral(TryteValue::Activated));
    }
    
    #[test]
    fn test_voltage_and_time() {
        let mut lexer = Lexer::new("-70mV 20ms 100us 5s");
        let tokens = lexer.tokenize();
        
        assert_eq!(tokens[0], Token::VoltageLiteral(-70.0));
        assert_eq!(tokens[1], Token::TimeLiteral(20.0, TimeUnit::Milliseconds));
        assert_eq!(tokens[2], Token::TimeLiteral(100.0, TimeUnit::Microseconds));
        assert_eq!(tokens[3], Token::TimeLiteral(5.0, TimeUnit::Seconds));
    }
    
    #[test]
    fn test_fire_forget_syntax() {
        let mut lexer = Lexer::new("fire!(+1) forget!(membrane: -80mV)");
        let tokens = lexer.tokenize();
        
        assert_eq!(tokens[0], Token::Fire);
        assert_eq!(tokens[1], Token::Bang);
        assert_eq!(tokens[2], Token::LeftParen);
        assert_eq!(tokens[3], Token::TryteLiteral(TryteValue::Activated));
        assert_eq!(tokens[4], Token::RightParen);
        
        assert_eq!(tokens[5], Token::Forget);
        assert_eq!(tokens[6], Token::Bang);
    }
    
    #[test]
    fn test_brain_declaration() {
        let source = r#"
        brain MyBrain {
            layer input[1000]
            synapses {
                plasticity: stdp(window: 20ms, ltp: 0.1, ltd: 0.05)
            }
        }
        "#;
        
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        
        assert_eq!(tokens[0], Token::Brain);
        assert_eq!(tokens[1], Token::Identifier("MyBrain".to_string()));
        assert_eq!(tokens[2], Token::LeftBrace);
    }
}