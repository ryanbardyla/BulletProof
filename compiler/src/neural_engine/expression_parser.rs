// 🔬 EXPRESSION PARSER FOR LEMS DYNAMICS
// Proper mathematical expression parsing and evaluation for neural dynamics

use std::collections::HashMap;
use std::str::Chars;
use std::iter::Peekable;

/// Mathematical expression parser for LEMS dynamics equations
pub struct ExpressionParser {
    /// Built-in functions
    functions: HashMap<String, fn(&[f64]) -> f64>,
}

/// Abstract syntax tree for mathematical expressions
#[derive(Debug, Clone, PartialEq)]
pub enum AST {
    /// Binary operation: left op right
    Binary(BinaryOp, Box<AST>, Box<AST>),
    /// Unary operation: op expr
    Unary(UnaryOp, Box<AST>),
    /// Function call: function(args...)
    Function(String, Vec<AST>),
    /// Variable reference
    Variable(String),
    /// Numeric literal
    Number(f64),
    /// Conditional expression: if condition then expr1 else expr2
    Conditional(Box<AST>, Box<AST>, Box<AST>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,      // +
    Subtract, // -
    Multiply, // *
    Divide,   // /
    Power,    // ^
    Modulo,   // %
    Equal,    // ==
    NotEqual, // !=
    Less,     // <
    Greater,  // >
    LessEq,   // <=
    GreaterEq,// >=
    And,      // &&
    Or,       // ||
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Plus,     // +expr
    Minus,    // -expr
    Not,      // !expr
}

/// Token types for lexical analysis
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Number(f64),
    Identifier(String),
    Operator(String),
    LeftParen,
    RightParen,
    Comma,
    EOF,
}

/// Tokenizer for mathematical expressions
pub struct Tokenizer<'a> {
    input: Peekable<Chars<'a>>,
    current_char: Option<char>,
}

/// Expression evaluation context
pub struct EvaluationContext {
    /// Variable values
    variables: HashMap<String, f64>,
    /// Constant values
    constants: HashMap<String, f64>,
}

#[derive(Debug)]
pub enum ParseError {
    UnexpectedToken(String),
    UnexpectedEOF,
    InvalidNumber(String),
    UnknownFunction(String),
    UnknownVariable(String),
    DivisionByZero,
    InvalidArguments(String),
}

impl ExpressionParser {
    pub fn new() -> Self {
        let mut functions = HashMap::new();
        
        // Mathematical functions
        functions.insert("sin".to_string(), |args: &[f64]| args[0].sin());
        functions.insert("cos".to_string(), |args: &[f64]| args[0].cos());
        functions.insert("tan".to_string(), |args: &[f64]| args[0].tan());
        functions.insert("exp".to_string(), |args: &[f64]| args[0].exp());
        functions.insert("ln".to_string(), |args: &[f64]| args[0].ln());
        functions.insert("log".to_string(), |args: &[f64]| args[0].log10());
        functions.insert("sqrt".to_string(), |args: &[f64]| args[0].sqrt());
        functions.insert("abs".to_string(), |args: &[f64]| args[0].abs());
        functions.insert("floor".to_string(), |args: &[f64]| args[0].floor());
        functions.insert("ceil".to_string(), |args: &[f64]| args[0].ceil());
        
        // Multi-argument functions
        functions.insert("pow".to_string(), |args: &[f64]| args[0].powf(args[1]));
        functions.insert("min".to_string(), |args: &[f64]| args[0].min(args[1]));
        functions.insert("max".to_string(), |args: &[f64]| args[0].max(args[1]));
        functions.insert("atan2".to_string(), |args: &[f64]| args[0].atan2(args[1]));
        
        // Special neural functions
        functions.insert("sigmoid".to_string(), |args: &[f64]| 1.0 / (1.0 + (-args[0]).exp()));
        functions.insert("tanh".to_string(), |args: &[f64]| args[0].tanh());
        functions.insert("relu".to_string(), |args: &[f64]| args[0].max(0.0));
        
        ExpressionParser { functions }
    }
    
    /// Parse an expression string into an AST
    pub fn parse(&self, input: &str) -> Result<AST, ParseError> {
        let mut tokenizer = Tokenizer::new(input);
        let tokens = tokenizer.tokenize()?;
        let mut parser = Parser::new(tokens);
        parser.parse_expression()
    }
    
    /// Evaluate an AST with given context
    pub fn evaluate(&self, ast: &AST, context: &EvaluationContext) -> Result<f64, ParseError> {
        match ast {
            AST::Number(n) => Ok(*n),
            
            AST::Variable(name) => {
                // Try variables first, then constants
                if let Some(&value) = context.variables.get(name) {
                    Ok(value)
                } else if let Some(&value) = context.constants.get(name) {
                    Ok(value)
                } else {
                    Err(ParseError::UnknownVariable(name.clone()))
                }
            },
            
            AST::Binary(op, left, right) => {
                let left_val = self.evaluate(left, context)?;
                let right_val = self.evaluate(right, context)?;
                
                match op {
                    BinaryOp::Add => Ok(left_val + right_val),
                    BinaryOp::Subtract => Ok(left_val - right_val),
                    BinaryOp::Multiply => Ok(left_val * right_val),
                    BinaryOp::Divide => {
                        if right_val.abs() < f64::EPSILON {
                            Err(ParseError::DivisionByZero)
                        } else {
                            Ok(left_val / right_val)
                        }
                    },
                    BinaryOp::Power => Ok(left_val.powf(right_val)),
                    BinaryOp::Modulo => Ok(left_val % right_val),
                    BinaryOp::Equal => Ok(if (left_val - right_val).abs() < f64::EPSILON { 1.0 } else { 0.0 }),
                    BinaryOp::NotEqual => Ok(if (left_val - right_val).abs() >= f64::EPSILON { 1.0 } else { 0.0 }),
                    BinaryOp::Less => Ok(if left_val < right_val { 1.0 } else { 0.0 }),
                    BinaryOp::Greater => Ok(if left_val > right_val { 1.0 } else { 0.0 }),
                    BinaryOp::LessEq => Ok(if left_val <= right_val { 1.0 } else { 0.0 }),
                    BinaryOp::GreaterEq => Ok(if left_val >= right_val { 1.0 } else { 0.0 }),
                    BinaryOp::And => Ok(if left_val != 0.0 && right_val != 0.0 { 1.0 } else { 0.0 }),
                    BinaryOp::Or => Ok(if left_val != 0.0 || right_val != 0.0 { 1.0 } else { 0.0 }),
                }
            },
            
            AST::Unary(op, expr) => {
                let val = self.evaluate(expr, context)?;
                
                match op {
                    UnaryOp::Plus => Ok(val),
                    UnaryOp::Minus => Ok(-val),
                    UnaryOp::Not => Ok(if val == 0.0 { 1.0 } else { 0.0 }),
                }
            },
            
            AST::Function(name, args) => {
                if let Some(func) = self.functions.get(name) {
                    let arg_values: Result<Vec<f64>, ParseError> = args.iter()
                        .map(|arg| self.evaluate(arg, context))
                        .collect();
                    
                    let arg_values = arg_values?;
                    Ok(func(&arg_values))
                } else {
                    Err(ParseError::UnknownFunction(name.clone()))
                }
            },
            
            AST::Conditional(condition, then_expr, else_expr) => {
                let condition_val = self.evaluate(condition, context)?;
                
                if condition_val != 0.0 {
                    self.evaluate(then_expr, context)
                } else {
                    self.evaluate(else_expr, context)
                }
            },
        }
    }
    
    /// Evaluate a string expression directly
    pub fn evaluate_string(&self, expression: &str, context: &EvaluationContext) -> Result<f64, ParseError> {
        let ast = self.parse(expression)?;
        self.evaluate(&ast, context)
    }
}

impl<'a> Tokenizer<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut chars = input.chars().peekable();
        let current_char = chars.next();
        
        Tokenizer {
            input: chars,
            current_char,
        }
    }
    
    pub fn tokenize(&mut self) -> Result<Vec<Token>, ParseError> {
        let mut tokens = Vec::new();
        
        while let Some(ch) = self.current_char {
            match ch {
                // Skip whitespace
                ' ' | '\t' | '\n' | '\r' => {
                    self.advance();
                },
                
                // Numbers
                '0'..='9' | '.' => {
                    tokens.push(self.read_number()?);
                },
                
                // Identifiers and keywords
                'a'..='z' | 'A'..='Z' | '_' => {
                    tokens.push(self.read_identifier());
                },
                
                // Operators and punctuation
                '+' => {
                    self.advance();
                    tokens.push(Token::Operator("+".to_string()));
                },
                '-' => {
                    self.advance();
                    tokens.push(Token::Operator("-".to_string()));
                },
                '*' => {
                    self.advance();
                    if self.current_char == Some('*') {
                        self.advance();
                        tokens.push(Token::Operator("**".to_string()));
                    } else {
                        tokens.push(Token::Operator("*".to_string()));
                    }
                },
                '/' => {
                    self.advance();
                    tokens.push(Token::Operator("/".to_string()));
                },
                '^' => {
                    self.advance();
                    tokens.push(Token::Operator("^".to_string()));
                },
                '%' => {
                    self.advance();
                    tokens.push(Token::Operator("%".to_string()));
                },
                '=' => {
                    self.advance();
                    if self.current_char == Some('=') {
                        self.advance();
                        tokens.push(Token::Operator("==".to_string()));
                    } else {
                        tokens.push(Token::Operator("=".to_string()));
                    }
                },
                '!' => {
                    self.advance();
                    if self.current_char == Some('=') {
                        self.advance();
                        tokens.push(Token::Operator("!=".to_string()));
                    } else {
                        tokens.push(Token::Operator("!".to_string()));
                    }
                },
                '<' => {
                    self.advance();
                    if self.current_char == Some('=') {
                        self.advance();
                        tokens.push(Token::Operator("<=".to_string()));
                    } else {
                        tokens.push(Token::Operator("<".to_string()));
                    }
                },
                '>' => {
                    self.advance();
                    if self.current_char == Some('=') {
                        self.advance();
                        tokens.push(Token::Operator(">=".to_string()));
                    } else {
                        tokens.push(Token::Operator(">".to_string()));
                    }
                },
                '&' => {
                    self.advance();
                    if self.current_char == Some('&') {
                        self.advance();
                        tokens.push(Token::Operator("&&".to_string()));
                    } else {
                        return Err(ParseError::UnexpectedToken("&".to_string()));
                    }
                },
                '|' => {
                    self.advance();
                    if self.current_char == Some('|') {
                        self.advance();
                        tokens.push(Token::Operator("||".to_string()));
                    } else {
                        return Err(ParseError::UnexpectedToken("|".to_string()));
                    }
                },
                '(' => {
                    self.advance();
                    tokens.push(Token::LeftParen);
                },
                ')' => {
                    self.advance();
                    tokens.push(Token::RightParen);
                },
                ',' => {
                    self.advance();
                    tokens.push(Token::Comma);
                },
                
                _ => {
                    return Err(ParseError::UnexpectedToken(ch.to_string()));
                }
            }
        }
        
        tokens.push(Token::EOF);
        Ok(tokens)
    }
    
    fn advance(&mut self) {
        self.current_char = self.input.next();
    }
    
    fn read_number(&mut self) -> Result<Token, ParseError> {
        let mut number_str = String::new();
        let mut has_dot = false;
        
        while let Some(ch) = self.current_char {
            match ch {
                '0'..='9' => {
                    number_str.push(ch);
                    self.advance();
                },
                '.' => {
                    if has_dot {
                        break;
                    }
                    has_dot = true;
                    number_str.push(ch);
                    self.advance();
                },
                _ => break,
            }
        }
        
        number_str.parse::<f64>()
            .map(Token::Number)
            .map_err(|_| ParseError::InvalidNumber(number_str))
    }
    
    fn read_identifier(&mut self) -> Token {
        let mut identifier = String::new();
        
        while let Some(ch) = self.current_char {
            match ch {
                'a'..='z' | 'A'..='Z' | '0'..='9' | '_' => {
                    identifier.push(ch);
                    self.advance();
                },
                _ => break,
            }
        }
        
        Token::Identifier(identifier)
    }
}

/// Recursive descent parser
struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, current: 0 }
    }
    
    fn parse_expression(&mut self) -> Result<AST, ParseError> {
        self.parse_logical_or()
    }
    
    fn parse_logical_or(&mut self) -> Result<AST, ParseError> {
        let mut left = self.parse_logical_and()?;
        
        while self.match_operator("||") {
            let right = self.parse_logical_and()?;
            left = AST::Binary(BinaryOp::Or, Box::new(left), Box::new(right));
        }
        
        Ok(left)
    }
    
    fn parse_logical_and(&mut self) -> Result<AST, ParseError> {
        let mut left = self.parse_equality()?;
        
        while self.match_operator("&&") {
            let right = self.parse_equality()?;
            left = AST::Binary(BinaryOp::And, Box::new(left), Box::new(right));
        }
        
        Ok(left)
    }
    
    fn parse_equality(&mut self) -> Result<AST, ParseError> {
        let mut left = self.parse_comparison()?;
        
        while let Some(op) = self.match_operators(&["==", "!="]) {
            let binary_op = match op.as_str() {
                "==" => BinaryOp::Equal,
                "!=" => BinaryOp::NotEqual,
                _ => unreachable!(),
            };
            let right = self.parse_comparison()?;
            left = AST::Binary(binary_op, Box::new(left), Box::new(right));
        }
        
        Ok(left)
    }
    
    fn parse_comparison(&mut self) -> Result<AST, ParseError> {
        let mut left = self.parse_addition()?;
        
        while let Some(op) = self.match_operators(&["<", ">", "<=", ">="]) {
            let binary_op = match op.as_str() {
                "<" => BinaryOp::Less,
                ">" => BinaryOp::Greater,
                "<=" => BinaryOp::LessEq,
                ">=" => BinaryOp::GreaterEq,
                _ => unreachable!(),
            };
            let right = self.parse_addition()?;
            left = AST::Binary(binary_op, Box::new(left), Box::new(right));
        }
        
        Ok(left)
    }
    
    fn parse_addition(&mut self) -> Result<AST, ParseError> {
        let mut left = self.parse_multiplication()?;
        
        while let Some(op) = self.match_operators(&["+", "-"]) {
            let binary_op = match op.as_str() {
                "+" => BinaryOp::Add,
                "-" => BinaryOp::Subtract,
                _ => unreachable!(),
            };
            let right = self.parse_multiplication()?;
            left = AST::Binary(binary_op, Box::new(left), Box::new(right));
        }
        
        Ok(left)
    }
    
    fn parse_multiplication(&mut self) -> Result<AST, ParseError> {
        let mut left = self.parse_power()?;
        
        while let Some(op) = self.match_operators(&["*", "/", "%"]) {
            let binary_op = match op.as_str() {
                "*" => BinaryOp::Multiply,
                "/" => BinaryOp::Divide,
                "%" => BinaryOp::Modulo,
                _ => unreachable!(),
            };
            let right = self.parse_power()?;
            left = AST::Binary(binary_op, Box::new(left), Box::new(right));
        }
        
        Ok(left)
    }
    
    fn parse_power(&mut self) -> Result<AST, ParseError> {
        let mut left = self.parse_unary()?;
        
        // Right associative
        if self.match_operator("^") || self.match_operator("**") {
            let right = self.parse_power()?;
            left = AST::Binary(BinaryOp::Power, Box::new(left), Box::new(right));
        }
        
        Ok(left)
    }
    
    fn parse_unary(&mut self) -> Result<AST, ParseError> {
        if let Some(op) = self.match_operators(&["+", "-", "!"]) {
            let unary_op = match op.as_str() {
                "+" => UnaryOp::Plus,
                "-" => UnaryOp::Minus,
                "!" => UnaryOp::Not,
                _ => unreachable!(),
            };
            let expr = self.parse_unary()?;
            Ok(AST::Unary(unary_op, Box::new(expr)))
        } else {
            self.parse_primary()
        }
    }
    
    fn parse_primary(&mut self) -> Result<AST, ParseError> {
        match self.current_token() {
            Token::Number(n) => {
                self.advance();
                Ok(AST::Number(*n))
            },
            
            Token::Identifier(name) => {
                let name = name.clone();
                self.advance();
                
                // Check if it's a function call
                if self.match_token(&Token::LeftParen) {
                    let mut args = Vec::new();
                    
                    if !self.check_token(&Token::RightParen) {
                        loop {
                            args.push(self.parse_expression()?);
                            
                            if !self.match_token(&Token::Comma) {
                                break;
                            }
                        }
                    }
                    
                    self.consume_token(&Token::RightParen)?;
                    Ok(AST::Function(name, args))
                } else {
                    Ok(AST::Variable(name))
                }
            },
            
            Token::LeftParen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.consume_token(&Token::RightParen)?;
                Ok(expr)
            },
            
            token => Err(ParseError::UnexpectedToken(format!("{:?}", token))),
        }
    }
    
    fn current_token(&self) -> &Token {
        self.tokens.get(self.current).unwrap_or(&Token::EOF)
    }
    
    fn advance(&mut self) {
        if self.current < self.tokens.len() {
            self.current += 1;
        }
    }
    
    fn match_token(&mut self, expected: &Token) -> bool {
        if std::mem::discriminant(self.current_token()) == std::mem::discriminant(expected) {
            self.advance();
            true
        } else {
            false
        }
    }
    
    fn match_operator(&mut self, op: &str) -> bool {
        if let Token::Operator(current_op) = self.current_token() {
            if current_op == op {
                self.advance();
                return true;
            }
        }
        false
    }
    
    fn match_operators(&mut self, ops: &[&str]) -> Option<String> {
        if let Token::Operator(current_op) = self.current_token() {
            for &op in ops {
                if current_op == op {
                    let result = current_op.clone();
                    self.advance();
                    return Some(result);
                }
            }
        }
        None
    }
    
    fn check_token(&self, expected: &Token) -> bool {
        std::mem::discriminant(self.current_token()) == std::mem::discriminant(expected)
    }
    
    fn consume_token(&mut self, expected: &Token) -> Result<(), ParseError> {
        if self.check_token(expected) {
            self.advance();
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken(format!("Expected {:?}, found {:?}", expected, self.current_token())))
        }
    }
}

impl EvaluationContext {
    pub fn new() -> Self {
        let mut constants = HashMap::new();
        
        // Mathematical constants
        constants.insert("pi".to_string(), std::f64::consts::PI);
        constants.insert("e".to_string(), std::f64::consts::E);
        constants.insert("tau".to_string(), 2.0 * std::f64::consts::PI);
        
        // Physical constants (in SI units)
        constants.insert("k_B".to_string(), 1.380649e-23); // Boltzmann constant
        constants.insert("N_A".to_string(), 6.02214076e23); // Avogadro's number
        constants.insert("R".to_string(), 8.314462618); // Gas constant
        
        EvaluationContext {
            variables: HashMap::new(),
            constants,
        }
    }
    
    pub fn set_variable(&mut self, name: &str, value: f64) {
        self.variables.insert(name.to_string(), value);
    }
    
    pub fn set_variables(&mut self, vars: HashMap<String, f64>) {
        self.variables.extend(vars);
    }
    
    pub fn get_variable(&self, name: &str) -> Option<f64> {
        self.variables.get(name).copied()
            .or_else(|| self.constants.get(name).copied())
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ParseError::UnexpectedToken(token) => write!(f, "Unexpected token: {}", token),
            ParseError::UnexpectedEOF => write!(f, "Unexpected end of input"),
            ParseError::InvalidNumber(num) => write!(f, "Invalid number: {}", num),
            ParseError::UnknownFunction(func) => write!(f, "Unknown function: {}", func),
            ParseError::UnknownVariable(var) => write!(f, "Unknown variable: {}", var),
            ParseError::DivisionByZero => write!(f, "Division by zero"),
            ParseError::InvalidArguments(msg) => write!(f, "Invalid arguments: {}", msg),
        }
    }
}

impl std::error::Error for ParseError {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_arithmetic() {
        let parser = ExpressionParser::new();
        let mut context = EvaluationContext::new();
        context.set_variable("x", 10.0);
        
        assert_eq!(parser.evaluate_string("2 + 3", &context).unwrap(), 5.0);
        assert_eq!(parser.evaluate_string("2 * 3 + 4", &context).unwrap(), 10.0);
        assert_eq!(parser.evaluate_string("(2 + 3) * 4", &context).unwrap(), 20.0);
        assert_eq!(parser.evaluate_string("x + 5", &context).unwrap(), 15.0);
    }
    
    #[test]
    fn test_functions() {
        let parser = ExpressionParser::new();
        let context = EvaluationContext::new();
        
        assert!((parser.evaluate_string("sin(0)", &context).unwrap() - 0.0).abs() < 1e-10);
        assert!((parser.evaluate_string("exp(0)", &context).unwrap() - 1.0).abs() < 1e-10);
        assert!((parser.evaluate_string("sqrt(16)", &context).unwrap() - 4.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_hodgkin_huxley_expressions() {
        let parser = ExpressionParser::new();
        let mut context = EvaluationContext::new();
        
        // Set up HH variables
        context.set_variable("v", -65.0);
        context.set_variable("m", 0.05);
        context.set_variable("h", 0.6);
        context.set_variable("n", 0.32);
        context.set_variable("gNa", 120.0);
        context.set_variable("gK", 36.0);
        context.set_variable("gL", 0.3);
        context.set_variable("ENa", 50.0);
        context.set_variable("EK", -77.0);
        context.set_variable("EL", -54.4);
        
        // Test rate constants
        let alpha_m = parser.evaluate_string("0.1 * (v + 40) / (1 - exp(-(v + 40) / 10))", &context).unwrap();
        assert!(alpha_m > 0.0);
        
        // Test ionic currents
        let i_na = parser.evaluate_string("gNa * m^3 * h * (v - ENa)", &context).unwrap();
        assert!(i_na != 0.0);
    }
    
    #[test]
    fn test_conditionals() {
        let parser = ExpressionParser::new();
        let mut context = EvaluationContext::new();
        context.set_variable("x", 5.0);
        
        // Test comparison operators
        assert_eq!(parser.evaluate_string("x > 3", &context).unwrap(), 1.0);
        assert_eq!(parser.evaluate_string("x < 3", &context).unwrap(), 0.0);
        assert_eq!(parser.evaluate_string("x == 5", &context).unwrap(), 1.0);
    }
    
    #[test]
    fn test_complex_expressions() {
        let parser = ExpressionParser::new();
        let mut context = EvaluationContext::new();
        context.set_variable("v", -50.0);
        
        // Complex neural dynamics expression
        let expr = "0.1 * (v + 40) / (1 - exp(-(v + 40) / 10)) * (1 - 0.05) - 4 * exp(-(v + 65) / 18) * 0.05";
        let result = parser.evaluate_string(expr, &context);
        assert!(result.is_ok());
    }
}