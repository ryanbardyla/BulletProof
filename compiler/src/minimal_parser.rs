// 🧬 MINIMAL NEURONLANG PARSER
// Converts tokens into Abstract Syntax Tree (AST)

use crate::minimal_lexer::{Token, Lexer};
use crate::source_location::{SourceLocation, LocatedToken, SourceSpan, LocatedError};

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // Literals
    Tryte(TryteValue),
    Number(f64),
    String(String),
    Identifier(String),
    
    // Arrays!
    Array(Vec<Expr>),
    Array2D(Vec<Vec<Expr>>),  // 2D arrays for matrices!
    
    // HashMap/Dictionary
    HashMap(Vec<(Expr, Expr)>), // Vec of (key, value) pairs
    
    // Struct creation
    StructInit {
        name: String,
        fields: Vec<(String, Expr)>,
    },
    
    // Field access
    FieldAccess {
        object: Box<Expr>,
        field: String,
    },
    
    // Enum variant
    EnumVariant {
        enum_name: String,
        variant: String,
    },
    
    Index {
        array: Box<Expr>,
        index: Box<Expr>,
    },
    
    // Binary operations
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOperator,
        right: Box<Expr>,
    },
    
    // Unary operations
    UnaryOp {
        op: UnaryOperator,
        operand: Box<Expr>,
    },
    
    // Pipe operation (special!)
    Pipe {
        input: Box<Expr>,
        stages: Vec<Expr>,
    },
    
    // Function call
    Call {
        name: String,
        args: Vec<Expr>,
    },
    
    // Lambda expression/closure
    Lambda {
        params: Vec<String>,
        body: Vec<Statement>,
    },
    
    // Biological operations
    Express(Box<Expr>),
    Synthesize(Box<Expr>),
    Mutate(Box<Expr>),
    
    // Neural Network primitives
    Neuron {
        weights: Box<Expr>,
        bias: Box<Expr>,
        activation: ActivationType,
    },
    Layer {
        neurons: Vec<Expr>,
        input_size: usize,
        output_size: usize,
    },
    Forward {
        layer: Box<Expr>,
        input: Box<Expr>,
    },
    Backward {
        layer: Box<Expr>,
        gradient: Box<Expr>,
        learning_rate: f64,
    },
    
    // Redis operations (REAL!)
    RedisConnect(String, u16), // host, port
    RedisGet(Box<Expr>),       // key
    RedisSet(Box<Expr>, Box<Expr>), // key, value
    RedisPublish(Box<Expr>, Box<Expr>), // channel, message
    RedisSubscribe(Box<Expr>),  // channel
    
    // Control flow
    If {
        condition: Box<Expr>,
        then_body: Vec<Statement>,
        else_body: Option<Vec<Statement>>,
    },
    
    // Match expression
    Match {
        expr: Box<Expr>,
        arms: Vec<MatchArm>,
    },
    
    // Block expression
    Block(Vec<Statement>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TryteValue {
    Positive,  // +1
    Baseline,  // 0
    Negative,  // -1
}

#[derive(Debug, Clone, PartialEq)]
pub enum ActivationType {
    Relu,
    Sigmoid,
    Tanh,
    Linear,
    Softmax,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: MatchPattern,
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MatchPattern {
    Literal(Expr),          // 42, "hello", +1
    Wildcard,              // _
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Plus,
    Minus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    // Variable binding
    Let {
        name: String,
        type_annotation: Option<String>,  // Optional type annotation
        value: Expr,
    },
    
    // Assignment (name = value)
    Assignment {
        name: String,
        value: Expr,
    },
    
    // Indexed assignment (array[index] = value)
    IndexedAssignment {
        array: String,
        index: Expr,
        value: Expr,
    },
    
    // Expression statement
    Expression(Expr),
    
    // Return statement
    Return(Option<Expr>),
    
    // Break and Continue statements
    Break,
    Continue,
    
    // Debug breakpoint
    Breakpoint,
    
    // Loop
    Loop {
        body: Vec<Statement>,
    },
    
    // While loop
    While {
        condition: Expr,
        body: Vec<Statement>,
    },
    
    // For loop
    For {
        variable: String,
        start: Expr,
        end: Expr,
        body: Vec<Statement>,
    },
    
    // For-in loop (iterate array)
    ForIn {
        variable: String,
        array: Expr,
        body: Vec<Statement>,
    },
    
    // If statement
    If {
        condition: Expr,
        then_body: Vec<Statement>,
        else_body: Option<Vec<Statement>>,
    },
    
    // Evolution!
    Evolve {
        target: String,
        generations: Option<usize>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    // Organism (main container)
    Organism {
        name: String,
        body: Vec<Declaration>,
    },
    
    // Cell (module)
    Cell {
        name: String,
        body: Vec<Declaration>,
    },
    
    // Function
    Function {
        name: String,
        params: Vec<String>,
        return_type: Option<String>,
        body: Vec<Statement>,
    },
    
    // Struct definition
    Struct {
        name: String,
        fields: Vec<(String, String)>, // (field_name, field_type)
    },
    
    // Enum definition
    Enum {
        name: String,
        variants: Vec<String>, // variant names
    },
    
    // Neural Network declarations
    NeuronDecl {
        name: String,
        input_size: usize,
        activation: ActivationType,
    },
    
    LayerDecl {
        name: String,
        input_size: usize,
        output_size: usize,
        activation: ActivationType,
    },
    
    NetworkDecl {
        name: String,
        layers: Vec<String>,  // Names of layers
    },
    
    // Gene (constant)
    Gene {
        name: String,
        value: Expr,
    },
    
    // Protein (type/struct)
    Protein {
        name: String,
        fields: Vec<(String, String)>,
    },
    
    // Import statement
    Import {
        module_path: String,  // e.g., "std.math" or "neural.layers"
        items: Option<Vec<String>>, // Some(["sin", "cos"]) or None for wildcard
    },
    
    // Module definition
    Module {
        name: String,
        exports: Vec<String>, // Names of exported functions/types
        body: Vec<Declaration>,
    },
    
    // Neural Network Layer
    Layer {
        name: String,
        input_size: usize,
        output_size: usize,
        activation: String,
        weights: Vec<Vec<f64>>,
        bias: Vec<f64>,
        ewc_importance: f64,  // For EWC
    },
}

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser {
            tokens,
            current: 0,
        }
    }
    
    pub fn get_current_index(&self) -> Option<usize> {
        if self.current < self.tokens.len() {
            Some(self.current)
        } else {
            None
        }
    }
    
    fn peek(&self) -> &Token {
        self.tokens.get(self.current).unwrap_or(&Token::Eof)
    }
    
    fn advance(&mut self) -> Token {
        let token = self.peek().clone();
        self.current += 1;
        token
    }
    
    fn expect(&mut self, expected: Token) -> Result<(), String> {
        let token = self.advance();
        if token != expected {
            Err(format!("Expected {:?}, found {:?}", expected, token))
        } else {
            Ok(())
        }
    }
    
    fn check(&self, token: &Token) -> bool {
        self.peek() == token
    }
    
    fn match_token(&mut self, tokens: &[Token]) -> bool {
        for token in tokens {
            if self.check(token) {
                self.advance();
                return true;
            }
        }
        false
    }
    
    fn skip_newlines(&mut self) {
        while self.check(&Token::Newline) {
            self.advance();
        }
    }
    
    // Helper to parse a block of declarations/statements inside braces
    fn parse_block_declarations(&mut self) -> Result<Vec<Declaration>, String> {
        let mut body = Vec::new();
        self.skip_newlines();
        
        while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
            self.skip_newlines();
            if self.check(&Token::RightBrace) || self.check(&Token::Eof) {
                break;
            }
            body.push(self.parse_declaration()?);
            self.skip_newlines();
        }
        
        Ok(body)
    }
    
    // Helper to parse a block of statements inside braces
    fn parse_block_statements(&mut self) -> Result<Vec<Statement>, String> {
        let mut body = Vec::new();
        self.skip_newlines();
        
        while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
            self.skip_newlines();
            if self.check(&Token::RightBrace) || self.check(&Token::Eof) {
                break;
            }
            body.push(self.parse_statement()?);
            self.skip_newlines();
        }
        
        Ok(body)
    }
    
    // Parse top-level declarations
    pub fn parse(&mut self) -> Result<Vec<Declaration>, String> {
        let mut declarations = Vec::new();
        
        while !self.check(&Token::Eof) {
            self.skip_newlines();
            if self.check(&Token::Eof) {
                break;
            }
            declarations.push(self.parse_declaration()?);
            self.skip_newlines();
        }
        
        Ok(declarations)
    }
    
    fn parse_declaration(&mut self) -> Result<Declaration, String> {
        match self.peek() {
            Token::Organism => self.parse_organism(),
            Token::Cell => self.parse_cell(),
            Token::Function => self.parse_function(),
            Token::Struct => self.parse_struct(),
            Token::Enum => self.parse_enum(),
            Token::Gene => self.parse_gene(),
            Token::Protein => self.parse_protein(),
            Token::Neuron => self.parse_neuron_decl(),
            Token::Layer => self.parse_layer_decl(),
            Token::Network => self.parse_network_decl(),
            Token::Import => self.parse_import(),
            Token::Module => self.parse_module(),
            _ => Err(format!("Unexpected token: {:?}", self.peek())),
        }
    }
    
    fn parse_organism(&mut self) -> Result<Declaration, String> {
        self.expect(Token::Organism)?;
        
        let name = if let Token::Identifier(n) = self.advance() {
            n
        } else {
            return Err("Expected organism name".to_string());
        };
        
        self.expect(Token::LeftBrace)?;
        let body = self.parse_block_declarations()?;
        
        self.expect(Token::RightBrace)?;
        
        Ok(Declaration::Organism { name, body })
    }
    
    fn parse_cell(&mut self) -> Result<Declaration, String> {
        self.expect(Token::Cell)?;
        
        let name = if let Token::Identifier(n) = self.advance() {
            n
        } else {
            return Err("Expected cell name".to_string());
        };
        
        self.expect(Token::LeftBrace)?;
        
        let mut body = Vec::new();
        while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
            body.push(self.parse_declaration()?);
        }
        
        self.expect(Token::RightBrace)?;
        
        Ok(Declaration::Cell { name, body })
    }
    
    fn parse_function(&mut self) -> Result<Declaration, String> {
        self.expect(Token::Function)?;
        
        let name = if let Token::Identifier(n) = self.advance() {
            n
        } else {
            return Err("Expected function name".to_string());
        };
        
        self.expect(Token::LeftParen)?;
        
        let mut params = Vec::new();
        while !self.check(&Token::RightParen) {
            if let Token::Identifier(param) = self.advance() {
                params.push(param);
                if !self.check(&Token::RightParen) {
                    self.expect(Token::Comma)?;
                }
            } else {
                return Err("Expected parameter name".to_string());
            }
        }
        
        self.expect(Token::RightParen)?;
        
        let return_type = if self.check(&Token::Arrow) {
            self.advance();
            if let Token::Identifier(typ) = self.advance() {
                Some(typ)
            } else {
                return Err("Expected return type".to_string());
            }
        } else {
            None
        };
        
        self.expect(Token::LeftBrace)?;
        let body = self.parse_block_statements()?;
        self.expect(Token::RightBrace)?;
        
        Ok(Declaration::Function {
            name,
            params,
            return_type,
            body,
        })
    }
    
    fn parse_enum(&mut self) -> Result<Declaration, String> {
        self.expect(Token::Enum)?;
        
        let name = if let Token::Identifier(n) = self.advance() {
            n
        } else {
            return Err("Expected enum name".to_string());
        };
        
        self.expect(Token::LeftBrace)?;
        
        let mut variants = Vec::new();
        while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
            // Parse variant name
            let variant_name = if let Token::Identifier(n) = self.advance() {
                n
            } else {
                return Err("Expected variant name".to_string());
            };
            
            variants.push(variant_name);
            
            // Check for comma
            if self.check(&Token::Comma) {
                self.advance();
            } else if !self.check(&Token::RightBrace) {
                // If not at end and no comma, it's an error
                return Err("Expected comma or closing brace".to_string());
            }
        }
        
        self.expect(Token::RightBrace)?;
        
        Ok(Declaration::Enum { name, variants })
    }
    
    fn parse_struct(&mut self) -> Result<Declaration, String> {
        self.expect(Token::Struct)?;
        
        let name = if let Token::Identifier(n) = self.advance() {
            n
        } else {
            return Err("Expected struct name".to_string());
        };
        
        self.expect(Token::LeftBrace)?;
        
        let mut fields = Vec::new();
        while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
            // Parse field name
            let field_name = if let Token::Identifier(n) = self.advance() {
                n
            } else {
                return Err("Expected field name".to_string());
            };
            
            // Expect colon
            self.expect(Token::Colon)?;
            
            // Parse field type
            let field_type = if let Token::Identifier(t) = self.advance() {
                t
            } else {
                return Err("Expected field type".to_string());
            };
            
            fields.push((field_name, field_type));
            
            // Check for comma or end of struct
            if !self.check(&Token::RightBrace) {
                if self.check(&Token::Comma) {
                    self.advance();
                }
            }
        }
        
        self.expect(Token::RightBrace)?;
        
        Ok(Declaration::Struct { name, fields })
    }
    
    fn parse_gene(&mut self) -> Result<Declaration, String> {
        self.expect(Token::Gene)?;
        
        let name = if let Token::Identifier(n) = self.advance() {
            n
        } else {
            return Err("Expected gene name".to_string());
        };
        
        self.expect(Token::Equal)?;
        
        let value = self.parse_expression()?;
        
        Ok(Declaration::Gene { name, value })
    }
    
    fn parse_protein(&mut self) -> Result<Declaration, String> {
        self.expect(Token::Protein)?;
        
        let name = if let Token::Identifier(n) = self.advance() {
            n
        } else {
            return Err("Expected protein name".to_string());
        };
        
        self.expect(Token::LeftBrace)?;
        
        let mut fields = Vec::new();
        while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
            if let Token::Identifier(field_name) = self.advance() {
                self.expect(Token::Colon)?;
                if let Token::Identifier(field_type) = self.advance() {
                    fields.push((field_name, field_type));
                    if !self.check(&Token::RightBrace) {
                        self.expect(Token::Comma)?;
                    }
                }
            }
        }
        
        self.expect(Token::RightBrace)?;
        
        Ok(Declaration::Protein { name, fields })
    }
    
    fn parse_neuron_decl(&mut self) -> Result<Declaration, String> {
        self.expect(Token::Neuron)?;
        
        let name = if let Token::Identifier(n) = self.advance() {
            n
        } else {
            return Err("Expected neuron name".to_string());
        };
        
        self.expect(Token::LeftParen)?;
        
        // Parse input_size
        let input_size = if let Token::Number(n) = self.advance() {
            n as usize
        } else {
            return Err("Expected input size".to_string());
        };
        
        self.expect(Token::Comma)?;
        
        // Parse activation function
        let activation = if let Token::Identifier(act) = self.advance() {
            match act.as_str() {
                "relu" => ActivationType::Relu,
                "sigmoid" => ActivationType::Sigmoid,
                "tanh" => ActivationType::Tanh,
                "linear" => ActivationType::Linear,
                "softmax" => ActivationType::Softmax,
                _ => return Err(format!("Unknown activation function: {}", act)),
            }
        } else {
            return Err("Expected activation function".to_string());
        };
        
        self.expect(Token::RightParen)?;
        
        Ok(Declaration::NeuronDecl { name, input_size, activation })
    }
    
    fn parse_layer_decl(&mut self) -> Result<Declaration, String> {
        self.expect(Token::Layer)?;
        
        let name = if let Token::Identifier(n) = self.advance() {
            n
        } else {
            return Err("Expected layer name".to_string());
        };
        
        self.expect(Token::LeftParen)?;
        
        // Parse input_size
        let input_size = if let Token::Number(n) = self.advance() {
            n as usize
        } else {
            return Err("Expected input size".to_string());
        };
        
        self.expect(Token::Comma)?;
        
        // Parse output_size
        let output_size = if let Token::Number(n) = self.advance() {
            n as usize
        } else {
            return Err("Expected output size".to_string());
        };
        
        self.expect(Token::Comma)?;
        
        // Parse activation
        let activation = if let Token::Identifier(act) = self.advance() {
            match act.as_str() {
                "relu" => ActivationType::Relu,
                "sigmoid" => ActivationType::Sigmoid,
                "tanh" => ActivationType::Tanh,
                "linear" => ActivationType::Linear,
                "softmax" => ActivationType::Softmax,
                _ => return Err(format!("Unknown activation function: {}", act)),
            }
        } else {
            return Err("Expected activation function".to_string());
        };
        
        self.expect(Token::RightParen)?;
        
        Ok(Declaration::LayerDecl { name, input_size, output_size, activation })
    }
    
    fn parse_network_decl(&mut self) -> Result<Declaration, String> {
        self.expect(Token::Network)?;
        
        let name = if let Token::Identifier(n) = self.advance() {
            n
        } else {
            return Err("Expected network name".to_string());
        };
        
        self.expect(Token::LeftBrace)?;
        
        let mut layers = Vec::new();
        while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
            if let Token::Identifier(layer_name) = self.advance() {
                layers.push(layer_name);
                if !self.check(&Token::RightBrace) {
                    // Optional comma or arrow between layers
                    if self.check(&Token::Comma) {
                        self.advance();
                    } else if self.check(&Token::Arrow) {
                        self.advance();
                    }
                }
            } else {
                return Err("Expected layer name in network".to_string());
            }
        }
        
        self.expect(Token::RightBrace)?;
        
        Ok(Declaration::NetworkDecl { name, layers })
    }
    
    fn parse_statement(&mut self) -> Result<Statement, String> {
        match self.peek() {
            Token::Let => self.parse_let(),
            Token::Return => self.parse_return(),
            Token::Break => self.parse_break(),
            Token::Continue => self.parse_continue(),
            Token::Breakpoint => self.parse_breakpoint(),
            Token::Loop => self.parse_loop(),
            Token::While => self.parse_while(),
            Token::For => self.parse_for(),
            Token::If => self.parse_if(),
            Token::Evolve => self.parse_evolve(),
            Token::Identifier(name) => {
                // Check if it's an assignment or indexed assignment
                let name_copy = name.clone();
                self.advance(); // consume identifier
                
                // Check for array indexing
                if self.check(&Token::LeftBracket) {
                    // It's an indexed assignment: array[index] = value
                    self.advance(); // consume [
                    let index = self.parse_expression()?;
                    self.expect(Token::RightBracket)?;
                    
                    if self.check(&Token::Equal) {
                        self.advance(); // consume =
                        let value = self.parse_expression()?;
                        if self.check(&Token::Semicolon) {
                            self.advance();
                        }
                        Ok(Statement::IndexedAssignment { 
                            array: name_copy, 
                            index, 
                            value 
                        })
                    } else {
                        // It's just an array access expression
                        self.current -= 3; // Go back to identifier
                        let expr = self.parse_expression()?;
                        if self.check(&Token::Semicolon) {
                            self.advance();
                        }
                        Ok(Statement::Expression(expr))
                    }
                } else if self.check(&Token::Equal) {
                    // It's a simple assignment!
                    self.advance(); // consume =
                    let value = self.parse_expression()?;
                    if self.check(&Token::Semicolon) {
                        self.advance();
                    }
                    Ok(Statement::Assignment { name: name_copy, value })
                } else {
                    // Put the identifier back and parse as expression
                    self.current -= 1;
                    let expr = self.parse_expression()?;
                    if self.check(&Token::Semicolon) {
                        self.advance();
                    }
                    Ok(Statement::Expression(expr))
                }
            },
            _ => {
                let expr = self.parse_expression()?;
                if self.check(&Token::Semicolon) {
                    self.advance();
                }
                Ok(Statement::Expression(expr))
            }
        }
    }
    
    fn parse_let(&mut self) -> Result<Statement, String> {
        self.expect(Token::Let)?;
        
        let name = if let Token::Identifier(n) = self.advance() {
            n
        } else {
            return Err("Expected variable name".to_string());
        };
        
        // Check for optional type annotation
        let type_annotation = if self.check(&Token::Colon) {
            self.advance(); // consume :
            if let Token::Identifier(type_name) = self.advance() {
                Some(type_name)
            } else {
                return Err("Expected type name after ':'".to_string());
            }
        } else {
            None
        };
        
        self.expect(Token::Equal)?;
        
        let value = self.parse_expression()?;
        
        if self.check(&Token::Semicolon) {
            self.advance();
        }
        
        Ok(Statement::Let { name, type_annotation, value })
    }
    
    fn parse_return(&mut self) -> Result<Statement, String> {
        self.expect(Token::Return)?;
        
        let value = if self.check(&Token::Semicolon) || self.check(&Token::RightBrace) {
            None
        } else {
            Some(self.parse_expression()?)
        };
        
        if self.check(&Token::Semicolon) {
            self.advance();
        }
        
        Ok(Statement::Return(value))
    }
    
    fn parse_break(&mut self) -> Result<Statement, String> {
        self.expect(Token::Break)?;
        
        if self.check(&Token::Semicolon) {
            self.advance();
        }
        
        Ok(Statement::Break)
    }
    
    fn parse_continue(&mut self) -> Result<Statement, String> {
        self.expect(Token::Continue)?;
        
        if self.check(&Token::Semicolon) {
            self.advance();
        }
        
        Ok(Statement::Continue)
    }
    
    fn parse_breakpoint(&mut self) -> Result<Statement, String> {
        self.expect(Token::Breakpoint)?;
        
        if self.check(&Token::Semicolon) {
            self.advance();
        }
        
        Ok(Statement::Breakpoint)
    }
    
    fn parse_loop(&mut self) -> Result<Statement, String> {
        self.expect(Token::Loop)?;
        self.expect(Token::LeftBrace)?;
        
        let mut body = Vec::new();
        while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
            body.push(self.parse_statement()?);
        }
        
        self.expect(Token::RightBrace)?;
        
        Ok(Statement::Loop { body })
    }
    
    fn parse_if(&mut self) -> Result<Statement, String> {
        self.expect(Token::If)?;
        
        // Parse condition
        let condition = self.parse_expression()?;
        
        // Parse then body
        self.expect(Token::LeftBrace)?;
        let mut then_body = Vec::new();
        while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
            then_body.push(self.parse_statement()?);
        }
        self.expect(Token::RightBrace)?;
        
        // Check for else
        let else_body = if self.match_token(&[Token::Else]) {
            self.expect(Token::LeftBrace)?;
            let mut body = Vec::new();
            while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
                body.push(self.parse_statement()?);
            }
            self.expect(Token::RightBrace)?;
            Some(body)
        } else {
            None
        };
        
        Ok(Statement::If { condition, then_body, else_body })
    }
    
    fn parse_while(&mut self) -> Result<Statement, String> {
        self.expect(Token::While)?;
        
        // Parse condition
        let condition = self.parse_expression()?;
        
        // Parse body
        self.expect(Token::LeftBrace)?;
        let mut body = Vec::new();
        while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
            body.push(self.parse_statement()?);
        }
        self.expect(Token::RightBrace)?;
        
        Ok(Statement::While { condition, body })
    }
    
    fn parse_for(&mut self) -> Result<Statement, String> {
        self.expect(Token::For)?;
        
        // Get loop variable
        let variable = match self.advance() {
            Token::Identifier(name) => name,
            _ => return Err("Expected identifier after 'for'".to_string()),
        };
        
        // Check for 'in' (for-in loop)
        if self.check(&Token::In) {
            self.advance();
            let array = self.parse_expression()?;
            
            self.expect(Token::LeftBrace)?;
            let mut body = Vec::new();
            while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
                body.push(self.parse_statement()?);
            }
            self.expect(Token::RightBrace)?;
            
            Ok(Statement::ForIn { variable, array, body })
        } else if self.check(&Token::Equal) {
            // Simple for loop: for i = 0; i < 10; i++ {
            self.advance();
            let start = self.parse_expression()?;
            
            // For now, simplified version
            // Just parse: for i = start_value { ... }
            // And assume it goes from start to start+5
            let end = Expr::Number(5.0);
            
            self.expect(Token::LeftBrace)?;
            let mut body = Vec::new();
            while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
                body.push(self.parse_statement()?);
            }
            self.expect(Token::RightBrace)?;
            
            Ok(Statement::For { variable, start, end, body })
        } else {
            Err("Expected 'in' or '=' after for loop variable".to_string())
        }
    }
    
    fn parse_evolve(&mut self) -> Result<Statement, String> {
        self.expect(Token::Evolve)?;
        
        let target = if let Token::Identifier(t) = self.advance() {
            t
        } else {
            return Err("Expected evolution target".to_string());
        };
        
        let generations = if self.check(&Token::LeftParen) {
            self.advance();
            let gen = if let Token::Number(n) = self.advance() {
                Some(n as usize)
            } else {
                None
            };
            self.expect(Token::RightParen)?;
            gen
        } else {
            None
        };
        
        Ok(Statement::Evolve { target, generations })
    }
    
    fn parse_expression(&mut self) -> Result<Expr, String> {
        self.parse_pipe()
    }
    
    fn parse_pipe(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_equality()?;
        
        if self.check(&Token::Pipe) {
            let mut stages = Vec::new();
            
            while self.match_token(&[Token::Pipe]) {
                stages.push(self.parse_equality()?);
            }
            
            expr = Expr::Pipe {
                input: Box::new(expr),
                stages,
            };
        }
        
        Ok(expr)
    }
    
    fn parse_equality(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_comparison()?;
        
        while self.match_token(&[Token::EqualEqual, Token::NotEqual]) {
            let op = match self.tokens[self.current - 1] {
                Token::EqualEqual => BinaryOperator::Equal,
                Token::NotEqual => BinaryOperator::NotEqual,
                _ => unreachable!(),
            };
            let right = self.parse_comparison()?;
            expr = Expr::BinaryOp {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    fn parse_comparison(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_term()?;
        
        while self.match_token(&[Token::Greater, Token::Less, Token::GreaterEqual, Token::LessEqual]) {
            let op = match self.tokens[self.current - 1] {
                Token::Greater => BinaryOperator::Greater,
                Token::Less => BinaryOperator::Less,
                Token::GreaterEqual => BinaryOperator::GreaterEqual,
                Token::LessEqual => BinaryOperator::LessEqual,
                _ => unreachable!(),
            };
            let right = self.parse_term()?;
            expr = Expr::BinaryOp {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    fn parse_term(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_factor()?;
        
        while self.match_token(&[Token::Plus, Token::Minus]) {
            let op = match self.tokens[self.current - 1] {
                Token::Plus => BinaryOperator::Add,
                Token::Minus => BinaryOperator::Subtract,
                _ => unreachable!(),
            };
            let right = self.parse_factor()?;
            expr = Expr::BinaryOp {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    fn parse_factor(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_unary()?;
        
        while self.match_token(&[Token::Star, Token::Slash]) {
            let op = match self.tokens[self.current - 1] {
                Token::Star => BinaryOperator::Multiply,
                Token::Slash => BinaryOperator::Divide,
                _ => unreachable!(),
            };
            let right = self.parse_unary()?;
            expr = Expr::BinaryOp {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }
        
        Ok(expr)
    }
    
    fn parse_unary(&mut self) -> Result<Expr, String> {
        if self.match_token(&[Token::Express]) {
            return Ok(Expr::Express(Box::new(self.parse_unary()?)));
        }
        
        if self.match_token(&[Token::Synthesize]) {
            return Ok(Expr::Synthesize(Box::new(self.parse_unary()?)));
        }
        
        if self.match_token(&[Token::Mutate]) {
            return Ok(Expr::Mutate(Box::new(self.parse_unary()?)));
        }
        
        // Handle unary minus
        if self.match_token(&[Token::Minus]) {
            let expr = self.parse_unary()?;
            return Ok(Expr::UnaryOp {
                op: UnaryOperator::Minus,
                operand: Box::new(expr),
            });
        }
        
        // Handle unary plus (optional)
        if self.match_token(&[Token::Plus]) {
            let expr = self.parse_unary()?;
            return Ok(Expr::UnaryOp {
                op: UnaryOperator::Plus,
                operand: Box::new(expr),
            });
        }
        
        self.parse_postfix()  // Changed to use postfix for array indexing
    }
    
    fn parse_postfix(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_primary()?;
        
        loop {
            // Handle array indexing
            if self.check(&Token::LeftBracket) {
                self.advance();
                let index = self.parse_expression()?;
                self.expect(Token::RightBracket)?;
                expr = Expr::Index {
                    array: Box::new(expr),
                    index: Box::new(index),
                };
            }
            // Handle field access
            else if self.check(&Token::Dot) {
                self.advance();
                let field = if let Token::Identifier(f) = self.advance() {
                    f
                } else {
                    return Err("Expected field name after '.'".to_string());
                };
                expr = Expr::FieldAccess {
                    object: Box::new(expr),
                    field,
                };
            }
            else {
                break;
            }
        }
        
        Ok(expr)
    }
    
    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Token::Positive => {
                self.advance();
                Ok(Expr::Tryte(TryteValue::Positive))
            }
            Token::Negative => {
                self.advance();
                Ok(Expr::Tryte(TryteValue::Negative))
            }
            Token::Baseline => {
                self.advance();
                Ok(Expr::Tryte(TryteValue::Baseline))
            }
            Token::Number(n) => {
                let num = *n;
                self.advance();
                Ok(Expr::Number(num))
            }
            Token::String(s) => {
                let string = s.clone();
                self.advance();
                Ok(Expr::String(string))
            }
            Token::Lambda => {
                self.advance();
                self.parse_lambda()
            }
            Token::Forward => {
                self.advance();
                self.expect(Token::LeftParen)?;
                let layer = Box::new(self.parse_expression()?);
                self.expect(Token::Comma)?;
                let input = Box::new(self.parse_expression()?);
                self.expect(Token::RightParen)?;
                Ok(Expr::Forward { layer, input })
            }
            Token::Backward => {
                self.advance();
                self.expect(Token::LeftParen)?;
                let layer = Box::new(self.parse_expression()?);
                self.expect(Token::Comma)?;
                let gradient = Box::new(self.parse_expression()?);
                self.expect(Token::Comma)?;
                // Parse learning rate as an expression
                let lr_expr = self.parse_expression()?;
                let learning_rate = match lr_expr {
                    Expr::Number(n) => n,
                    _ => 0.01, // Default learning rate
                };
                self.expect(Token::RightParen)?;
                Ok(Expr::Backward { layer, gradient, learning_rate })
            }
            Token::Identifier(name) => {
                let ident = name.clone();
                self.advance();
                
                // Check for enum variant (Name::Variant)
                if self.check(&Token::Colon) {
                    // Check if it's double colon
                    let next_pos = self.current + 1;
                    if next_pos < self.tokens.len() && self.tokens[next_pos] == Token::Colon {
                        self.advance(); // consume first colon
                        self.advance(); // consume second colon
                        
                        if let Token::Identifier(variant) = self.advance() {
                            return Ok(Expr::EnumVariant {
                                enum_name: ident,
                                variant,
                            });
                        } else {
                            return Err("Expected variant name after ::".to_string());
                        }
                    }
                    // Otherwise, backtrack - it's not a double colon
                    self.current -= 1;
                }
                
                // Check for struct initialization
                if self.check(&Token::LeftBrace) {
                    self.advance();
                    let mut fields = Vec::new();
                    
                    while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
                        // Parse field name
                        let field_name = if let Token::Identifier(n) = self.advance() {
                            n
                        } else {
                            return Err("Expected field name in struct initialization".to_string());
                        };
                        
                        // Expect colon
                        self.expect(Token::Colon)?;
                        
                        // Parse field value
                        let field_value = self.parse_expression()?;
                        
                        fields.push((field_name, field_value));
                        
                        // Check for comma
                        if !self.check(&Token::RightBrace) {
                            if self.check(&Token::Comma) {
                                self.advance();
                            }
                        }
                    }
                    
                    self.expect(Token::RightBrace)?;
                    Ok(Expr::StructInit { name: ident, fields })
                }
                // Check for function call
                else if self.check(&Token::LeftParen) {
                    self.advance();
                    let mut args = Vec::new();
                    
                    while !self.check(&Token::RightParen) && !self.check(&Token::Eof) {
                        args.push(self.parse_expression()?);
                        if !self.check(&Token::RightParen) {
                            self.expect(Token::Comma)?;
                        }
                    }
                    
                    self.expect(Token::RightParen)?;
                    Ok(Expr::Call { name: ident, args })
                } else {
                    Ok(Expr::Identifier(ident))
                }
            }
            Token::LeftParen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.expect(Token::RightParen)?;
                Ok(expr)
            }
            Token::LeftBracket => {
                // Array literal [1, 2, 3] or 2D array [[1,2],[3,4]]
                self.advance();
                let mut elements = Vec::new();
                let mut is_2d = false;
                
                // Handle empty array
                if self.check(&Token::RightBracket) {
                    self.advance();
                    return Ok(Expr::Array(elements));
                }
                
                // Check if first element is an array (making this 2D)
                if self.check(&Token::LeftBracket) {
                    is_2d = true;
                    let mut rows = Vec::new();
                    
                    // Parse 2D array
                    loop {
                        self.expect(Token::LeftBracket)?;
                        let mut row = Vec::new();
                        
                        loop {
                            row.push(self.parse_expression()?);
                            if self.check(&Token::Comma) {
                                self.advance();
                                if self.check(&Token::RightBracket) {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                        
                        self.expect(Token::RightBracket)?;
                        rows.push(row);
                        
                        if self.check(&Token::Comma) {
                            self.advance();
                            if self.check(&Token::RightBracket) {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    
                    self.expect(Token::RightBracket)?;
                    return Ok(Expr::Array2D(rows));
                }
                
                // Parse 1D array
                loop {
                    elements.push(self.parse_expression()?);
                    
                    if self.check(&Token::Comma) {
                        self.advance();
                        // Check for trailing comma
                        if self.check(&Token::RightBracket) {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                
                self.expect(Token::RightBracket)?;
                Ok(Expr::Array(elements))
            }
            Token::If => self.parse_if_expression(),
            Token::Match => self.parse_match_expression(),
            Token::LeftBrace => {
                // Check if it's a HashMap literal or block expression
                // Look ahead to see if we have key: value pattern
                let mut lookahead = self.current + 1;
                let mut is_hashmap = false;
                
                // Simple heuristic: if we see identifier/string followed by colon, it's likely a HashMap
                if lookahead < self.tokens.len() {
                    match &self.tokens[lookahead] {
                        Token::Identifier(_) | Token::String(_) | Token::Number(_) => {
                            if lookahead + 1 < self.tokens.len() && self.tokens[lookahead + 1] == Token::Colon {
                                is_hashmap = true;
                            }
                        }
                        Token::RightBrace => {
                            // Empty braces {} - treat as empty HashMap
                            is_hashmap = true;
                        }
                        _ => {}
                    }
                }
                
                if is_hashmap {
                    self.parse_hashmap_literal()
                } else {
                    self.parse_block_expression()
                }
            },
            _ => Err(format!("Unexpected token in expression: {:?}", self.peek())),
        }
    }
    
    fn parse_if_expression(&mut self) -> Result<Expr, String> {
        self.expect(Token::If)?;
        
        let condition = Box::new(self.parse_expression()?);
        
        self.expect(Token::LeftBrace)?;
        let mut then_body = Vec::new();
        while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
            then_body.push(self.parse_statement()?);
        }
        self.expect(Token::RightBrace)?;
        
        let else_body = if self.match_token(&[Token::Identifier("else".to_string())]) {
            self.expect(Token::LeftBrace)?;
            let mut body = Vec::new();
            while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
                body.push(self.parse_statement()?);
            }
            self.expect(Token::RightBrace)?;
            Some(body)
        } else {
            None
        };
        
        Ok(Expr::If {
            condition,
            then_body,
            else_body,
        })
    }
    
    fn parse_block_expression(&mut self) -> Result<Expr, String> {
        self.expect(Token::LeftBrace)?;
        
        let mut statements = Vec::new();
        while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
            statements.push(self.parse_statement()?);
        }
        
        self.expect(Token::RightBrace)?;
        
        Ok(Expr::Block(statements))
    }
    
    fn parse_hashmap_literal(&mut self) -> Result<Expr, String> {
        self.expect(Token::LeftBrace)?;
        
        let mut pairs = Vec::new();
        
        // Handle empty HashMap {}
        if self.check(&Token::RightBrace) {
            self.advance();
            return Ok(Expr::HashMap(pairs));
        }
        
        // Parse key-value pairs
        loop {
            // Parse key
            let key = self.parse_expression()?;
            
            // Expect colon
            self.expect(Token::Colon)?;
            
            // Parse value
            let value = self.parse_expression()?;
            
            // Add pair
            pairs.push((key, value));
            
            // Check for comma or end
            if self.check(&Token::Comma) {
                self.advance();
                // Allow trailing comma
                if self.check(&Token::RightBrace) {
                    break;
                }
            } else {
                break;
            }
        }
        
        self.expect(Token::RightBrace)?;
        Ok(Expr::HashMap(pairs))
    }
    
    fn parse_match_expression(&mut self) -> Result<Expr, String> {
        self.expect(Token::Match)?;
        
        // Parse the expression to match on
        let expr = Box::new(self.parse_expression()?);
        
        self.expect(Token::LeftBrace)?;
        
        let mut arms = Vec::new();
        
        // Parse match arms
        while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
            let arm = self.parse_match_arm()?;
            arms.push(arm);
        }
        
        self.expect(Token::RightBrace)?;
        
        Ok(Expr::Match { expr, arms })
    }
    
    fn parse_match_arm(&mut self) -> Result<MatchArm, String> {
        // Parse pattern
        let pattern = self.parse_match_pattern()?;
        
        // Expect =>
        self.expect(Token::FatArrow)?;
        
        // Parse body - either a single expression or a block
        let mut body = Vec::new();
        
        if self.check(&Token::LeftBrace) {
            // Block body
            self.advance();
            while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
                body.push(self.parse_statement()?);
            }
            self.expect(Token::RightBrace)?;
        } else {
            // Expression body
            let expr = self.parse_expression()?;
            body.push(Statement::Expression(expr));
        }
        
        // Optional comma
        if self.check(&Token::Comma) {
            self.advance();
        }
        
        Ok(MatchArm { pattern, body })
    }
    
    fn parse_match_pattern(&mut self) -> Result<MatchPattern, String> {
        match self.peek() {
            Token::Identifier(name) if name == "_" => {
                self.advance();
                Ok(MatchPattern::Wildcard)
            }
            _ => {
                let expr = self.parse_primary()?;
                Ok(MatchPattern::Literal(expr))
            }
        }
    }
    
    // Parse lambda expressions: lambda(x, y) => { statements... }
    fn parse_lambda(&mut self) -> Result<Expr, String> {
        // Expect parentheses with parameters
        self.expect(Token::LeftParen)?;
        
        let mut params = Vec::new();
        
        // Parse parameters if any
        if !self.check(&Token::RightParen) {
            loop {
                if let Token::Identifier(param) = self.advance() {
                    params.push(param);
                } else {
                    return Err("Expected parameter name".to_string());
                }
                
                if self.match_token(&[Token::Comma]) {
                    continue;
                } else {
                    break;
                }
            }
        }
        
        self.expect(Token::RightParen)?;
        self.expect(Token::FatArrow)?; // =>
        
        // Parse body - either single expression or block
        let body = if self.check(&Token::LeftBrace) {
            self.advance(); // consume {
            let mut statements = Vec::new();
            
            while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
                statements.push(self.parse_statement()?);
            }
            
            self.expect(Token::RightBrace)?;
            statements
        } else {
            // Single expression - wrap in return statement
            let expr = self.parse_expression()?;
            vec![Statement::Return(Some(expr))]
        };
        
        Ok(Expr::Lambda { params, body })
    }
    
    // Parse import statements: import "module.path" or import "module.path" { item1, item2 }
    fn parse_import(&mut self) -> Result<Declaration, String> {
        self.expect(Token::Import)?;
        
        // Parse module path (string literal)
        let module_path = if let Token::String(path) = self.advance() {
            path
        } else {
            return Err("Expected module path string after import".to_string());
        };
        
        // Check for selective imports: import "module" { item1, item2 }
        let items = if self.check(&Token::LeftBrace) {
            self.advance(); // consume {
            
            let mut imported_items = Vec::new();
            while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
                if let Token::Identifier(item) = self.advance() {
                    imported_items.push(item);
                } else {
                    return Err("Expected identifier in import list".to_string());
                }
                
                if self.match_token(&[Token::Comma]) {
                    continue;
                } else {
                    break;
                }
            }
            
            self.expect(Token::RightBrace)?;
            Some(imported_items)
        } else {
            None // Wildcard import
        };
        
        // Optional semicolon
        self.match_token(&[Token::Semicolon]);
        
        Ok(Declaration::Import { module_path, items })
    }
    
    // Parse module definitions: module ModuleName { ... }
    fn parse_module(&mut self) -> Result<Declaration, String> {
        self.expect(Token::Module)?;
        
        // Parse module name
        let name = if let Token::Identifier(n) = self.advance() {
            n
        } else {
            return Err("Expected module name".to_string());
        };
        
        self.expect(Token::LeftBrace)?;
        
        let mut body = Vec::new();
        let mut exports = Vec::new();
        
        // Parse module body
        while !self.check(&Token::RightBrace) && !self.check(&Token::Eof) {
            // Check for export declarations
            if self.check(&Token::Export) {
                self.advance(); // consume export
                
                // Parse what's being exported (function, struct, etc.)
                let exported_decl = self.parse_declaration()?;
                
                // Extract the name for the exports list
                match &exported_decl {
                    Declaration::Function { name, .. } => exports.push(name.clone()),
                    Declaration::Struct { name, .. } => exports.push(name.clone()),
                    Declaration::Enum { name, .. } => exports.push(name.clone()),
                    _ => {} // Other declarations don't add to exports
                }
                
                body.push(exported_decl);
            } else {
                // Regular (private) declaration
                body.push(self.parse_declaration()?);
            }
        }
        
        self.expect(Token::RightBrace)?;
        
        Ok(Declaration::Module { name, exports, body })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_organism() {
        let input = "organism HelloWorld { }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        let ast = parser.parse().unwrap();
        
        assert_eq!(ast.len(), 1);
        match &ast[0] {
            Declaration::Organism { name, body } => {
                assert_eq!(name, "HelloWorld");
                assert_eq!(body.len(), 0);
            }
            _ => panic!("Expected organism declaration"),
        }
    }
    
    #[test]
    fn test_parse_function() {
        let input = "organism Test { fn metabolize(glucose) -> ATP { return glucose * 2 } }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        let ast = parser.parse().unwrap();
        
        match &ast[0] {
            Declaration::Organism { body, .. } => {
                assert_eq!(body.len(), 1);
                match &body[0] {
                    Declaration::Function { name, params, return_type, .. } => {
                        assert_eq!(name, "metabolize");
                        assert_eq!(params, &vec!["glucose".to_string()]);
                        assert_eq!(return_type, &Some("ATP".to_string()));
                    }
                    _ => panic!("Expected function declaration"),
                }
            }
            _ => panic!("Expected organism declaration"),
        }
    }
    
    #[test]
    fn test_parse_pipe() {
        let input = "organism Test { fn process() { data |> transform |> output } }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        let ast = parser.parse().unwrap();
        
        // Verify pipe expression was parsed
        match &ast[0] {
            Declaration::Organism { body, .. } => {
                match &body[0] {
                    Declaration::Function { body, .. } => {
                        match &body[0] {
                            Statement::Expression(Expr::Pipe { stages, .. }) => {
                                assert_eq!(stages.len(), 2);
                            }
                            _ => panic!("Expected pipe expression"),
                        }
                    }
                    _ => panic!("Expected function"),
                }
            }
            _ => panic!("Expected organism"),
        }
    }
}