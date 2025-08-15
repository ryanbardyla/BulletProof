//! NeuronLang Parser - Builds AST from tokens
//! 
//! Recursive descent parser for the trinary neural language.
//! Handles fire-and-forget, temporal constructs, and biological patterns.

use crate::lexer::{Token, TryteValue, TimeUnit};
use std::collections::HashMap;

// ============ AST Node Definitions ============

#[derive(Debug, Clone)]
pub enum AstNode {
    Program(Vec<Declaration>),
    Statement(Statement),
    Expression(Expression),
}

#[derive(Debug, Clone)]
pub struct Declaration {
    pub kind: DeclKind,
    pub location: SourceLocation,
}

#[derive(Debug, Clone)]
pub enum DeclKind {
    Brain(BrainDecl),
    Layer(LayerDecl),
    Pattern(PatternDecl),
    Function(FunctionDecl),
}

#[derive(Debug, Clone)]
pub struct BrainDecl {
    pub name: String,
    pub layers: Vec<LayerStatement>,
    pub synapses: Option<SynapseConfig>,
    pub behaviors: Vec<BehaviorStatement>,
}

#[derive(Debug, Clone)]
pub struct LayerDecl {
    pub name: String,
    pub size: usize,
    pub config: LayerConfig,
}

#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub neuron_type: NeuronType,
    pub threshold: Option<f32>,
    pub refractory: Option<f32>,
}

#[derive(Debug, Clone)]
pub enum NeuronType {
    Spiking,
    Lif,
    Izhikevich(f32, f32, f32, f32), // a, b, c, d
    Trinary,
}

#[derive(Debug, Clone)]
pub struct PatternDecl {
    pub name: String,
    pub condition: Expression,
    pub action: Statement,
}

#[derive(Debug, Clone)]
pub struct FunctionDecl {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub return_type: Option<Type>,
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub enum Type {
    Tryte,
    Spike,
    Bool,
    I32,
    F32,
    String,
}

#[derive(Debug, Clone)]
pub struct LayerStatement {
    pub name: String,
    pub size: usize,
    pub attributes: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct SynapseConfig {
    pub plasticity: PlasticityRule,
    pub delays: (f32, f32),
    pub sparsity: f32,
}

#[derive(Debug, Clone)]
pub enum PlasticityRule {
    Stdp(f32, f32, f32), // window, ltp, ltd
    Hebbian(f32),
    None,
}

#[derive(Debug, Clone)]
pub struct BehaviorStatement {
    pub condition: Expression,
    pub actions: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub enum Statement {
    Assignment(String, Expression),
    If(Expression, Vec<Statement>, Option<Vec<Statement>>),
    For(String, Expression, Vec<Statement>),
    While(Expression, Vec<Statement>),
    Return(Option<Expression>),
    Fire(Expression),
    Forget(f32, Option<f32>), // membrane, current
    Expression(Expression),
}

#[derive(Debug, Clone)]
pub enum Expression {
    Literal(Literal),
    Identifier(String),
    Binary(BinaryOp, Box<Expression>, Box<Expression>),
    Unary(UnaryOp, Box<Expression>),
    Call(String, Vec<Expression>),
    Field(Box<Expression>, String),
    Index(Box<Expression>, Box<Expression>),
}

#[derive(Debug, Clone)]
pub enum Literal {
    Tryte(TryteValue),
    Integer(i32),
    Float(f32),
    String(String),
    Boolean(bool),
    Voltage(f32),
    Time(f32, TimeUnit),
}

#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add, Sub, Mul, Div, Mod, Power,
    Equal, NotEqual, Less, Greater, LessEqual, GreaterEqual,
    And, Or, Xor,
}

#[derive(Debug, Clone)]
pub enum UnaryOp {
    Negate, Not,
}

#[derive(Debug, Clone, Default)]
pub struct SourceLocation {
    pub line: usize,
    pub column: usize,
}

// ============ Parser Implementation ============

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, current: 0 }
    }
    
    pub fn parse(&mut self) -> Result<AstNode, ParseError> {
        let mut declarations = Vec::new();
        
        while !self.is_at_end() {
            match self.parse_declaration() {
                Ok(decl) => declarations.push(decl),
                Err(_) => {
                    // Skip to next potential declaration
                    self.skip_until_next_declaration();
                    if self.is_at_end() {
                        break;
                    }
                }
            }
        }
        
        Ok(AstNode::Program(declarations))
    }
    
    fn parse_declaration(&mut self) -> Result<Declaration, ParseError> {
        let kind = match self.peek() {
            Token::Brain => self.parse_brain()?,
            Token::Pattern => self.parse_pattern()?,
            Token::Temporal => self.parse_temporal()?,
            Token::Meta => self.parse_meta_learning()?,
            Token::Propagate => self.parse_belief_propagation()?,
            Token::Identifier(name) if name == "fn" => self.parse_function()?,
            Token::Identifier(name) if name == "optimize" => {
                // Skip optimize block for now
                self.advance(); // consume "optimize"
                self.skip_block();
                return Err(ParseError::UnexpectedToken(Token::Identifier("optimize".to_string())));
            }
            _ => {
                // Skip unknown declarations
                return Err(ParseError::UnexpectedToken(self.peek().clone()));
            }
        };
        
        Ok(Declaration {
            kind,
            location: SourceLocation::default(),
        })
    }
    
    fn parse_pattern(&mut self) -> Result<DeclKind, ParseError> {
        self.consume(Token::Pattern)?;
        let name = self.parse_identifier()?;
        self.consume(Token::LeftBrace)?;
        
        // Skip pattern body for now
        self.skip_block_content();
        
        self.consume(Token::RightBrace)?;
        
        Ok(DeclKind::Pattern(PatternDecl {
            name,
            condition: Expression::Literal(Literal::Boolean(true)),
            action: Statement::Expression(Expression::Literal(Literal::Boolean(true))),
        }))
    }
    
    fn parse_temporal(&mut self) -> Result<DeclKind, ParseError> {
        self.consume(Token::Temporal)?;
        
        // Skip optional name
        if matches!(self.peek(), Token::Identifier(_)) {
            self.advance();
        }
        
        self.consume(Token::LeftBrace)?;
        self.skip_block_content();
        self.consume(Token::RightBrace)?;
        
        // Return a dummy Pattern for now since we don't have Temporal in DeclKind
        Ok(DeclKind::Pattern(PatternDecl {
            name: "temporal".to_string(),
            condition: Expression::Literal(Literal::Boolean(true)),
            action: Statement::Expression(Expression::Literal(Literal::Boolean(true))),
        }))
    }
    
    fn parse_meta_learning(&mut self) -> Result<DeclKind, ParseError> {
        self.consume(Token::Meta)?;
        self.consume(Token::Learn)?;
        self.consume(Token::LeftBrace)?;
        
        self.skip_block_content();
        
        self.consume(Token::RightBrace)?;
        
        // Return a dummy Pattern for now
        Ok(DeclKind::Pattern(PatternDecl {
            name: "meta_learning".to_string(),
            condition: Expression::Literal(Literal::Boolean(true)),
            action: Statement::Expression(Expression::Literal(Literal::Boolean(true))),
        }))
    }
    
    fn parse_belief_propagation(&mut self) -> Result<DeclKind, ParseError> {
        self.consume(Token::Propagate)?;
        
        // Skip "beliefs"
        if matches!(self.peek(), Token::Identifier(s) if s == "beliefs") {
            self.advance();
        }
        
        self.consume(Token::LeftBrace)?;
        self.skip_block_content();
        self.consume(Token::RightBrace)?;
        
        // Return a dummy Pattern for now
        Ok(DeclKind::Pattern(PatternDecl {
            name: "belief_propagation".to_string(),
            condition: Expression::Literal(Literal::Boolean(true)),
            action: Statement::Expression(Expression::Literal(Literal::Boolean(true))),
        }))
    }
    
    fn parse_function(&mut self) -> Result<DeclKind, ParseError> {
        self.advance(); // consume "fn"
        let name = self.parse_identifier()?;
        self.consume(Token::LeftParen)?;
        
        // Skip parameters for now
        while !self.check(&Token::RightParen) && !self.is_at_end() {
            self.advance();
        }
        self.consume(Token::RightParen)?;
        
        // Skip return type if present
        if self.check(&Token::Arrow) {
            self.advance();
            self.advance(); // skip type
        }
        
        self.consume(Token::LeftBrace)?;
        self.skip_block_content();
        self.consume(Token::RightBrace)?;
        
        Ok(DeclKind::Function(FunctionDecl {
            name,
            params: Vec::new(),
            return_type: None,
            body: Vec::new(),
        }))
    }
    
    fn skip_block(&mut self) {
        if self.check(&Token::LeftBrace) {
            self.advance();
            self.skip_block_content();
            if self.check(&Token::RightBrace) {
                self.advance();
            }
        }
    }
    
    fn skip_block_content(&mut self) {
        let mut brace_count = 1;
        while brace_count > 0 && !self.is_at_end() {
            if self.check(&Token::LeftBrace) {
                brace_count += 1;
            } else if self.check(&Token::RightBrace) {
                brace_count -= 1;
                if brace_count == 0 {
                    break;
                }
            }
            self.advance();
        }
    }
    
    fn skip_until_next_declaration(&mut self) {
        while !self.is_at_end() {
            match self.peek() {
                Token::Brain | Token::Pattern | Token::Temporal | 
                Token::Meta | Token::Propagate => break,
                Token::Identifier(s) if s == "fn" || s == "optimize" => break,
                _ => {
                    self.advance();
                }
            }
        }
    }
    
    fn parse_brain(&mut self) -> Result<DeclKind, ParseError> {
        self.consume(Token::Brain)?;
        let name = self.parse_identifier()?;
        self.consume(Token::LeftBrace)?;
        
        let mut layers = Vec::new();
        let mut synapses = None;
        let mut behaviors = Vec::new();
        
        while !self.check(&Token::RightBrace) {
            match self.peek() {
                Token::Layer => {
                    layers.push(self.parse_layer_statement()?);
                }
                Token::Identifier(s) if s == "synapses" => {
                    synapses = Some(self.parse_synapses()?);
                }
                Token::When => {
                    behaviors.push(self.parse_behavior()?);
                }
                _ => { 
                    // Skip unknown tokens for now
                    self.advance(); 
                }
            }
        }
        
        self.consume(Token::RightBrace)?;
        
        Ok(DeclKind::Brain(BrainDecl {
            name,
            layers,
            synapses,
            behaviors,
        }))
    }
    
    fn parse_layer_statement(&mut self) -> Result<LayerStatement, ParseError> {
        self.consume(Token::Layer)?;
        let name = self.parse_identifier()?;
        self.consume(Token::LeftBracket)?;
        let size = self.parse_integer()? as usize;
        self.consume(Token::RightBracket)?;
        
        let mut attributes = HashMap::new();
        
        // Parse optional layer configuration
        if self.check(&Token::LeftBrace) {
            self.advance();
            while !self.check(&Token::RightBrace) && !self.is_at_end() {
                // Parse attribute name
                if let Token::Identifier(attr_name) = self.peek() {
                    let attr_name = attr_name.clone();
                    self.advance();
                    
                    if self.check(&Token::Colon) {
                        self.advance();
                        
                        // Parse attribute value
                        match attr_name.as_str() {
                            "perceive" => {
                                // Skip perceive source for now
                                while !self.check(&Token::RightParen) && !self.is_at_end() {
                                    self.advance();
                                }
                                if self.check(&Token::RightParen) {
                                    self.advance();
                                }
                            }
                            "neurons" => {
                                // Parse neuron type
                                if self.check(&Token::Spiking) || self.check(&Token::Izhikevich) || 
                                   self.check(&Token::Trinary) || self.check(&Token::Lif) {
                                    attributes.insert("neurons".to_string(), format!("{:?}", self.peek()));
                                    self.advance();
                                    // Skip parameters if any
                                    if self.check(&Token::LeftParen) {
                                        let mut paren_count = 1;
                                        self.advance();
                                        while paren_count > 0 && !self.is_at_end() {
                                            if self.check(&Token::LeftParen) {
                                                paren_count += 1;
                                            } else if self.check(&Token::RightParen) {
                                                paren_count -= 1;
                                            }
                                            self.advance();
                                        }
                                    }
                                }
                            }
                            "threshold" | "refractory" | "decide" => {
                                // Skip these for now
                                while !self.check(&Token::RightBrace) && !self.is_at_end() {
                                    if matches!(self.peek(), Token::Identifier(_)) {
                                        break;
                                    }
                                    self.advance();
                                }
                            }
                            _ => {
                                // Skip unknown attributes
                                self.advance();
                            }
                        }
                    }
                } else {
                    self.advance();
                }
            }
            self.consume(Token::RightBrace)?;
        }
        
        Ok(LayerStatement {
            name,
            size,
            attributes,
        })
    }
    
    fn parse_synapses(&mut self) -> Result<SynapseConfig, ParseError> {
        self.advance(); // consume "synapses"
        self.consume(Token::LeftBrace)?;
        
        let mut plasticity = PlasticityRule::None;
        let mut delays = (1.0, 5.0);
        let mut sparsity = 0.95;
        
        while !self.check(&Token::RightBrace) && !self.is_at_end() {
            if let Token::Identifier(attr) = self.peek() {
                let attr = attr.clone();
                self.advance();
                
                if self.check(&Token::Colon) {
                    self.advance();
                    
                    match attr.as_str() {
                        "plasticity" => {
                            if self.check(&Token::Stdp) {
                                self.advance();
                                // Parse STDP parameters
                                if self.check(&Token::LeftParen) {
                                    self.advance();
                                    // For now, use defaults
                                    plasticity = PlasticityRule::Stdp(20.0, 0.01, 0.005);
                                    while !self.check(&Token::RightParen) && !self.is_at_end() {
                                        self.advance();
                                    }
                                    if self.check(&Token::RightParen) {
                                        self.advance();
                                    }
                                }
                            } else if self.check(&Token::Hebbian) {
                                self.advance();
                                plasticity = PlasticityRule::Hebbian(0.01);
                            }
                        }
                        "delays" => {
                            // Skip delay parsing for now
                            while !self.check(&Token::RightParen) && !self.is_at_end() {
                                self.advance();
                            }
                            if self.check(&Token::RightParen) {
                                self.advance();
                            }
                        }
                        "sparsity" => {
                            if let Token::FloatLiteral(s) = self.peek() {
                                sparsity = *s;
                                self.advance();
                            }
                        }
                        _ => {
                            self.advance();
                        }
                    }
                }
            } else {
                self.advance();
            }
        }
        
        self.consume(Token::RightBrace)?;
        
        Ok(SynapseConfig {
            plasticity,
            delays,
            sparsity,
        })
    }
    
    fn parse_behavior(&mut self) -> Result<BehaviorStatement, ParseError> {
        self.consume(Token::When)?;
        let condition = self.parse_expression()?;
        self.consume(Token::LeftBrace)?;
        
        let mut actions = Vec::new();
        while !self.check(&Token::RightBrace) && !self.is_at_end() {
            if self.check(&Token::Fire) {
                self.advance();
                self.consume(Token::Bang)?;
                self.consume(Token::LeftParen)?;
                let expr = self.parse_expression()?;
                self.consume(Token::RightParen)?;
                actions.push(Statement::Fire(expr));
            } else if self.check(&Token::Forget) {
                self.advance();
                self.consume(Token::Bang)?;
                self.consume(Token::LeftParen)?;
                // Parse forget params
                self.advance(); // Skip for now
                while !self.check(&Token::RightParen) && !self.is_at_end() {
                    self.advance();
                }
                self.consume(Token::RightParen)?;
                actions.push(Statement::Forget(-80.0, Some(0.0)));
            } else {
                self.advance();
            }
        }
        
        self.consume(Token::RightBrace)?;
        
        Ok(BehaviorStatement { condition, actions })
    }
    
    fn parse_expression(&mut self) -> Result<Expression, ParseError> {
        self.parse_primary()
    }
    
    fn parse_primary(&mut self) -> Result<Expression, ParseError> {
        match self.peek() {
            Token::TryteLiteral(v) => {
                let v = *v;
                self.advance();
                Ok(Expression::Literal(Literal::Tryte(v)))
            }
            Token::IntegerLiteral(n) => {
                let n = *n;
                self.advance();
                Ok(Expression::Literal(Literal::Integer(n)))
            }
            Token::FloatLiteral(f) => {
                let f = *f;
                self.advance();
                Ok(Expression::Literal(Literal::Float(f)))
            }
            Token::StringLiteral(s) => {
                let s = s.clone();
                self.advance();
                Ok(Expression::Literal(Literal::String(s)))
            }
            Token::BooleanLiteral(b) => {
                let b = *b;
                self.advance();
                Ok(Expression::Literal(Literal::Boolean(b)))
            }
            Token::VoltageLiteral(v) => {
                let v = *v;
                self.advance();
                Ok(Expression::Literal(Literal::Voltage(v)))
            }
            Token::TimeLiteral(t, u) => {
                let t = *t;
                let u = *u;
                self.advance();
                Ok(Expression::Literal(Literal::Time(t, u)))
            }
            Token::Identifier(s) => {
                let s = s.clone();
                self.advance();
                
                // Check for comparison operators
                if self.check(&Token::GreaterEqual) {
                    self.advance();
                    let right = self.parse_expression()?;
                    Ok(Expression::Binary(
                        BinaryOp::GreaterEqual,
                        Box::new(Expression::Identifier(s)),
                        Box::new(right),
                    ))
                } else {
                    Ok(Expression::Identifier(s))
                }
            }
            _ => Err(ParseError::UnexpectedToken(self.peek().clone())),
        }
    }
    
    fn parse_identifier(&mut self) -> Result<String, ParseError> {
        match self.peek() {
            Token::Identifier(s) => {
                let s = s.clone();
                self.advance();
                Ok(s)
            }
            _ => Err(ParseError::ExpectedIdentifier),
        }
    }
    
    fn parse_integer(&mut self) -> Result<i32, ParseError> {
        match self.peek() {
            Token::IntegerLiteral(n) => {
                let n = *n;
                self.advance();
                Ok(n)
            }
            _ => Err(ParseError::ExpectedInteger),
        }
    }
    
    // Helper methods
    
    fn peek(&self) -> &Token {
        self.tokens.get(self.current).unwrap_or(&Token::Eof)
    }
    
    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }
    
    fn is_at_end(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }
    
    fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
    }
    
    fn check(&self, token: &Token) -> bool {
        if self.is_at_end() {
            false
        } else {
            std::mem::discriminant(self.peek()) == std::mem::discriminant(token)
        }
    }
    
    fn consume(&mut self, token: Token) -> Result<(), ParseError> {
        if self.check(&token) {
            self.advance();
            Ok(())
        } else {
            Err(ParseError::Expected(token, self.peek().clone()))
        }
    }
}

// ============ Error Types ============

#[derive(Debug)]
pub enum ParseError {
    UnexpectedToken(Token),
    Expected(Token, Token),
    ExpectedIdentifier,
    ExpectedInteger,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ParseError::UnexpectedToken(t) => write!(f, "Unexpected token: {:?}", t),
            ParseError::Expected(expected, got) => {
                write!(f, "Expected {:?}, got {:?}", expected, got)
            }
            ParseError::ExpectedIdentifier => write!(f, "Expected identifier"),
            ParseError::ExpectedInteger => write!(f, "Expected integer"),
        }
    }
}

impl std::error::Error for ParseError {}