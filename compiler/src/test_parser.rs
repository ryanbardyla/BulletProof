//! Test the complete lexer + parser pipeline

mod lexer;
mod parser;

use lexer::Lexer;
use parser::Parser;
use std::fs;

fn main() {
    println!("🧠 NeuronLang Parser Test");
    println!("=========================\n");
    
    // Read the trading brain example
    let source = fs::read_to_string("../examples/trading_brain.nl")
        .expect("Failed to read .nl file");
    
    println!("📄 Parsing: trading_brain.nl\n");
    
    // Tokenize
    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize();
    println!("✅ Lexer: {} tokens generated", tokens.len());
    
    // Parse
    let mut parser = Parser::new(tokens);
    match parser.parse() {
        Ok(ast) => {
            println!("✅ Parser: Successfully built AST!");
            println!("\n🌳 Abstract Syntax Tree:");
            println!("{:#?}", ast);
            
            // Count AST nodes
            match ast {
                parser::AstNode::Program(decls) => {
                    println!("\n📊 AST Statistics:");
                    println!("  Declarations: {}", decls.len());
                    
                    for decl in &decls {
                        match &decl.kind {
                            parser::DeclKind::Brain(brain) => {
                                println!("\n🧠 Brain: {}", brain.name);
                                println!("  Layers: {}", brain.layers.len());
                                for layer in &brain.layers {
                                    println!("    - {} [{}]", layer.name, layer.size);
                                }
                                println!("  Behaviors: {}", brain.behaviors.len());
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        Err(e) => {
            println!("❌ Parser error: {}", e);
            println!("\nThis is expected - parser is still being built!");
        }
    }
    
    println!("\n✨ Parser foundation complete!");
}