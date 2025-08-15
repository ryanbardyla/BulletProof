//! Test the lexer on real .nl files

mod lexer;
use lexer::{Lexer, Token};
use std::fs;

fn main() {
    println!("🧠 NeuronLang Lexer Test");
    println!("========================\n");
    
    // Read the trading brain example
    let source = fs::read_to_string("../examples/trading_brain.nl")
        .expect("Failed to read .nl file");
    
    println!("📄 Source file: trading_brain.nl");
    println!("📏 Size: {} bytes\n", source.len());
    
    // Tokenize
    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize();
    
    println!("🔤 Tokenization Results:");
    println!("  Total tokens: {}", tokens.len());
    
    // Count token types
    let mut tryte_count = 0;
    let mut keyword_count = 0;
    let mut identifier_count = 0;
    let mut literal_count = 0;
    
    for token in &tokens {
        match token {
            Token::TryteLiteral(_) => tryte_count += 1,
            Token::Brain | Token::Layer | Token::Fire | Token::Forget |
            Token::Spiking | Token::Izhikevich | Token::Trinary |
            Token::Stdp | Token::Ewc | Token::Meta | Token::Learn |
            Token::Propagate | Token::Consolidate | Token::Synthesize |
            Token::Compress | Token::Pattern | Token::Temporal |
            Token::Trade | Token::HyperLiquid | Token::Redis => keyword_count += 1,
            Token::Identifier(_) => identifier_count += 1,
            Token::IntegerLiteral(_) | Token::FloatLiteral(_) | 
            Token::StringLiteral(_) | Token::BooleanLiteral(_) |
            Token::VoltageLiteral(_) | Token::TimeLiteral(_, _) => literal_count += 1,
            _ => {}
        }
    }
    
    println!("\n📊 Token Statistics:");
    println!("  Tryte literals: {}", tryte_count);
    println!("  Keywords: {}", keyword_count);
    println!("  Identifiers: {}", identifier_count);
    println!("  Other literals: {}", literal_count);
    
    // Show first 50 tokens
    println!("\n🔍 First 50 tokens:");
    for (i, token) in tokens.iter().take(50).enumerate() {
        println!("  {:3}: {}", i, token);
    }
    
    // Find and display special constructs
    println!("\n✨ Special Constructs Found:");
    
    // Find fire! and forget!
    for i in 0..tokens.len().saturating_sub(2) {
        if matches!(tokens[i], Token::Fire) && matches!(tokens[i+1], Token::Bang) {
            println!("  ✓ fire! construct at token {}", i);
        }
        if matches!(tokens[i], Token::Forget) && matches!(tokens[i+1], Token::Bang) {
            println!("  ✓ forget! construct at token {}", i);
        }
    }
    
    // Find temporal values
    for token in &tokens {
        if let Token::VoltageLiteral(v) = token {
            println!("  ✓ Voltage literal: {}mV", v);
        }
        if let Token::TimeLiteral(t, u) = token {
            println!("  ✓ Time literal: {}{:?}", t, u);
        }
    }
    
    println!("\n✅ Lexer test complete!");
}