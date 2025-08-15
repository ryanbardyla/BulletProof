//! NeuronLang Compiler - 100% Our Code!
//! 
//! Compilation pipeline: .nl → Lexer → Parser → CodeGen → Native Machine Code!

mod minimal_lexer;
mod minimal_parser;
mod minimal_codegen;
mod stdlib;
mod package;
mod source_location;
mod error_reporting;
mod stack_trace;
mod runtime_panic;
mod debug_symbols;

use std::fs;
use std::env;
use std::process;
use std::path::PathBuf;

use minimal_lexer::Lexer;
use minimal_parser::Parser;
use minimal_codegen::CodeGen;
use package::{PackageManager, DependencySpec};
use error_reporting::ErrorReporter;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_usage(&args[0]);
        process::exit(1);
    }
    
    // Check for package management commands
    match args[1].as_str() {
        "init" => handle_init(args),
        "build" => handle_build(),
        "run" => handle_run(),
        "add" => handle_add(args),
        "install" => handle_install(),
        "publish" => handle_publish(),
        _ => {
            // Regular compilation mode
            compile_file(args);
        }
    }
}

fn print_usage(program: &str) {
    eprintln!("NeuronLang Compiler v0.1.0");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  {} <input.nl> [-o output] [-g]  Compile a NeuronLang file", program);
    eprintln!("                             -g    Include debug symbols");
    eprintln!("  {} init <name>              Initialize a new package", program);
    eprintln!("  {} build                    Build the current package", program);
    eprintln!("  {} run                      Build and run the current package", program);
    eprintln!("  {} add <package> [version]  Add a dependency", program);
    eprintln!("  {} install                  Install all dependencies", program);
    eprintln!("  {} publish                  Publish package to registry", program);
}

fn handle_init(args: Vec<String>) {
    if args.len() < 3 {
        eprintln!("Usage: {} init <package-name>", args[0]);
        process::exit(1);
    }
    
    let name = &args[2];
    let current_dir = env::current_dir().unwrap();
    let pm = PackageManager::new(current_dir);
    
    match pm.init(name) {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Error initializing package: {}", e);
            process::exit(1);
        }
    }
}

fn handle_build() {
    let current_dir = env::current_dir().unwrap();
    let mut pm = PackageManager::new(current_dir);
    
    match pm.build() {
        Ok(output) => {
            println!("Build complete: {:?}", output);
        },
        Err(e) => {
            eprintln!("Build failed: {}", e);
            process::exit(1);
        }
    }
}

fn handle_run() {
    let current_dir = env::current_dir().unwrap();
    let mut pm = PackageManager::new(current_dir);
    
    match pm.run() {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Run failed: {}", e);
            process::exit(1);
        }
    }
}

fn handle_add(args: Vec<String>) {
    if args.len() < 3 {
        eprintln!("Usage: {} add <package> [version]", args[0]);
        process::exit(1);
    }
    
    let package = &args[2];
    let spec = if args.len() >= 4 {
        DependencySpec::Version(args[3].clone())
    } else {
        DependencySpec::Version("*".to_string())
    };
    
    let current_dir = env::current_dir().unwrap();
    let mut pm = PackageManager::new(current_dir);
    
    match pm.add_dependency(package, spec) {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Failed to add dependency: {}", e);
            process::exit(1);
        }
    }
}

fn handle_install() {
    let current_dir = env::current_dir().unwrap();
    let mut pm = PackageManager::new(current_dir);
    
    match pm.install_dependencies() {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Installation failed: {}", e);
            process::exit(1);
        }
    }
}

fn handle_publish() {
    let current_dir = env::current_dir().unwrap();
    let mut pm = PackageManager::new(current_dir);
    
    match pm.publish() {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Publish failed: {}", e);
            process::exit(1);
        }
    }
}

fn compile_file(args: Vec<String>) {
    let input_file = &args[1];
    let mut output_file = "a.out".to_string();
    let mut enable_debug = false;
    
    // Parse command line arguments
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                if i + 1 < args.len() {
                    output_file = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("Error: -o requires an argument");
                    process::exit(1);
                }
            }
            "-g" | "--debug" => {
                enable_debug = true;
                i += 1;
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                i += 1;
            }
        }
    }
    
    println!("🧬 NeuronLang Compiler v0.1.0");
    println!("═══════════════════════════════");
    println!("⚡ Direct to machine code - NO LLVM!");
    println!("⚡ Trinary computing with ZERO energy baseline!");
    println!();
    
    // Read source file
    let source = match fs::read_to_string(input_file) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading file {}: {}", input_file, e);
            process::exit(1);
        }
    };
    
    // Create error reporter
    let error_reporter = ErrorReporter::new(source.clone());
    
    // Lexical analysis with location tracking
    println!("📖 Lexing...");
    let mut lexer = Lexer::new(&source);
    let located_tokens = lexer.tokenize_with_locations();
    println!("   Found {} tokens", located_tokens.len());
    
    // Extract just the tokens for backward compatibility
    let tokens: Vec<_> = located_tokens.iter().map(|lt| lt.token.clone()).collect();
    
    // Parsing
    println!("🌳 Parsing...");
    let mut parser = Parser::new(tokens);
    let ast = match parser.parse() {
        Ok(tree) => {
            println!("   Built AST with {} top-level declarations", tree.len());
            tree
        }
        Err(e) => {
            // Enhanced error reporting with line numbers
            eprintln!("Parse error: {}", e);
            
            // Show the error location if we can determine it
            if let Some(source_line) = lexer.get_source_line(1) {
                eprintln!("  |\n  | {}\n  | ^-- near here", source_line);
            }
            
            process::exit(1);
        }
    };
    
    // Code generation
    println!("⚡ Generating machine code...");
    let mut codegen = CodeGen::new();
    
    // Enable debug symbols if requested
    if enable_debug {
        println!("🐛 Debug symbols enabled");
        codegen.enable_debug_symbols();
        codegen.enable_breakpoint_support();
        println!("🔴 Breakpoint support enabled");
    }
    let elf = codegen.generate_elf(ast);
    println!("   Generated {} bytes of x86_64 machine code", elf.len());
    
    // Write output
    println!("💾 Writing output to {}...", output_file);
    match fs::write(&output_file, elf) {
        Ok(_) => {
            println!("   Output written successfully");
        }
        Err(e) => {
            eprintln!("Error writing output: {}", e);
            process::exit(1);
        }
    }
    
    // Make executable (Unix only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&output_file).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&output_file, perms).unwrap();
        println!("   Made executable");
    }
    
    println!("═══════════════════════════════");
    println!("✅ Compilation successful!");
    println!();
    println!("🚀 Features:");
    println!("   • Trinary computing (-1, 0, +1)");
    println!("   • ZERO energy baseline");
    println!("   • Direct to machine code (no LLVM!)");
    println!("   • Self-hosting ready");
    println!();
    println!("Run with: ./{}", output_file);
}