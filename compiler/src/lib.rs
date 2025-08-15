// NeuronLang Compiler - 100% Our Code!
// Direct compilation: .nl → Rust → Native binary

pub mod lexer;
pub mod parser;
pub mod codegen;
pub mod graph_storage;
pub mod ewc_meta_learning;
pub mod stdlib;
pub mod package;
pub mod register_allocator;
pub mod peephole_optimizer;
pub mod source_location;
pub mod error_reporting;
pub mod stack_trace;
pub mod runtime_panic;
pub mod debug_symbols;
pub mod breakpoint;

// Minimal compiler modules (used by package manager)
pub mod minimal_lexer;
pub mod minimal_parser;
pub mod minimal_codegen;

// Neural execution engine with conscious field architecture
pub mod neural_engine;