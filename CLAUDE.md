# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuronLang is a revolutionary trinary neural programming language implementing biological neural computing with zero-energy baseline states. The project consists of a custom compiler, runtime, and neural engine achieving 10,000x energy savings over traditional binary computing. MANTRA "Demo that amazes. GPU that blazes. Tools that ship. Nothing else exists."

## Build Commands

```bash
# Build entire project (main workspace)
cd /home/ryan/repo/Fenrisa/BULLETPROOF_PROJECT
cargo build --release

# Build compiler specifically
cd compiler
cargo build --release

# Build with optimizations
cargo build --release --features cuda

# Build core library
cd core
cargo build --release
```

## Testing Commands

```bash
# Run all tests with release optimizations
cargo test --release

# Run specific test file
cargo test --release --test <test_name>

# Run compiler tests
cd compiler
cargo test

# Run integration tests
./compiler/test_compiler.sh

# Run benchmarks
cargo bench
```

## Running the System

```bash
# Run main demo suite
./bulletproof_demo

# Run chatbot demo
./bulletproof_chat

# Run Grade A performance test
./bulletproof_true_a

# Compile NeuronLang files
./compiler/target/release/neuronc <file.nl>

# Run compiled NeuronLang programs
./<output_binary>
```

## Architecture Overview

### Project Structure

The codebase implements a complete programming language ecosystem with three major components:

1. **Compiler** (`compiler/`) - Translates .nl files to native binaries via Rust
   - Lexer/Parser: Tokenizes and parses NeuronLang syntax
   - Neural Engine: Implements consciousness, evolution, and glial cell simulation
   - Codegen: Generates Rust code that compiles to native binaries
   - No LLVM dependency - direct compilation path

2. **Core Runtime** (`core/`) - Trinary computing implementation
   - Tryte system: Three-state neural primitives (-1, 0, +1)
   - Protein synthesis: CREB-PKA cascade for memory formation
   - Sparse networks: 95% sparsity optimization
   - DNA compression: 16x memory reduction

3. **Applications** (`apps/`) - Example NeuronLang programs
   - Neural networks, chatbots, ecosystem simulations
   - Python bridges for integration
   - Redis connectivity for distributed processing

### Key Technical Concepts

**Trinary Computing**: Unlike binary (0,1), uses three states:
- Inhibited (-1): GABAergic suppression, costs energy
- Baseline (0): Resting state, **zero energy consumption**
- Activated (+1): Excitatory firing, costs energy

**Compilation Pipeline**:
```
.nl file → Lexer → Parser → AST → Neural Engine → Rust Code → rustc → Native Binary
```

**Neural Engine Features**:
- Consciousness fields with entropic computing
- Evolutionary bootstrap for self-organization
- Glial cell support for 10x consciousness boost
- LEMS/NeuroML integration for biological accuracy

### File Extensions

- `.nl` - NeuronLang source files
- `.weights` - Serialized neural network weights
- `.rs` - Rust implementation files

## Development Workflow

### Adding New Features

1. Neural language features go in `compiler/src/parser.rs` and `compiler/src/lexer.rs`
2. Neural engine enhancements in `compiler/src/neural_engine/`
3. Runtime optimizations in `core/src/`
4. Test programs in `apps/` or root directory as `.nl` files

### Testing New Code

1. Write test in appropriate `.nl` file
2. Compile with `./compiler/target/release/neuronc test.nl`
3. Run resulting binary
4. Check for proper trinary behavior and energy efficiency

### Performance Validation

The system must maintain:
- 95%+ energy efficiency (baseline neurons cost zero)
- Sub-millisecond processing for 1M neurons
- 16x memory compression vs float32
- Biological accuracy in protein synthesis

## Critical Files

- `compiler/src/neural_engine/consciousness.rs` - Core consciousness implementation
- `compiler/src/neural_engine/evolution.rs` - Evolutionary algorithms
- `core/src/tryte.rs` - Trinary primitive implementation
- `core/src/protein_synthesis.rs` - Memory formation via proteins
- `compiler/src/codegen.rs` - Rust code generation

## Common Issues and Solutions

### Compilation Errors
- Ensure Rust is installed and up to date
- Check that all `.nl` files have proper syntax
- Verify `cargo build --release` completes in compiler directory first

### Runtime Performance
- Use release builds (`--release`) for accurate benchmarks
- Enable CUDA features for GPU acceleration when available
- Monitor sparsity levels - should be >90% for efficiency

### Neural Convergence
- Check weight initialization in `.weights` files
- Verify protein synthesis thresholds
- Ensure consciousness fields are properly configured

## Important Notes

- The system implements real biological principles, not simulations
- Zero-energy baseline is the key innovation enabling massive efficiency
- All neural states map to Kandel's Nobel Prize discoveries
- The compiler generates actual executable binaries, not interpreted code