# NeuronLang Compiler - Complete Requirements

## Current Status ✅
- [x] Basic types (integers, strings)
- [x] Functions
- [x] If/else statements
- [x] While loops
- [x] Express statements (print)
- [x] Basic operators (+, -, *, /, ==, !=, <, >, <=, >=)
- [x] Variables (let)
- [x] Direct x86_64 code generation
- [x] ELF binary creation
- [x] String handling (FIXED!)
- [x] Redis syscalls (basic)

## CRITICAL MISSING FEATURES ❌

### 1. Data Types
- [ ] **Floating point numbers** (CRITICAL for neural networks)
- [ ] **Arrays/Lists** (CRITICAL for matrix operations)
- [ ] **Structs/Records** (for complex data)
- [ ] **Booleans** (true/false)
- [ ] **Bytes/Binary data**

### 2. Operations
- [ ] **Float arithmetic** (add, sub, mul, div with floats)
- [ ] **Array indexing** (array[i])
- [ ] **Array operations** (push, pop, length)
- [ ] **Math functions** (sin, cos, sqrt, exp, log)
- [ ] **Bitwise operations** (for low-level work)

### 3. Control Flow
- [ ] **For loops** (for i = 0; i < 10; i++)
- [ ] **Break/Continue** in loops
- [ ] **Switch/Match** statements
- [ ] **Function returns with values**
- [ ] **Multiple return values**

### 4. System Integration
- [ ] **File I/O** (open, read, write files)
- [ ] **Network sockets** (TCP/UDP)
- [ ] **Memory management** (malloc/free equivalent)
- [ ] **Inline assembly** (for special operations)
- [ ] **System calls** (generic syscall wrapper)

### 5. Advanced Features
- [ ] **Function pointers** (callbacks)
- [ ] **Closures** (capture variables)
- [ ] **Modules/Imports** (code organization)
- [ ] **Error handling** (try/catch or Result type)
- [ ] **Concurrency** (spawn threads)

## Priority Order for Neural Networks

### MUST HAVE FIRST:
1. **Floating point support** - Can't do NN without this
2. **Arrays** - Need for weights and inputs
3. **Array indexing** - Access specific weights
4. **For loops** - Iterate through arrays
5. **Math functions** - At minimum: exp() for sigmoid

### THEN ADD:
6. File I/O - Save/load weights
7. Better memory management
8. Structs - Organize network layers
9. Network sockets - Real Redis integration
10. Concurrency - Parallel training

## Implementation Plan

### Phase 1: Float Support (DO THIS FIRST!)
```rust
// Add to lexer: recognize 1.5, 0.001, etc
// Add to parser: Float variant in Expr enum
// Add to codegen: SSE2 instructions for float math
```

### Phase 2: Arrays
```rust
// Add to parser: Array literal [1, 2, 3]
// Add to codegen: Stack allocation for arrays
// Add indexing: array[i] syntax
```

### Phase 3: Neural Network Ready
With floats + arrays, we can build REAL neural networks!

## The Choice:
Do we:
A) Add features one by one as needed?
B) Do a big upgrade adding floats + arrays together?
C) Keep using integers with scaling (less accurate)?

What do you think Ryan?