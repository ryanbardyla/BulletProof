# Research Document 001: LLVM IR Deep Dive

## Objective
Understand LLVM IR at the deepest level to design NeuronLang's compilation pipeline.

## Key Questions to Answer
1. How does LLVM handle vectorization for tensors?
2. What are the limitations for GPU code generation?
3. How can we optimize for AI workloads specifically?
4. Should we use LLVM directly or build our own backend?

## LLVM IR Basics

### What is LLVM IR?
- **Three forms**: Human-readable (.ll), Bitcode (.bc), In-memory
- **SSA form**: Static Single Assignment - each variable assigned once
- **Type system**: Strongly typed with explicit casts
- **Platform independent**: But can include target-specific intrinsics

### Example: Simple Tensor Addition
```llvm
; Function to add two tensors element-wise
define void @tensor_add(float* %a, float* %b, float* %result, i64 %size) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next_i, %loop ]
  %a_ptr = getelementptr float, float* %a, i64 %i
  %b_ptr = getelementptr float, float* %b, i64 %i
  %r_ptr = getelementptr float, float* %result, i64 %i
  
  %a_val = load float, float* %a_ptr
  %b_val = load float, float* %b_ptr
  %sum = fadd float %a_val, %b_val
  store float %sum, float* %r_ptr
  
  %next_i = add i64 %i, 1
  %cond = icmp slt i64 %next_i, %size
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}
```

## Problems with LLVM for AI

### 1. **No Native Tensor Type**
- Tensors are just pointers to memory
- Shape information lost at IR level
- No built-in broadcasting semantics

### 2. **Limited Vectorization**
- Auto-vectorization often fails for complex patterns
- SIMD intrinsics are platform-specific
- No tensor-aware optimizations

### 3. **GPU Support Issues**
- NVPTX backend is limited
- Can't express GPU parallelism patterns well
- Memory hierarchy not exposed (shared, constant, etc.)

### 4. **No Autodiff Support**
- Must implement gradient tracking separately
- Can't optimize forward and backward together
- No gradient checkpointing primitives

## MLIR: The Better Alternative?

### What is MLIR?
- **Multi-Level IR**: Multiple abstraction levels
- **Dialect system**: Extensible with custom operations
- **Built for ML**: Created by Google for TensorFlow

### MLIR Dialects for AI
```mlir
// Tensor dialect - high level
func @matmul(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>) -> tensor<2x4xf32> {
  %result = tensor.matmul %a, %b : tensor<2x3xf32>, tensor<3x4xf32> -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}

// Linalg dialect - mid level
func @matmul_linalg(%a: memref<2x3xf32>, %b: memref<3x4xf32>, %c: memref<2x4xf32>) {
  linalg.matmul ins(%a, %b : memref<2x3xf32>, memref<3x4xf32>)
                outs(%c : memref<2x4xf32>)
  return
}

// GPU dialect - low level
gpu.func @matmul_kernel(%a: memref<2x3xf32>, %b: memref<3x4xf32>, %c: memref<2x4xf32>) {
  %tid = gpu.thread_id x
  %bid = gpu.block_id x
  // Kernel implementation
  gpu.return
}
```

## Triton: OpenAI's Approach

### Key Insights from Triton
1. **Block-based programming**: Work on tensor blocks, not elements
2. **Auto-tuning**: Automatically find optimal parameters
3. **Python-like syntax**: But compiles to PTX
4. **Fusion-friendly**: Automatic kernel fusion

### Example Triton Kernel
```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak):
    pid = tl.program_id(0)
    # Block-level operations
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    a_block = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :])
    # Automatic memory coalescing, bank conflict avoidance
```

## Design Decision: Our Compilation Pipeline

### NeuronLang → NeuronIR → MLIR → GPU/CPU

**Why this approach?**
1. **NeuronIR**: Preserves tensor semantics and shapes
2. **MLIR**: Leverages existing optimizations
3. **Multiple backends**: CPU via LLVM, GPU via Triton/CUDA
4. **Gradual lowering**: Optimize at each level

### NeuronIR Design
```
// High-level tensor operations with shape info
operation matmul<M, N, K> {
  inputs: [tensor<M, K>, tensor<K, N>]
  outputs: [tensor<M, N>]
  attributes: {differentiable: true}
  lowering: {
    cpu: "neuron.cpu.matmul",
    gpu: "neuron.gpu.matmul"
  }
}
```

## Benchmarks to Run

### Test Programs
1. Matrix multiplication (various sizes)
2. Convolution (different kernels)
3. Attention mechanism
4. Full transformer forward pass

### Metrics to Measure
- Compilation time
- Binary size
- Execution speed
- Memory usage
- GPU utilization

## Next Research Steps

1. **Study MLIR in detail** - Understand dialect system
2. **Analyze Triton internals** - Learn from their GPU compilation
3. **Benchmark LLVM vs MLIR** - For tensor operations
4. **Design NeuronIR specification** - Our custom IR format

## Key Findings

✅ **LLVM alone is insufficient** - Need higher-level IR for tensors
✅ **MLIR is promising** - But still needs custom dialects
✅ **Triton has good ideas** - Block programming and auto-tuning
✅ **Multi-level approach needed** - Different optimizations at each level

## Recommendation

**Build NeuronLang with multi-level compilation:**
1. NeuronLang (source) 
2. NeuronIR (tensor-aware IR)
3. MLIR dialects (optimization)
4. LLVM/PTX (code generation)

This gives us control over tensor semantics while leveraging existing infrastructure.

---

*Research completed: Day 1*
*Next: Research tensor type systems and shape inference*