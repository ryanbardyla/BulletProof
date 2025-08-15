# NeuronLang Syntax Specification

**Version 1.0** - The World's First Biologically-Inspired Programming Language

## Overview

NeuronLang introduces revolutionary programming constructs based on biological neural networks, featuring:
- **Trinary computing** with three-state neurons (inhibited/baseline/activated)
- **Protein synthesis** for memory formation and consolidation
- **Sparse computation** with automatic baseline skipping
- **Biological memory phases** following Kandel's Nobel Prize research

## Core Data Types

### Trinary Types
```neuron
tryte           // Three-state value: inhibited (-1), baseline (0), activated (+1)
tensor<tryte>   // Multi-dimensional tryte arrays
neuron          // Complete neuron with state, threshold, and proteins
synapse         // Connection between neurons with weight and plasticity
protein         // Protein concentration (0.0 to 1.0)
```

### Traditional Types (for compatibility)
```neuron
f32, f64        // Floating point numbers
i32, i64        // Integers  
bool            // Boolean values
string          // Text strings
```

## Tryte Literals and Operations

### Tryte Values
```neuron
let inhibited_state = inhibited;  // -1, suppressive
let resting_state = baseline;     // 0, zero energy!
let excited_state = activated;    // +1, excitatory

// Automatic type inference
let auto_tryte = baseline;  // Inferred as tryte
```

### Tryte Arithmetic (Balanced Ternary)
```neuron
// Addition follows balanced ternary rules
inhibited + inhibited = activated   // -1 + -1 = +1 (overflow wraps)
baseline + activated = activated    // 0 + 1 = 1
activated + activated = inhibited   // 1 + 1 = -1 (overflow wraps)

// Multiplication
inhibited * activated = inhibited   // -1 * 1 = -1
activated * activated = activated   // 1 * 1 = 1
baseline * anything = baseline      // 0 * x = 0 (sparse!)
```

### Pattern Matching on Trytes
```neuron
match signal {
    inhibited => process_negative(),
    baseline => skip_processing(),   // Zero energy!
    activated => process_positive(),
}

// Range matching for thresholds
match neuron_output {
    x if x > 0.5 => activated,
    x if x < -0.5 => inhibited,
    _ => baseline,
}
```

## Biological Constructs

### Neuron Declaration
```neuron
neuron MemoryNeuron {
    id: "hippocampal_ca1",
    state: baseline,              // Current tryte state
    threshold: 1.5,               // Activation threshold
    
    // Protein concentrations (0.0 to 1.0)
    proteins: {
        CREB: 0.1,       // Transcription factor
        PKA: 0.2,        // Protein Kinase A  
        MAPK: 0.15,      // Mitogen-activated protein kinase
        CaMKII: 0.3,     // Calcium/calmodulin-dependent kinase
        BDNF: 0.1,       // Brain-derived neurotrophic factor
        Arc: 0.05,       // Activity-regulated cytoskeleton
        PSD95: 0.2,      // Postsynaptic density protein
        Synaptophysin: 0.15,
    }
}
```

### Synapse Connections
```neuron
synapse PlasticSynapse {
    from: neuron1,
    to: neuron2,
    weight: 1.2,                  // Synaptic strength (-2.0 to +2.0)
    plasticity: activated,        // Tryte plasticity state
    
    // LTP/LTD rules
    fn update_weight(pre_spike: bool, post_spike: bool) -> void {
        if pre_spike && post_spike {
            potentiate();  // Long-term potentiation
        } else if pre_spike && !post_spike {
            depress();     // Long-term depression
        }
    }
}
```

### Protein Synthesis
```neuron
// Conditional protein synthesis (Kandel's CREB-PKA cascade)
protein CREB synthesize(0.8) when PKA > 0.5;

// Multiple protein synthesis
protein { CREB: 0.8, Arc: 0.6, BDNF: 0.4 } synthesize when learning_signal;

// Time-dependent synthesis
protein CREB synthesize(0.7) for duration(hours: 3);
```

### Memory Consolidation
```neuron
// Memory phase transitions
consolidate pattern {
    phase: ShortTerm;        // < 1 hour, no protein synthesis
}

consolidate pattern {
    phase: EarlyLTP;         // 1-3 hours, minimal proteins
}

consolidate pattern {
    phase: LateLTP;          // 3+ hours, requires CREB > 0.7
    requires: { CREB > 0.7, Arc > 0.5 };
}

consolidate pattern {
    phase: Consolidated;     // Permanent storage
}

// Memory reconsolidation (updating existing memories)
reconsolidate existing_memory {
    phase: Reconsolidating;
    update_with: new_information;
};
```

## Sparse Computation

### Sparse Processing
```neuron
// Skip baseline neurons for 60-95% efficiency gain
sparse network {
    skip_baseline: true;         // Skip zero-energy neurons
    threshold: 0.1;              // Activation threshold
} |> process_layer(weights)

// Conditional sparse processing  
sparse_when(condition) network {
    skip_baseline: true;
} |> activation_function
```

### Tensor Operations with Sparsity
```neuron
// Sparse tensor operations
let sparse_output = tensor<tryte, [1000, 512]> {
    sparsity: 0.95;  // 95% of values are baseline
} |> sparse_matmul(weights);

// Automatic sparsity detection
let auto_sparse = detect_sparsity(network);
if auto_sparse > 0.8 {
    process_sparse(network);
} else {
    process_dense(network);
}
```

## Function Definitions

### Basic Functions
```neuron
fn activation_function(input: tryte) -> tryte {
    match input {
        inhibited => inhibited,
        baseline => baseline,    // Passthrough at zero cost
        activated => activated,
    }
}

// Generic function with biological constraints
fn protein_dependent<T>(
    input: T, 
    protein_type: ProteinType,
    threshold: f32
) -> T where T: NeuralType {
    if protein_concentration(protein_type) > threshold {
        return enhanced_processing(input);
    } else {
        return input;
    }
}
```

### Decorators
```neuron
// Differentiable functions for gradient computation
@differentiable
fn neural_layer(input: tensor<tryte, [512]>) -> tensor<tryte, [256]> {
    input |> sparse_linear(weights) |> tryte_activation
}

// Parallel processing
@parallel
fn process_batch(batch: tensor<tryte, [32, 1000]>) -> tensor<tryte, [32, 10]> {
    parallel for sample in batch {
        sample |> forward_pass
    }
}

// Device placement
@device(cuda:0)
fn gpu_compute(data: tensor<tryte, [10000]>) -> tensor<tryte, [10000]> {
    sparse data { skip_baseline: true } |> gpu_kernel
}
```

## Control Flow

### Pattern Matching
```neuron
match neuron.state {
    inhibited => {
        protein PKA synthesize(0.3);  // Mild response
        return inhibited;
    },
    baseline => {
        // Zero energy - do nothing
        return baseline;
    },
    activated => {
        protein CREB synthesize(0.8);  // Strong response
        consolidate_pattern();
        return activated;
    },
}
```

### Loops with Parallelism
```neuron
// Standard for loop
for epoch in 0..100 {
    train_epoch(data);
    
    // Protein synthesis every 10 epochs
    if epoch % 10 == 0 {
        protein CREB synthesize(0.7);
        consolidate_memories();
    }
}

// Parallel for loop
parallel for neuron in network {
    if neuron.state != baseline {  // Skip baseline neurons
        process_neuron(neuron);
    }
}

// Parallel processing with thread count
parallel(threads: 16) for batch in dataset {
    let result = forward_pass(batch);
    accumulate_gradients(result);
}
```

### Conditional Processing
```neuron
if protein_concentration(CREB) > 0.7 {
    // Late-phase LTP possible
    consolidate_memory(LateLTP);
} else if protein_concentration(PKA) > 0.5 {
    // Early-phase LTP only
    consolidate_memory(EarlyLTP);  
} else {
    // Short-term memory only
    consolidate_memory(ShortTerm);
}
```

## Model Definitions

### Neural Network Models
```neuron
model BiologicalNet {
    // Layer definitions
    input_layer: tensor<tryte, [1000]>,
    hidden1: tensor<tryte, [512]>,  
    hidden2: tensor<tryte, [256]>,
    output: tensor<tryte, [10]>,
    
    // Protein state for the entire network
    network_proteins: {
        CREB: 0.1,
        PKA: 0.2, 
        BDNF: 0.1,
    },
    
    // Forward pass with sparse computation
    @differentiable
    fn forward(input: tensor<tryte, [1000]>) -> tensor<tryte, [10]> {
        // Layer 1: 95% sparsity
        let h1 = sparse input {
            skip_baseline: true;
        } |> linear_tryte(hidden1);
        
        // Layer 2: Protein-modulated activation
        let h2 = h1 |> protein_activation(CREB, 0.6) |> linear_tryte(hidden2);
        
        // Output with memory consolidation
        let output = h2 |> linear_tryte(output);
        
        // Consolidate learned patterns
        consolidate output { phase: LateLTP };
        
        return output;
    }
}
```

### Training Configuration
```neuron
train BiologicalNet {
    dataset: "neural_patterns",
    
    optimizer: BiologicalAdam {
        learning_rate: 0.001,
        protein_modulation: true,      // Use protein concentrations
        creb_factor: 0.8,              // CREB affects learning rate
        sparse_gradients: true,        // Skip baseline gradients
    },
    
    loss: TryteCrossEntropy,
    
    callbacks: [
        ProteinSynthesis {             // Trigger every N steps
            interval: 100,
            proteins: { CREB: 0.7, Arc: 0.5 }
        },
        MemoryConsolidation {          // Periodic consolidation
            interval: 1000,
            phase: LateLTP,
        }
    ],
    
    epochs: 50,
    batch_size: 32,
    device: cuda:0,
}
```

## Pipeline Operators

### Data Flow Pipelines
```neuron
// Functional pipeline processing
let result = input_data
    |> preprocess_trytes
    |> sparse { skip_baseline: true }
    |> neural_layer_1
    |> protein_activation(CREB, 0.6)
    |> neural_layer_2  
    |> consolidate { phase: LateLTP }
    |> postprocess;

// Branching pipelines
let (path1, path2) = input
    |> split_data
    |> branch(
        |> excitatory_pathway |> potentiate,
        |> inhibitory_pathway |> depress
    );
```

## Error Handling

### Biological Constraint Validation
```neuron
fn validate_protein_synthesis(creb_level: f32) -> Result<void, NeuronError> {
    if creb_level < 0.7 {
        return Err(NeuronError::InsufficientCREB {
            required: 0.7,
            actual: creb_level,
            message: "Late-phase LTP requires CREB > 0.7"
        });
    }
    
    Ok(())
}

// Automatic constraint checking
@validate_proteins
fn late_ltp_consolidation(pattern: tensor<tryte, [512]>) -> void {
    consolidate pattern { phase: LateLTP };  // Auto-validates CREB > 0.7
}
```

## Memory Management

### Protein Lifecycle
```neuron
// Automatic protein degradation
protein CREB synthesize(0.8) lifetime(hours: 6);

// Protein half-life modeling
protein Arc synthesize(0.6) half_life(minutes: 30);

// Cleanup after memory consolidation
consolidate pattern { 
    phase: Consolidated;
    cleanup_proteins: true;  // Free temporary proteins
};
```

### Sparse Memory Optimization
```neuron
// Automatic garbage collection of baseline neurons
sparse_gc network {
    threshold: 0.1;          // Collect neurons below threshold
    preserve_synapses: true; // Keep connection structure
}

// Memory pool for tryte operations
@memory_pool(size: "1GB")
fn large_tryte_computation(data: tensor<tryte, [1000000]>) -> tensor<tryte, [1000000]> {
    // Efficient memory usage for massive tryte tensors
    sparse data |> parallel_process
}
```

## Comments and Documentation

### Documentation Comments
```neuron
/// Implements Kandel's CREB-PKA cascade for memory formation
/// Based on Nobel Prize research (2000)
///
/// # Arguments
/// * `stimulus_strength` - Synaptic input strength (0.0 to 2.0)
/// * `duration` - Stimulus duration in milliseconds
///
/// # Returns
/// * Memory consolidation phase achieved
fn creb_pka_cascade(stimulus_strength: f32, duration: u32) -> MemoryPhase {
    // Implementation details...
}

/// Trinary neuron with biological protein dynamics
/// 
/// # Protein Requirements
/// - CREB > 0.7: Required for late-phase LTP
/// - PKA > 0.5: Required for CREB activation  
/// - Arc > 0.4: Required for synaptic plasticity
neuron DocumentedNeuron {
    // Implementation...
}
```

### Inline Comments
```neuron
// Single-line biological comment
let creb_threshold = 0.7;  // CREB activation threshold from Kandel research

/* Multi-line comment explaining
   the biological mechanism behind
   protein synthesis cascades */
```

## Revolutionary Advantages

### Energy Efficiency
```neuron
// Baseline neurons consume ZERO energy (unlike binary always-on)
let resting_network = tensor<tryte, [1000000]> filled_with(baseline);
let energy_cost = calculate_energy(resting_network);  // Returns 0.0!
```

### Sparse Processing
```neuron
// Automatically skip 60-95% of computations
let sparse_result = huge_network
    |> sparse { skip_baseline: true }  // Massive efficiency gain
    |> complex_processing;
```

### Biological Learning
```neuron
// Protein-driven learning matches brain mechanisms
protein CREB synthesize(0.8);  // Better than pure gradient descent
consolidate learned_patterns { phase: LateLTP };  // Permanent retention
```

This syntax specification defines the complete NeuronLang language, enabling the world's first biologically-accurate programming paradigm with revolutionary efficiency gains through trinary computing and sparse processing.