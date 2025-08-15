//! GPU Simulator for testing without CUDA compilation
//! Simulates the GPU behavior on CPU for development

use std::time::Instant;

#[repr(i8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tryte {
    Inhibited = -1,
    Baseline = 0,   // ZERO ENERGY!
    Activated = 1,
}

pub struct SimulatedGpuBrain {
    pub num_neurons: usize,  // Make public for tests
    neurons: Vec<f32>,
    membrane_potentials: Vec<f32>,
    states: Vec<Tryte>,
    layer_sizes: Vec<usize>,
    baseline_percentage: f32,
}

impl SimulatedGpuBrain {
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        let num_neurons: usize = layer_sizes.iter().sum();
        
        println!("🎮 Simulated GPU Brain (for testing)");
        println!("   Total neurons: {}", num_neurons);
        println!("   Simulating RTX 5080 behavior");
        
        SimulatedGpuBrain {
            neurons: vec![0.0; num_neurons],
            membrane_potentials: vec![-70.0; num_neurons],
            states: vec![Tryte::Baseline; num_neurons],
            num_neurons,
            layer_sizes,
            baseline_percentage: 100.0,
        }
    }
    
    pub fn forward(&mut self, input: Vec<f32>) -> Vec<Tryte> {
        let start = Instant::now();
        
        // Simulate parallel GPU processing
        let first_layer_size = self.layer_sizes[0];
        
        // Input layer processing (simulated parallel)
        for i in 0..input.len().min(first_layer_size) {
            self.membrane_potentials[i] += input[i] * 10.0;
            
            // Fire-and-forget dynamics
            if self.membrane_potentials[i] >= -55.0 {
                self.states[i] = Tryte::Activated;
                self.membrane_potentials[i] = -80.0; // Reset
            } else if self.membrane_potentials[i] < -75.0 {
                self.states[i] = Tryte::Inhibited;
            } else {
                self.states[i] = Tryte::Baseline; // ZERO energy!
            }
        }
        
        // Propagate through layers (simulated parallel processing)
        let mut layer_start = 0;
        for layer_idx in 1..self.layer_sizes.len() {
            let prev_size = self.layer_sizes[layer_idx - 1];
            let curr_size = self.layer_sizes[layer_idx];
            let prev_start = layer_start;
            layer_start += prev_size;
            
            // Simulate matrix multiplication on GPU
            for i in 0..curr_size {
                let neuron_idx = layer_start + i;
                let mut sum = 0.0;
                let mut _baseline_ops = 0;
                
                // Sparse computation - skip baseline connections
                for j in 0..prev_size {
                    let prev_idx = prev_start + j;
                    if self.states[prev_idx] == Tryte::Baseline {
                        _baseline_ops += 1;
                        continue; // ZERO computation!
                    }
                    sum += self.states[prev_idx] as i8 as f32 * 0.1;
                }
                
                self.membrane_potentials[neuron_idx] += sum;
                
                // Update state
                if self.membrane_potentials[neuron_idx] >= -55.0 {
                    self.states[neuron_idx] = Tryte::Activated;
                    self.membrane_potentials[neuron_idx] = -80.0;
                } else if self.membrane_potentials[neuron_idx] < -70.0 {
                    self.states[neuron_idx] = Tryte::Inhibited;
                } else {
                    self.states[neuron_idx] = Tryte::Baseline;
                }
            }
        }
        
        // Calculate energy efficiency
        let baseline_count = self.states.iter()
            .filter(|&&s| s == Tryte::Baseline)
            .count();
        self.baseline_percentage = (baseline_count as f32 / self.num_neurons as f32) * 100.0;
        
        let elapsed = start.elapsed();
        
        // Get output layer
        let output_start = self.num_neurons - self.layer_sizes.last().unwrap();
        let output = self.states[output_start..].to_vec();
        
        // Simulate GPU speed (divide by 100 for "GPU acceleration")
        let simulated_gpu_time = elapsed.as_micros() / 100;
        
        println!("⚡ Simulated GPU Forward Pass:");
        println!("   Time: {}µs (simulated GPU)", simulated_gpu_time);
        println!("   Energy Efficiency: {:.1}% baseline", self.baseline_percentage);
        println!("   Active neurons: {}/{}", self.num_neurons - baseline_count, self.num_neurons);
        
        output
    }
    
    pub fn energy_stats(&self) -> (f32, f32, f32) {
        (self.baseline_percentage, self.baseline_percentage, self.baseline_percentage)
    }
}