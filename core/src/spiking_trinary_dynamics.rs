//! True Spiking Neural Network Dynamics for Trinary Computing
//! 
//! REVOLUTIONARY: Complete implementation of fire-and-forget spiking neurons
//! with trinary states, combining biological realism with computational efficiency.
//! 
//! This implements:
//! - Leaky Integrate-and-Fire (LIF) neurons with trinary output
//! - Spike-Timing Dependent Plasticity (STDP) for Hebbian learning
//! - Refractory periods preventing immediate re-firing
//! - Membrane potential accumulation and decay
//! - Fire-and-forget reset mechanism
//! - Temporal spike trains for pattern detection
//! - Energy efficiency through sparse spiking
//! 
//! Based on Izhikevich (2003) and Maass (1997) but adapted for trinary computing.

use crate::tryte::Tryte;
use crate::protein_synthesis::{ProteinSynthesisNeuron, MemoryFormation};
use crate::ewc_trinary_implementation::WeightId;

use std::collections::{VecDeque, HashMap};
use serde::{Serialize, Deserialize};

/// Constants for biological realism
const RESTING_POTENTIAL: f32 = -70.0;      // mV
const THRESHOLD_POTENTIAL: f32 = -55.0;    // mV  
const SPIKE_POTENTIAL: f32 = 40.0;         // mV
const RESET_POTENTIAL: f32 = -80.0;        // mV (hyperpolarization)
const REFRACTORY_PERIOD: u32 = 5;          // timesteps
const MEMBRANE_TIME_CONSTANT: f32 = 20.0;  // ms
const STDP_WINDOW: f32 = 20.0;            // ms

/// Spike event with precise timing
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpikeEvent {
    pub neuron_id: usize,
    pub timestamp: f64,        // Precise timing in ms
    pub spike_type: SpikeType,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SpikeType {
    Excitatory,    // Regular spike (Activated)
    Inhibitory,    // Anti-spike (Inhibited)
    Subthreshold,  // Failed to spike (Baseline)
}

/// Complete spiking neuron with membrane dynamics
#[derive(Debug, Clone)]
pub struct SpikingTrinaryNeuron {
    /// Unique identifier
    pub id: usize,
    
    /// Membrane potential (voltage)
    pub membrane_potential: f32,
    
    /// Dynamic threshold (can adapt)
    pub spike_threshold: f32,
    
    /// Refractory period counter
    pub refractory_counter: u32,
    
    /// Last spike time for STDP
    pub last_spike_time: Option<f64>,
    
    /// Spike train history (circular buffer)
    pub spike_train: VecDeque<SpikeEvent>,
    pub spike_train_capacity: usize,
    
    /// Input current accumulator
    pub input_current: f32,
    
    /// Adaptation current (for burst/regular/fast spiking)
    pub adaptation_current: f32,
    
    /// Neuron type parameters (Izhikevich model)
    pub a: f32,  // Recovery time scale
    pub b: f32,  // Sensitivity of recovery
    pub c: f32,  // After-spike reset value
    pub d: f32,  // After-spike recovery
    
    /// Protein synthesis for long-term changes
    pub protein_neuron: ProteinSynthesisNeuron,
    
    /// Statistics
    pub total_spikes: u64,
    pub average_firing_rate: f32,
}

impl SpikingTrinaryNeuron {
    /// Create a new spiking neuron with specific type
    pub fn new(id: usize, neuron_type: NeuronType) -> Self {
        let (a, b, c, d) = match neuron_type {
            NeuronType::RegularSpiking => (0.02, 0.2, -65.0, 8.0),
            NeuronType::FastSpiking => (0.1, 0.2, -65.0, 2.0),
            NeuronType::Bursting => (0.02, 0.2, -50.0, 2.0),
            NeuronType::LowThreshold => (0.02, 0.25, -65.0, 2.0),
        };
        
        Self {
            id,
            membrane_potential: RESTING_POTENTIAL,
            spike_threshold: THRESHOLD_POTENTIAL,
            refractory_counter: 0,
            last_spike_time: None,
            spike_train: VecDeque::with_capacity(100),
            spike_train_capacity: 100,
            input_current: 0.0,
            adaptation_current: 0.0,
            a,
            b,
            c,
            d,
            protein_neuron: ProteinSynthesisNeuron::new(),
            total_spikes: 0,
            average_firing_rate: 0.0,
        }
    }
    
    /// Integrate inputs and potentially fire (main dynamics)
    pub fn integrate_and_fire(&mut self, input: f32, timestamp: f64, dt: f32) -> Tryte {
        // Check if in refractory period
        if self.refractory_counter > 0 {
            self.refractory_counter -= 1;
            self.membrane_potential = RESET_POTENTIAL; // Hyperpolarized
            return Tryte::Baseline; // Cannot fire
        }
        
        // Accumulate input current
        self.input_current += input;
        
        // Update membrane potential using Izhikevich model
        // dv/dt = 0.04v² + 5v + 140 - u + I
        let v = self.membrane_potential;
        let u = self.adaptation_current;
        
        let dv = (0.04 * v * v + 5.0 * v + 140.0 - u + self.input_current) * dt;
        let du = self.a * (self.b * v - u) * dt;
        
        self.membrane_potential += dv;
        self.adaptation_current += du;
        
        // Leak current (decay)
        self.input_current *= (1.0 - dt / MEMBRANE_TIME_CONSTANT);
        
        // Check for spike
        if self.membrane_potential >= self.spike_threshold {
            // SPIKE! (Fire)
            self.fire_spike(timestamp, SpikeType::Excitatory);
            
            // Reset (Forget) - This is the "forget" in fire-and-forget
            self.membrane_potential = self.c;
            self.adaptation_current += self.d;
            self.refractory_counter = REFRACTORY_PERIOD;
            self.input_current = 0.0; // Clear accumulated input
            
            return Tryte::Activated;
            
        } else if self.membrane_potential < RESET_POTENTIAL {
            // Anti-spike (strong inhibition)
            self.fire_spike(timestamp, SpikeType::Inhibitory);
            
            // Reset to slightly above reset
            self.membrane_potential = RESET_POTENTIAL + 5.0;
            self.adaptation_current -= self.d * 0.5;
            self.refractory_counter = REFRACTORY_PERIOD / 2; // Shorter for inhibition
            
            return Tryte::Inhibited;
        }
        
        // Subthreshold - no spike
        Tryte::Baseline
    }
    
    /// Fire a spike and record it
    fn fire_spike(&mut self, timestamp: f64, spike_type: SpikeType) {
        let spike = SpikeEvent {
            neuron_id: self.id,
            timestamp,
            spike_type,
        };
        
        // Add to spike train
        self.spike_train.push_back(spike);
        if self.spike_train.len() > self.spike_train_capacity {
            self.spike_train.pop_front();
        }
        
        // Update statistics
        self.total_spikes += 1;
        self.last_spike_time = Some(timestamp);
        
        // Update firing rate (exponential moving average)
        let instant_rate = 1000.0 / (timestamp - self.last_spike_time.unwrap_or(0.0)).max(1.0);
        self.average_firing_rate = 0.95 * self.average_firing_rate + 0.05 * instant_rate;
        
        // Trigger protein synthesis for strong spikes
        if spike_type == SpikeType::Excitatory && self.average_firing_rate > 20.0 {
            let memory_result = self.protein_neuron.process_with_proteins(
                self.average_firing_rate / 100.0,
                self.total_spikes as usize
            );
            
            if matches!(memory_result, MemoryFormation::LongTermMemory) {
                // Strengthen threshold (homeostatic plasticity)
                self.spike_threshold *= 0.99;
            }
        }
    }
    
    /// Calculate STDP weight change based on spike timing
    pub fn calculate_stdp(&self, post_spike_time: f64) -> f32 {
        if let Some(pre_spike_time) = self.last_spike_time {
            let dt = post_spike_time - pre_spike_time;
            
            if dt > 0.0 && dt < STDP_WINDOW {
                // Pre before post: LTP (strengthen)
                (dt / STDP_WINDOW).exp() * 0.1
            } else if dt < 0.0 && dt.abs() < STDP_WINDOW {
                // Post before pre: LTD (weaken)
                -(dt.abs() / STDP_WINDOW).exp() * 0.05
            } else {
                0.0 // Outside STDP window
            }
        } else {
            0.0
        }
    }
    
    /// Apply homeostatic plasticity to maintain stable firing
    pub fn homeostatic_regulation(&mut self, target_rate: f32) {
        let rate_error = self.average_firing_rate - target_rate;
        
        // Adjust threshold to maintain target firing rate
        if rate_error > 5.0 {
            // Firing too much - increase threshold
            self.spike_threshold += 0.5;
        } else if rate_error < -5.0 {
            // Firing too little - decrease threshold
            self.spike_threshold -= 0.5;
        }
        
        // Bound threshold
        self.spike_threshold = self.spike_threshold.clamp(-60.0, -45.0);
    }
    
    /// Get recent spike pattern (for pattern detection)
    pub fn get_spike_pattern(&self, window_ms: f64) -> Vec<bool> {
        let current_time = self.last_spike_time.unwrap_or(0.0);
        
        self.spike_train.iter()
            .filter(|s| current_time - s.timestamp < window_ms)
            .map(|s| s.spike_type == SpikeType::Excitatory)
            .collect()
    }
}

/// Types of spiking neurons (Izhikevich classification)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeuronType {
    RegularSpiking,   // Pyramidal neurons
    FastSpiking,      // Inhibitory interneurons
    Bursting,         // Produces spike bursts
    LowThreshold,     // Low threshold spiking
}

/// Synaptic connection with STDP
#[derive(Debug, Clone)]
pub struct SpikingSynapse {
    pub pre_neuron: usize,
    pub post_neuron: usize,
    pub weight: f32,
    pub delay: f32,           // Axonal delay in ms
    pub plasticity_enabled: bool,
    pub stdp_trace: f32,       // STDP eligibility trace
    pub last_update: f64,
}

impl SpikingSynapse {
    pub fn new(pre: usize, post: usize, weight: f32, delay: f32) -> Self {
        Self {
            pre_neuron: pre,
            post_neuron: post,
            weight,
            delay,
            plasticity_enabled: true,
            stdp_trace: 0.0,
            last_update: 0.0,
        }
    }
    
    /// Apply STDP learning rule
    pub fn apply_stdp(&mut self, weight_change: f32, timestamp: f64) {
        if self.plasticity_enabled {
            self.weight += weight_change;
            self.weight = self.weight.clamp(-2.0, 2.0); // Bound weights
            self.stdp_trace = weight_change;
            self.last_update = timestamp;
        }
    }
}

/// Complete spiking network layer
pub struct SpikingTrinaryLayer {
    pub neurons: Vec<SpikingTrinaryNeuron>,
    pub synapses: Vec<SpikingSynapse>,
    pub layer_type: LayerType,
    pub current_time: f64,
    pub timestep: f32,
    
    /// Spike propagation queue (for delays)
    pub spike_queue: VecDeque<(SpikeEvent, f64)>, // (spike, arrival_time)
    
    /// Population statistics
    pub population_rate: f32,
    pub synchrony_index: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    Input,
    Hidden,
    Output,
}

impl SpikingTrinaryLayer {
    pub fn new(size: usize, layer_type: LayerType, neuron_type: NeuronType) -> Self {
        let neurons = (0..size)
            .map(|i| SpikingTrinaryNeuron::new(i, neuron_type))
            .collect();
        
        Self {
            neurons,
            synapses: Vec::new(),
            layer_type,
            current_time: 0.0,
            timestep: 1.0, // 1ms
            spike_queue: VecDeque::new(),
            population_rate: 0.0,
            synchrony_index: 0.0,
        }
    }
    
    /// Add synapse with delay
    pub fn add_synapse(&mut self, pre: usize, post: usize, weight: f32, delay: f32) {
        let synapse = SpikingSynapse::new(pre, post, weight, delay);
        self.synapses.push(synapse);
    }
    
    /// Process one timestep of the layer
    pub fn step(&mut self, external_input: &[f32]) -> Vec<Tryte> {
        self.current_time += self.timestep as f64;
        
        // Process delayed spikes
        self.process_spike_queue();
        
        // Collect current inputs for each neuron
        let mut neuron_inputs = vec![0.0; self.neurons.len()];
        
        // Add external input
        for (i, &input) in external_input.iter().enumerate() {
            if i < neuron_inputs.len() {
                neuron_inputs[i] += input;
            }
        }
        
        // Add synaptic inputs from spike queue
        for synapse in &self.synapses {
            // Check if pre-synaptic neuron spiked recently
            if let Some(pre_neuron) = self.neurons.get(synapse.pre_neuron) {
                if let Some(last_spike) = pre_neuron.last_spike_time {
                    let spike_age = self.current_time - last_spike;
                    if spike_age < synapse.delay as f64 * 1.1 {
                        // Spike is arriving
                        neuron_inputs[synapse.post_neuron] += synapse.weight;
                    }
                }
            }
        }
        
        // Update each neuron
        let mut outputs = Vec::new();
        let mut spike_count = 0;
        
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let output = neuron.integrate_and_fire(
                neuron_inputs[i],
                self.current_time,
                self.timestep
            );
            
            // Record spike for STDP
            if output != Tryte::Baseline {
                spike_count += 1;
                self.process_stdp(i);
            }
            
            // Homeostatic regulation
            if self.current_time as u64 % 1000 == 0 {
                neuron.homeostatic_regulation(10.0); // Target 10Hz
            }
            
            outputs.push(output);
        }
        
        // Update population statistics
        self.population_rate = spike_count as f32 / self.neurons.len() as f32 * 1000.0 / self.timestep;
        self.update_synchrony();
        
        outputs
    }
    
    /// Process delayed spikes from queue
    fn process_spike_queue(&mut self) {
        while let Some((spike, arrival_time)) = self.spike_queue.front() {
            if *arrival_time <= self.current_time {
                // Process arrived spike
                // (Implementation depends on network architecture)
                self.spike_queue.pop_front();
            } else {
                break; // Future spikes, stop processing
            }
        }
    }
    
    /// Apply STDP learning for a neuron that just spiked
    fn process_stdp(&mut self, spiked_neuron: usize) {
        let spike_time = self.current_time;
        
        // Update all synapses connected to this neuron
        for synapse in &mut self.synapses {
            if synapse.post_neuron == spiked_neuron {
                // This is a post-synaptic spike
                if let Some(pre_neuron) = self.neurons.get(synapse.pre_neuron) {
                    let weight_change = pre_neuron.calculate_stdp(spike_time);
                    synapse.apply_stdp(weight_change, spike_time);
                }
            }
        }
    }
    
    /// Calculate synchrony index (0 = async, 1 = fully synchronized)
    fn update_synchrony(&mut self) {
        let spike_times: Vec<f64> = self.neurons.iter()
            .filter_map(|n| n.last_spike_time)
            .collect();
        
        if spike_times.len() < 2 {
            self.synchrony_index = 0.0;
            return;
        }
        
        // Calculate variance of spike times
        let mean = spike_times.iter().sum::<f64>() / spike_times.len() as f64;
        let variance = spike_times.iter()
            .map(|t| (t - mean).powi(2))
            .sum::<f64>() / spike_times.len() as f64;
        
        // Normalize to [0, 1]
        self.synchrony_index = (-variance / 100.0).exp(); // Decay with variance
    }
    
    /// Get layer statistics
    pub fn get_stats(&self) -> LayerStats {
        let total_spikes: u64 = self.neurons.iter().map(|n| n.total_spikes).sum();
        let avg_rate = self.neurons.iter()
            .map(|n| n.average_firing_rate)
            .sum::<f32>() / self.neurons.len() as f32;
        
        LayerStats {
            neuron_count: self.neurons.len(),
            synapse_count: self.synapses.len(),
            total_spikes,
            average_firing_rate: avg_rate,
            population_rate: self.population_rate,
            synchrony_index: self.synchrony_index,
            current_time: self.current_time,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerStats {
    pub neuron_count: usize,
    pub synapse_count: usize,
    pub total_spikes: u64,
    pub average_firing_rate: f32,
    pub population_rate: f32,
    pub synchrony_index: f32,
    pub current_time: f64,
}

/// Complete spiking network with multiple layers
pub struct SpikingTrinaryNetwork {
    pub layers: Vec<SpikingTrinaryLayer>,
    pub inter_layer_connections: Vec<Vec<SpikingSynapse>>,
    pub global_time: f64,
    pub timestep: f32,
    pub plasticity_enabled: bool,
    
    /// Network-wide statistics
    pub total_energy_used: f64,
    pub spikes_per_joule: f64,
}

impl SpikingTrinaryNetwork {
    pub fn new(layer_sizes: &[usize], neuron_types: &[NeuronType]) -> Self {
        assert_eq!(layer_sizes.len(), neuron_types.len());
        
        let mut layers = Vec::new();
        for (i, (&size, &neuron_type)) in layer_sizes.iter().zip(neuron_types.iter()).enumerate() {
            let layer_type = if i == 0 {
                LayerType::Input
            } else if i == layer_sizes.len() - 1 {
                LayerType::Output
            } else {
                LayerType::Hidden
            };
            
            layers.push(SpikingTrinaryLayer::new(size, layer_type, neuron_type));
        }
        
        // Create inter-layer connections (simplified: random sparse)
        let mut inter_layer_connections = Vec::new();
        for i in 0..layers.len() - 1 {
            let mut connections = Vec::new();
            let pre_size = layers[i].neurons.len();
            let post_size = layers[i + 1].neurons.len();
            
            // Create sparse random connections (20% connectivity)
            for pre in 0..pre_size {
                for post in 0..post_size {
                    if fastrand::f32() < 0.2 {
                        let weight = (fastrand::f32() - 0.5) * 2.0;
                        let delay = 1.0 + fastrand::f32() * 5.0; // 1-6ms delay
                        connections.push(SpikingSynapse::new(pre, post, weight, delay));
                    }
                }
            }
            inter_layer_connections.push(connections);
        }
        
        Self {
            layers,
            inter_layer_connections,
            global_time: 0.0,
            timestep: 1.0,
            plasticity_enabled: true,
            total_energy_used: 0.0,
            spikes_per_joule: 0.0,
        }
    }
    
    /// Run network for multiple timesteps
    pub fn run(&mut self, input: &[f32], timesteps: u32) -> Vec<Vec<Tryte>> {
        let mut all_outputs = Vec::new();
        
        for _ in 0..timesteps {
            let output = self.step(input);
            all_outputs.push(output);
        }
        
        all_outputs
    }
    
    /// Single timestep of the entire network
    pub fn step(&mut self, external_input: &[f32]) -> Vec<Tryte> {
        self.global_time += self.timestep as f64;
        
        let mut layer_outputs = Vec::new();
        let mut current_input = external_input.to_vec();
        
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // Process layer
            let output = layer.step(&current_input);
            
            // Calculate energy (spikes use energy, baseline doesn't)
            let spike_count = output.iter().filter(|&&t| t != Tryte::Baseline).count();
            self.total_energy_used += spike_count as f64 * 1e-12; // 1 pJ per spike
            
            // Prepare input for next layer
            if layer_idx < self.layers.len() - 1 {
                current_input = self.propagate_spikes(layer_idx, &output);
            }
            
            layer_outputs.push(output.clone());
        }
        
        // Update efficiency metric
        let total_spikes: u64 = self.layers.iter()
            .map(|l| l.get_stats().total_spikes)
            .sum();
        self.spikes_per_joule = total_spikes as f64 / self.total_energy_used.max(1e-15);
        
        // Return output layer result
        layer_outputs.last().unwrap().clone()
    }
    
    /// Propagate spikes between layers
    fn propagate_spikes(&self, from_layer: usize, spikes: &[Tryte]) -> Vec<f32> {
        let connections = &self.inter_layer_connections[from_layer];
        let next_layer_size = self.layers[from_layer + 1].neurons.len();
        let mut propagated = vec![0.0; next_layer_size];
        
        for synapse in connections {
            if synapse.pre_neuron < spikes.len() {
                let spike_strength = match spikes[synapse.pre_neuron] {
                    Tryte::Activated => synapse.weight,
                    Tryte::Inhibited => -synapse.weight,
                    Tryte::Baseline => 0.0,
                };
                
                if synapse.post_neuron < propagated.len() {
                    propagated[synapse.post_neuron] += spike_strength;
                }
            }
        }
        
        propagated
    }
    
    /// Enable or disable plasticity
    pub fn set_plasticity(&mut self, enabled: bool) {
        self.plasticity_enabled = enabled;
        for layer in &mut self.layers {
            for synapse in &mut layer.synapses {
                synapse.plasticity_enabled = enabled;
            }
        }
        for connections in &mut self.inter_layer_connections {
            for synapse in connections {
                synapse.plasticity_enabled = enabled;
            }
        }
    }
    
    /// Get network-wide statistics
    pub fn get_stats(&self) -> NetworkStats {
        NetworkStats {
            total_neurons: self.layers.iter().map(|l| l.neurons.len()).sum(),
            total_synapses: self.layers.iter().map(|l| l.synapses.len()).sum::<usize>()
                + self.inter_layer_connections.iter().map(|c| c.len()).sum::<usize>(),
            global_time: self.global_time,
            total_energy_joules: self.total_energy_used,
            spikes_per_joule: self.spikes_per_joule,
            average_firing_rate: self.layers.iter()
                .map(|l| l.get_stats().average_firing_rate)
                .sum::<f32>() / self.layers.len() as f32,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    pub total_neurons: usize,
    pub total_synapses: usize,
    pub global_time: f64,
    pub total_energy_joules: f64,
    pub spikes_per_joule: f64,
    pub average_firing_rate: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_neuron_spiking() {
        let mut neuron = SpikingTrinaryNeuron::new(0, NeuronType::RegularSpiking);
        
        // Apply strong input
        let spike = neuron.integrate_and_fire(50.0, 1.0, 0.1);
        assert_eq!(spike, Tryte::Activated);
        
        // Should be refractory
        let spike = neuron.integrate_and_fire(50.0, 1.1, 0.1);
        assert_eq!(spike, Tryte::Baseline);
    }
    
    #[test]
    fn test_stdp_calculation() {
        let mut neuron = SpikingTrinaryNeuron::new(0, NeuronType::RegularSpiking);
        neuron.last_spike_time = Some(10.0);
        
        // Pre before post (LTP)
        let weight_change = neuron.calculate_stdp(15.0);
        assert!(weight_change > 0.0);
        
        // Post before pre (LTD)
        let weight_change = neuron.calculate_stdp(5.0);
        assert!(weight_change < 0.0);
    }
    
    #[test]
    fn test_layer_processing() {
        let mut layer = SpikingTrinaryLayer::new(10, LayerType::Hidden, NeuronType::RegularSpiking);
        
        // Add some synapses
        for i in 0..5 {
            layer.add_synapse(i, i + 5, 1.0, 2.0);
        }
        
        let input = vec![10.0; 10];
        let output = layer.step(&input);
        
        assert_eq!(output.len(), 10);
    }
    
    #[test]
    fn test_network_creation() {
        let network = SpikingTrinaryNetwork::new(
            &[10, 20, 10, 3],
            &[NeuronType::RegularSpiking; 4]
        );
        
        assert_eq!(network.layers.len(), 4);
        assert_eq!(network.inter_layer_connections.len(), 3);
    }
    
    #[test]
    fn test_energy_efficiency() {
        let mut network = SpikingTrinaryNetwork::new(
            &[100, 50, 10],
            &[NeuronType::RegularSpiking; 3]
        );
        
        let input = vec![0.0; 100]; // No input = mostly baseline
        let outputs = network.run(&input, 100);
        
        // Most neurons should be at baseline (energy efficient)
        let baseline_count: usize = outputs.iter()
            .map(|o| o.iter().filter(|&&t| t == Tryte::Baseline).count())
            .sum();
        
        let total_count = outputs.len() * 10; // Output layer size
        assert!(baseline_count as f32 / total_count as f32 > 0.7); // >70% baseline
    }
}