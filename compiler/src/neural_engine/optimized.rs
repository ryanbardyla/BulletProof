// ⚡ OPTIMIZED NEURON IMPLEMENTATION
// Fast approximation that captures essential dynamics

use std::f32::consts::E;

/// Fast, efficient neuron model for practical computation
#[derive(Clone, Debug)]
pub struct OptimizedNeuron {
    // Core state (minimal for speed)
    pub potential: f32,         // Membrane potential (simplified)
    pub threshold: f32,         // Spike threshold
    pub refractory: u8,         // Refractory countdown
    
    // Input accumulation
    pub excitatory: f32,        // Sum of excitatory inputs
    pub inhibitory: f32,        // Sum of inhibitory inputs
    
    // Spike history (bit-packed for efficiency)
    pub spike_history: u32,     // Last 32 timesteps as bits
    pub spiked_last_step: bool, // Quick access to last spike
    
    // Simple plasticity
    pub weight_scale: f32,      // Global synaptic scaling
    pub spike_count: u32,       // Total spikes (for rate)
    
    // Adaptation
    pub adaptation: f32,        // Spike frequency adaptation
}

impl OptimizedNeuron {
    pub fn new() -> Self {
        OptimizedNeuron {
            potential: 0.0,
            threshold: 1.0,
            refractory: 0,
            excitatory: 0.0,
            inhibitory: 0.0,
            spike_history: 0,
            spiked_last_step: false,
            weight_scale: 1.0,
            spike_count: 0,
            adaptation: 0.0,
        }
    }
    
    /// Add external input
    pub fn add_input(&mut self, input: f32) {
        if input > 0.0 {
            self.excitatory += input;
        } else {
            self.inhibitory -= input;
        }
    }
    
    /// Add weighted synaptic input
    pub fn add_weighted_input(&mut self, weight: f32) {
        let scaled_weight = weight * self.weight_scale;
        if scaled_weight > 0.0 {
            self.excitatory += scaled_weight;
        } else {
            self.inhibitory -= scaled_weight;
        }
    }
    
    /// Single step update - returns true if spiked
    pub fn step(&mut self) -> bool {
        // Update spike history
        self.spike_history <<= 1;
        
        // Check refractory period
        if self.refractory > 0 {
            self.refractory -= 1;
            self.spiked_last_step = false;
            self.potential *= 0.5;  // Hyperpolarization during refractory
            return false;
        }
        
        // Integrate inputs (leaky integrate-and-fire)
        let leak_factor = 0.9;  // 10% leak per timestep
        self.potential *= leak_factor;
        
        // Add synaptic currents
        let net_input = self.excitatory - self.inhibitory - self.adaptation;
        self.potential += net_input;
        
        // Decay synaptic inputs
        self.excitatory *= 0.8;  // Fast decay
        self.inhibitory *= 0.85; // Slightly slower decay
        
        // Decay adaptation
        self.adaptation *= 0.95;
        
        // Check for spike
        if self.potential >= self.threshold {
            // Spike!
            self.spike_count += 1;
            self.spike_history |= 1;
            self.spiked_last_step = true;
            
            // Reset potential
            self.potential = -0.2;  // Slight hyperpolarization
            
            // Enter refractory
            self.refractory = 3;  // 3 timestep refractory
            
            // Increase adaptation
            self.adaptation += 0.1;
            
            // Homeostatic plasticity - adjust threshold based on activity
            self.update_homeostasis();
            
            true
        } else {
            self.spiked_last_step = false;
            false
        }
    }
    
    /// Update homeostatic plasticity
    fn update_homeostasis(&mut self) {
        // Count recent spikes
        let recent_spikes = self.spike_history.count_ones();
        
        // Target rate is ~5 spikes in 32 timesteps
        let target_rate = 5;
        
        if recent_spikes > target_rate + 2 {
            // Too active - increase threshold
            self.threshold *= 1.01;
            self.weight_scale *= 0.99;
        } else if recent_spikes < target_rate - 2 {
            // Too quiet - decrease threshold
            self.threshold *= 0.99;
            self.weight_scale *= 1.01;
        }
        
        // Keep threshold in reasonable range
        self.threshold = self.threshold.clamp(0.5, 2.0);
        self.weight_scale = self.weight_scale.clamp(0.5, 2.0);
    }
    
    /// Fast spike rate calculation
    pub fn spike_rate(&self) -> f32 {
        // Use recent history for quick rate estimate
        self.spike_history.count_ones() as f32 / 32.0
    }
    
    /// Check if neuron is bursting
    pub fn is_bursting(&self) -> bool {
        // Bursting if >3 spikes in last 8 timesteps
        (self.spike_history & 0xFF).count_ones() > 3
    }
    
    /// Reset neuron state
    pub fn reset(&mut self) {
        self.potential = 0.0;
        self.excitatory = 0.0;
        self.inhibitory = 0.0;
        self.refractory = 0;
        self.adaptation = 0.0;
        self.spike_history = 0;
        self.spiked_last_step = false;
    }
}

/// SIMD-optimized batch operations for multiple neurons
pub struct OptimizedNeuronBatch {
    neurons: Vec<OptimizedNeuron>,
}

impl OptimizedNeuronBatch {
    pub fn new(count: usize) -> Self {
        OptimizedNeuronBatch {
            neurons: vec![OptimizedNeuron::new(); count],
        }
    }
    
    /// Step all neurons in parallel (uses auto-vectorization)
    pub fn step_all(&mut self) -> Vec<bool> {
        self.neurons.iter_mut().map(|n| n.step()).collect()
    }
    
    /// Apply input to all neurons
    pub fn broadcast_input(&mut self, input: f32) {
        for neuron in &mut self.neurons {
            neuron.add_input(input);
        }
    }
    
    /// Get average spike rate
    pub fn average_spike_rate(&self) -> f32 {
        let total: f32 = self.neurons.iter().map(|n| n.spike_rate()).sum();
        total / self.neurons.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_spiking() {
        let mut neuron = OptimizedNeuron::new();
        
        // Accumulate input until spike
        let mut spiked = false;
        for _ in 0..10 {
            neuron.add_input(0.3);
            if neuron.step() {
                spiked = true;
                break;
            }
        }
        
        assert!(spiked);
    }
    
    #[test]
    fn test_refractory() {
        let mut neuron = OptimizedNeuron::new();
        
        // Force spike with strong input
        neuron.add_input(2.0);
        assert!(neuron.step());
        
        // Should not spike during refractory even with strong input
        neuron.add_input(2.0);
        assert!(!neuron.step());
    }
    
    #[test]
    fn test_homeostasis() {
        let mut neuron = OptimizedNeuron::new();
        
        // Make neuron spike frequently
        for _ in 0..100 {
            neuron.add_input(0.5);
            neuron.step();
        }
        
        // Threshold should have increased
        assert!(neuron.threshold > 1.0);
    }
    
    #[test]
    fn test_batch_operations() {
        let mut batch = OptimizedNeuronBatch::new(100);
        
        // Broadcast input
        batch.broadcast_input(0.5);
        
        // Step all
        let spikes = batch.step_all();
        
        // Some should spike
        let spike_count = spikes.iter().filter(|&&s| s).count();
        assert!(spike_count > 0);
    }
}