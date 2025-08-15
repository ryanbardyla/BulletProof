// 🧬 BIOLOGICAL NEURON IMPLEMENTATION
// Scientifically accurate simulation based on Hodgkin-Huxley model

use std::f64::consts::E;

/// Neurotransmitter types for synaptic transmission
#[derive(Clone, Copy, Debug)]
pub enum Neurotransmitter {
    Glutamate,    // Primary excitatory
    GABA,         // Primary inhibitory  
    Dopamine,     // Modulatory
    Serotonin,    // Modulatory
    Acetylcholine,// Modulatory
}

/// Biologically accurate neuron model
pub struct BiologicalNeuron {
    // Membrane dynamics (Hodgkin-Huxley model)
    pub membrane_potential: f64,     // Voltage in mV (-70 to +40)
    pub threshold: f64,               // Spike threshold (typically -55 mV)
    
    // Ion channel gating variables
    sodium_activation: f64,          // m gate (0 to 1)
    sodium_inactivation: f64,        // h gate (0 to 1)
    potassium_activation: f64,        // n gate (0 to 1)
    
    // Ion conductances (in mS/cm²)
    g_na: f64,  // Sodium conductance
    g_k: f64,   // Potassium conductance
    g_l: f64,   // Leak conductance
    
    // Reversal potentials (in mV)
    e_na: f64,  // Sodium reversal potential
    e_k: f64,   // Potassium reversal potential
    e_l: f64,   // Leak reversal potential
    
    // Synaptic inputs
    excitatory_conductance: f64,     // Total excitatory input
    inhibitory_conductance: f64,     // Total inhibitory input
    
    // Calcium dynamics (for plasticity)
    calcium_concentration: f64,      // Internal Ca2+ concentration
    
    // Refractory period
    refractory_timer: f64,           // Time since last spike
    absolute_refractory: f64,        // Absolute refractory period (2 ms)
    
    // Plasticity and adaptation
    pub last_spike_time: f64,        // For STDP calculations
    spike_count: u32,                // Total spikes
    adaptation_current: f64,         // Spike frequency adaptation
    
    // Metabolic state
    atp_level: f64,                  // Energy availability
    
    // Injected current (for experiments)
    injected_current: f64,
}

impl BiologicalNeuron {
    pub fn new() -> Self {
        BiologicalNeuron {
            // Resting potential
            membrane_potential: -65.0,
            threshold: -55.0,
            
            // Initial gate states (resting values)
            sodium_activation: 0.05,
            sodium_inactivation: 0.6,
            potassium_activation: 0.32,
            
            // Standard conductances (Hodgkin-Huxley squid axon)
            g_na: 120.0,
            g_k: 36.0,
            g_l: 0.3,
            
            // Reversal potentials
            e_na: 50.0,
            e_k: -77.0,
            e_l: -54.4,
            
            // No synaptic input initially
            excitatory_conductance: 0.0,
            inhibitory_conductance: 0.0,
            
            // Calcium and plasticity
            calcium_concentration: 0.0001,  // Low resting Ca2+
            
            // Refractory state
            refractory_timer: 0.0,
            absolute_refractory: 2.0,  // 2 ms
            
            // History
            last_spike_time: -1000.0,
            spike_count: 0,
            adaptation_current: 0.0,
            
            // Full energy
            atp_level: 1.0,
            
            injected_current: 0.0,
        }
    }
    
    /// Inject external current (for experiments)
    pub fn inject_current(&mut self, current: f64) {
        self.injected_current = current;
    }
    
    /// Receive synaptic input
    pub fn receive_synapse(&mut self, neurotransmitter: Neurotransmitter, strength: f64) {
        match neurotransmitter {
            Neurotransmitter::Glutamate => {
                // Excitatory - increases conductance
                self.excitatory_conductance += strength;
                // Trigger calcium influx
                self.calcium_concentration += strength * 0.001;
            },
            Neurotransmitter::GABA => {
                // Inhibitory - increases inhibitory conductance
                self.inhibitory_conductance += strength;
            },
            Neurotransmitter::Dopamine => {
                // Modulatory - affects threshold
                self.threshold -= strength * 0.1;
            },
            Neurotransmitter::Serotonin => {
                // Modulatory - affects adaptation
                self.adaptation_current *= 1.0 - strength * 0.1;
            },
            Neurotransmitter::Acetylcholine => {
                // Modulatory - affects potassium channels
                self.g_k *= 1.0 - strength * 0.05;
            },
        }
    }
    
    /// Step the neuron simulation by dt seconds
    pub fn step(&mut self, dt: f64) -> bool {
        // Check if in absolute refractory period
        if self.refractory_timer > 0.0 {
            self.refractory_timer -= dt * 1000.0;  // Convert to ms
            return false;
        }
        
        // Update ion channel gates
        self.update_gates(dt);
        
        // Calculate currents
        let i_na = self.sodium_current();
        let i_k = self.potassium_current();
        let i_l = self.leak_current();
        let i_syn = self.synaptic_current();
        let i_adapt = self.adaptation_current;
        
        // Total membrane current
        let i_total = self.injected_current - (i_na + i_k + i_l + i_syn + i_adapt);
        
        // Update membrane potential (Euler integration)
        let capacitance = 1.0;  // μF/cm²
        self.membrane_potential += (i_total / capacitance) * dt * 1000.0;  // dt in ms
        
        // Decay synaptic conductances
        self.excitatory_conductance *= E.powf(-dt / 0.005);  // 5 ms decay
        self.inhibitory_conductance *= E.powf(-dt / 0.010);  // 10 ms decay
        
        // Update calcium (decay)
        self.calcium_concentration *= E.powf(-dt / 0.100);  // 100 ms decay
        
        // Update adaptation current
        self.adaptation_current *= E.powf(-dt / 0.100);  // 100 ms decay
        
        // Check for spike
        let spiked = if self.membrane_potential >= self.threshold {
            // Spike!
            self.spike_count += 1;
            self.last_spike_time = self.spike_count as f64 * dt;
            
            // Reset membrane potential
            self.membrane_potential = -65.0;
            
            // Enter refractory period
            self.refractory_timer = self.absolute_refractory;
            
            // Increase adaptation current
            self.adaptation_current += 0.1;
            
            // Calcium spike
            self.calcium_concentration += 0.01;
            
            // Use ATP
            self.atp_level *= 0.999;
            
            true
        } else {
            false
        };
        
        // Restore ATP slowly
        self.atp_level = (self.atp_level + 0.0001).min(1.0);
        
        // Clear injected current
        self.injected_current = 0.0;
        
        spiked
    }
    
    /// Update ion channel gating variables
    fn update_gates(&mut self, dt: f64) {
        let v = self.membrane_potential;
        
        // Sodium activation (m gate)
        let alpha_m = 0.1 * (v + 40.0) / (1.0 - E.powf(-(v + 40.0) / 10.0));
        let beta_m = 4.0 * E.powf(-(v + 65.0) / 18.0);
        let m_inf = alpha_m / (alpha_m + beta_m);
        let tau_m = 1.0 / (alpha_m + beta_m);
        self.sodium_activation += (m_inf - self.sodium_activation) * dt / tau_m;
        
        // Sodium inactivation (h gate)
        let alpha_h = 0.07 * E.powf(-(v + 65.0) / 20.0);
        let beta_h = 1.0 / (1.0 + E.powf(-(v + 35.0) / 10.0));
        let h_inf = alpha_h / (alpha_h + beta_h);
        let tau_h = 1.0 / (alpha_h + beta_h);
        self.sodium_inactivation += (h_inf - self.sodium_inactivation) * dt / tau_h;
        
        // Potassium activation (n gate)
        let alpha_n = 0.01 * (v + 55.0) / (1.0 - E.powf(-(v + 55.0) / 10.0));
        let beta_n = 0.125 * E.powf(-(v + 65.0) / 80.0);
        let n_inf = alpha_n / (alpha_n + beta_n);
        let tau_n = 1.0 / (alpha_n + beta_n);
        self.potassium_activation += (n_inf - self.potassium_activation) * dt / tau_n;
    }
    
    /// Calculate sodium current
    fn sodium_current(&self) -> f64 {
        let m3h = self.sodium_activation.powi(3) * self.sodium_inactivation;
        self.g_na * m3h * (self.membrane_potential - self.e_na)
    }
    
    /// Calculate potassium current
    fn potassium_current(&self) -> f64 {
        let n4 = self.potassium_activation.powi(4);
        self.g_k * n4 * (self.membrane_potential - self.e_k)
    }
    
    /// Calculate leak current
    fn leak_current(&self) -> f64 {
        self.g_l * (self.membrane_potential - self.e_l)
    }
    
    /// Calculate synaptic current
    fn synaptic_current(&self) -> f64 {
        // Excitatory reversal at 0 mV, inhibitory at -70 mV
        let i_exc = self.excitatory_conductance * (self.membrane_potential - 0.0);
        let i_inh = self.inhibitory_conductance * (self.membrane_potential + 70.0);
        i_exc + i_inh
    }
    
    /// Check if neuron is currently spiking
    pub fn is_spiking(&self) -> bool {
        self.refractory_timer > (self.absolute_refractory - 0.5)
    }
    
    /// Get spike rate (Hz)
    pub fn spike_rate(&self) -> f64 {
        if self.last_spike_time > 0.0 {
            self.spike_count as f64 / self.last_spike_time
        } else {
            0.0
        }
    }
    
    /// Get current metabolic cost
    pub fn metabolic_cost(&self) -> f64 {
        // Cost based on ion pumping needed
        let pump_cost = (self.membrane_potential + 65.0).abs() / 100.0;
        let spike_cost = if self.is_spiking() { 0.1 } else { 0.0 };
        pump_cost + spike_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_resting_potential() {
        let mut neuron = BiologicalNeuron::new();
        
        // Should stay near resting potential without input
        for _ in 0..1000 {
            neuron.step(0.001);
        }
        
        assert!((neuron.membrane_potential + 65.0).abs() < 5.0);
    }
    
    #[test]
    fn test_spike_generation() {
        let mut neuron = BiologicalNeuron::new();
        
        // Strong current injection should cause spike
        neuron.inject_current(10.0);
        let mut spiked = false;
        
        for _ in 0..100 {
            if neuron.step(0.001) {
                spiked = true;
                break;
            }
        }
        
        assert!(spiked);
    }
    
    #[test]
    fn test_refractory_period() {
        let mut neuron = BiologicalNeuron::new();
        
        // Cause first spike
        neuron.inject_current(20.0);
        while !neuron.step(0.001) {}
        
        // Try to spike immediately after - should fail due to refractory
        neuron.inject_current(20.0);
        let second_spike = neuron.step(0.001);
        
        assert!(!second_spike);
    }
}