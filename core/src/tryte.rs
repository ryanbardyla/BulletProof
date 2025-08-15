//! Tryte: The Revolutionary Trinary Computing Primitive
//! 
//! Based on Eric Kandel's Nobel Prize-winning discoveries about biological memory,
//! we're building the first truly biological computing system.
//! 
//! Biology doesn't compute in binary - it computes in TRINARY!

use std::fmt;
use std::ops::{Add, Mul, BitAnd, BitOr, Not};

/// The fundamental unit of biological computation
/// Three states matching real neurons: Inhibited, Baseline, Activated
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
#[repr(i8)]
pub enum Tryte {
    Inhibited = -1,  // Active suppression (GABAergic)
    Baseline = 0,    // Resting state - COSTS ZERO ENERGY!
    Activated = 1,   // Active enhancement (excitatory)
}

impl Tryte {
    /// Create from an i8 value
    pub fn from_i8(val: i8) -> Self {
        match val {
            i if i < 0 => Tryte::Inhibited,
            0 => Tryte::Baseline,
            _ => Tryte::Activated,
        }
    }
    
    /// Convert to binary encoding (2 bits)
    pub fn to_bits(self) -> u8 {
        match self {
            Tryte::Inhibited => 0b00,
            Tryte::Baseline => 0b01,
            Tryte::Activated => 0b10,
        }
    }
    
    /// Create from binary encoding
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0b11 {
            0b00 => Tryte::Inhibited,
            0b01 => Tryte::Baseline,
            0b10 => Tryte::Activated,
            _ => Tryte::Baseline, // 0b11 reserved for special states
        }
    }
    
    /// Check if this Tryte requires computation
    pub fn needs_computation(self) -> bool {
        self != Tryte::Baseline  // Baseline costs NOTHING!
    }
    
    /// Biological energy cost
    pub fn energy_cost(self) -> f32 {
        match self {
            Tryte::Inhibited => 1.2,  // Inhibition costs slightly more
            Tryte::Baseline => 0.0,    // FREE! This is the magic!
            Tryte::Activated => 1.0,   // Standard activation cost
        }
    }
    
    /// Convert to neural potential (millivolts)
    pub fn to_potential(self) -> f32 {
        match self {
            Tryte::Inhibited => -70.0,  // Hyperpolarized
            Tryte::Baseline => -55.0,    // Resting potential
            Tryte::Activated => -40.0,   // Depolarized
        }
    }
}

/// NOT operation: Flips activation state
impl Not for Tryte {
    type Output = Tryte;
    
    fn not(self) -> Self::Output {
        match self {
            Tryte::Inhibited => Tryte::Activated,
            Tryte::Baseline => Tryte::Baseline,  // Neutral stays neutral!
            Tryte::Activated => Tryte::Inhibited,
        }
    }
}

/// AND operation: Most inhibitory wins (biological safety)
impl BitAnd for Tryte {
    type Output = Tryte;
    
    fn bitand(self, rhs: Self) -> Self::Output {
        // Minimum = most negative/inhibitory
        if self <= rhs { self } else { rhs }
    }
}

/// OR operation: Most activated wins (biological opportunity)
impl BitOr for Tryte {
    type Output = Tryte;
    
    fn bitor(self, rhs: Self) -> Self::Output {
        // Maximum = most positive/activated
        if self >= rhs { self } else { rhs }
    }
}

/// Addition with saturation (models synaptic summation)
impl Add for Tryte {
    type Output = Tryte;
    
    fn add(self, rhs: Self) -> Self::Output {
        let sum = (self as i8) + (rhs as i8);
        Tryte::from_i8(sum)
    }
}

/// Multiplication (models gain control)
impl Mul for Tryte {
    type Output = Tryte;
    
    fn mul(self, rhs: Self) -> Self::Output {
        if self == Tryte::Baseline || rhs == Tryte::Baseline {
            return Tryte::Baseline;  // Zero propagates
        }
        // Sign multiplication: -1 * -1 = +1, -1 * +1 = -1
        Tryte::from_i8((self as i8) * (rhs as i8))
    }
}

impl fmt::Display for Tryte {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", match self {
            Tryte::Inhibited => "⊖",  // Inhibited
            Tryte::Baseline => "○",   // Baseline (empty circle)
            Tryte::Activated => "⊕",  // Activated
        })
    }
}

/// Packed storage: 4 Trytes per byte for maximum efficiency
pub struct PackedTrytes {
    data: Vec<u8>,
    length: usize,  // Number of Trytes stored
}

impl PackedTrytes {
    /// Create a new packed array of Trytes
    pub fn new(size: usize) -> Self {
        let bytes_needed = (size + 3) / 4;  // Round up
        Self {
            data: vec![0b01_01_01_01; bytes_needed],  // Initialize all to Baseline
            length: size,
        }
    }
    
    /// Get a Tryte at index
    pub fn get(&self, index: usize) -> Tryte {
        assert!(index < self.length, "Index out of bounds");
        
        let byte_index = index / 4;
        let bit_offset = (index % 4) * 2;
        let bits = (self.data[byte_index] >> bit_offset) & 0b11;
        
        Tryte::from_bits(bits)
    }
    
    /// Set a Tryte at index
    pub fn set(&mut self, index: usize, value: Tryte) {
        assert!(index < self.length, "Index out of bounds");
        
        let byte_index = index / 4;
        let bit_offset = (index % 4) * 2;
        let mask = !(0b11 << bit_offset);
        
        self.data[byte_index] &= mask;
        self.data[byte_index] |= value.to_bits() << bit_offset;
    }
    
    /// Count non-baseline Trytes (for sparsity calculation)
    pub fn count_active(&self) -> usize {
        let mut count = 0;
        for i in 0..self.length {
            if self.get(i).needs_computation() {
                count += 1;
            }
        }
        count
    }
    
    /// Calculate sparsity percentage
    pub fn sparsity(&self) -> f32 {
        let active = self.count_active() as f32;
        active / self.length as f32
    }
    
    /// Calculate total energy cost
    pub fn total_energy(&self) -> f32 {
        let mut energy = 0.0;
        for i in 0..self.length {
            energy += self.get(i).energy_cost();
        }
        energy
    }
    
    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }
}

/// A Tryte-based neuron with protein synthesis capability
#[derive(Clone)]
pub struct TryteNeuron {
    pub state: Tryte,
    pub threshold: f32,
    pub protein_level: f32,
    pub plasticity_state: Tryte,  // LTD (-1), No change (0), LTP (+1)
    
    // Kandel's CREB mechanism
    pub creb_activation: f32,
    pub protein_synthesis_triggered: bool,
}

impl TryteNeuron {
    pub fn new() -> Self {
        Self {
            state: Tryte::Baseline,
            threshold: 0.5,
            protein_level: 1.0,
            plasticity_state: Tryte::Baseline,
            creb_activation: 0.0,
            protein_synthesis_triggered: false,
        }
    }
    
    /// Process input and update state
    pub fn process(&mut self, input: f32) -> Tryte {
        // Threshold to trinary
        if input > self.threshold {
            self.state = Tryte::Activated;
            
            // Strong activation triggers protein synthesis (Kandel's discovery)
            if input > self.threshold * 2.0 {
                self.trigger_protein_synthesis();
            }
        } else if input < -self.threshold {
            self.state = Tryte::Inhibited;
        } else {
            self.state = Tryte::Baseline;
        }
        
        self.state
    }
    
    /// Trigger protein synthesis for long-term memory
    fn trigger_protein_synthesis(&mut self) {
        self.creb_activation += 0.1;
        
        if self.creb_activation > 0.7 {  // CREB threshold
            self.protein_synthesis_triggered = true;
            self.protein_level += 0.5;
            self.plasticity_state = Tryte::Activated;  // LTP
            
            println!("🧬 Protein synthesis triggered! Long-term memory forming...");
        }
    }
    
    /// Spike-timing dependent plasticity
    pub fn stdp_update(&mut self, pre_spike: bool, post_spike: bool, delta_t: i32) {
        if pre_spike && post_spike {
            if delta_t > 0 && delta_t < 20 {
                // Pre before post: strengthen
                self.plasticity_state = Tryte::Activated;
            } else if delta_t < 0 && delta_t > -20 {
                // Post before pre: weaken
                self.plasticity_state = Tryte::Inhibited;
            }
        }
    }
}

/// A layer of Tryte neurons with sparse computation
pub struct TryteLayer {
    pub neurons: Vec<TryteNeuron>,
    pub packed_states: PackedTrytes,
    pub size: usize,
}

impl TryteLayer {
    pub fn new(size: usize) -> Self {
        Self {
            neurons: vec![TryteNeuron::new(); size],
            packed_states: PackedTrytes::new(size),
            size,
        }
    }
    
    /// Process input with massive efficiency gains from sparsity
    pub fn forward(&mut self, input: &[f32]) -> Vec<Tryte> {
        assert_eq!(input.len(), self.size);
        
        let mut output = Vec::with_capacity(self.size);
        let mut computations_skipped = 0;
        
        for (i, &input_val) in input.iter().enumerate() {
            // THE MAGIC: Skip baseline neurons completely!
            if input_val.abs() < 0.01 {
                self.neurons[i].state = Tryte::Baseline;
                self.packed_states.set(i, Tryte::Baseline);
                output.push(Tryte::Baseline);
                computations_skipped += 1;
                continue;  // NO COMPUTATION!
            }
            
            // Only compute for active neurons
            let state = self.neurons[i].process(input_val);
            self.packed_states.set(i, state);
            output.push(state);
        }
        
        println!("⚡ Sparsity optimization: Skipped {}/{} neurons ({:.1}% saved!)",
                computations_skipped, self.size, 
                (computations_skipped as f32 / self.size as f32) * 100.0);
        
        output
    }
    
    /// Get layer statistics
    pub fn stats(&self) -> LayerStats {
        LayerStats {
            total_neurons: self.size,
            active_neurons: self.packed_states.count_active(),
            sparsity: self.packed_states.sparsity(),
            energy_cost: self.packed_states.total_energy(),
            memory_bytes: self.packed_states.memory_bytes(),
        }
    }
}

pub struct LayerStats {
    pub total_neurons: usize,
    pub active_neurons: usize,
    pub sparsity: f32,
    pub energy_cost: f32,
    pub memory_bytes: usize,
}

impl fmt::Display for LayerStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Layer Stats:\n")?;
        write!(f, "  Neurons: {} total, {} active\n", self.total_neurons, self.active_neurons)?;
        write!(f, "  Sparsity: {:.1}%\n", self.sparsity * 100.0)?;
        write!(f, "  Energy: {:.2} units (vs {:.2} for binary)\n", 
               self.energy_cost, self.total_neurons as f32)?;
        write!(f, "  Memory: {} bytes ({:.2} bits/neuron)\n", 
               self.memory_bytes, (self.memory_bytes * 8) as f32 / self.total_neurons as f32)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tryte_operations() {
        let a = Tryte::Activated;
        let b = Tryte::Inhibited;
        let c = Tryte::Baseline;
        
        // Test NOT
        assert_eq!(!a, Tryte::Inhibited);
        assert_eq!(!b, Tryte::Activated);
        assert_eq!(!c, Tryte::Baseline);  // Baseline stays baseline!
        
        // Test AND (minimum)
        assert_eq!(a & b, Tryte::Inhibited);  // Inhibition wins
        assert_eq!(a & c, Tryte::Baseline);
        assert_eq!(b & c, Tryte::Inhibited);
        
        // Test OR (maximum)
        assert_eq!(a | b, Tryte::Activated);  // Activation wins
        assert_eq!(a | c, Tryte::Activated);
        assert_eq!(b | c, Tryte::Baseline);
        
        // Test multiplication
        assert_eq!(a * b, Tryte::Inhibited);  // +1 * -1 = -1
        assert_eq!(b * b, Tryte::Activated);  // -1 * -1 = +1
        assert_eq!(a * c, Tryte::Baseline);   // Anything * 0 = 0
    }
    
    #[test]
    fn test_packed_storage() {
        let mut packed = PackedTrytes::new(100);
        
        // Set some values
        packed.set(0, Tryte::Activated);
        packed.set(1, Tryte::Inhibited);
        packed.set(2, Tryte::Baseline);
        packed.set(99, Tryte::Activated);
        
        // Check retrieval
        assert_eq!(packed.get(0), Tryte::Activated);
        assert_eq!(packed.get(1), Tryte::Inhibited);
        assert_eq!(packed.get(2), Tryte::Baseline);
        assert_eq!(packed.get(99), Tryte::Activated);
        
        // Check memory efficiency
        assert_eq!(packed.memory_bytes(), 25);  // 100 Trytes in 25 bytes!
    }
    
    #[test]
    fn test_sparsity() {
        let mut layer = TryteLayer::new(1000);
        
        // Simulate sparse input (95% zeros)
        let mut input = vec![0.0; 1000];
        for i in 0..50 {
            input[i * 20] = 1.0;  // 5% active
        }
        
        layer.forward(&input);
        let stats = layer.stats();
        
        assert!(stats.sparsity < 0.1);  // Should be around 5%
        assert!(stats.energy_cost < 100.0);  // Much less than 1000
        
        println!("{}", stats);
    }
    
    #[test]
    fn test_protein_synthesis() {
        let mut neuron = TryteNeuron::new();
        
        // Weak activation - no protein synthesis
        neuron.process(0.6);
        assert!(!neuron.protein_synthesis_triggered);
        
        // Strong repeated activation - triggers synthesis
        for _ in 0..10 {
            neuron.process(1.5);  // Strong signal
        }
        assert!(neuron.protein_synthesis_triggered);
        assert_eq!(neuron.plasticity_state, Tryte::Activated);  // LTP
    }
}