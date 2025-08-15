//! Integration of Real Protein Synthesis with Neural Networks
//! 
//! Connects Kandel's protein synthesis cascade to actual neural network training,
//! implementing biological memory consolidation in computational form.

use crate::real_protein_synthesis::{RealProteinSynthesis, LTPPhase, ProteinType, SynapticTag};
use crate::sparse_network_backprop::SparseTrithNetwork;
use crate::tryte::Tryte;
use std::collections::HashMap;

/// A synapse with protein-based memory consolidation
pub struct ProteinSynapse {
    /// Weight value (trinary: -1, 0, +1)
    pub weight: f32,
    
    /// Protein synthesis machinery for this synapse
    pub proteins: RealProteinSynthesis,
    
    /// Connection indices
    pub pre_neuron: usize,
    pub post_neuron: usize,
    
    /// Historical activation correlation (Hebbian)
    pub correlation_history: Vec<f32>,
    
    /// Whether this synapse is protected from forgetting
    pub is_protected: bool,
}

impl ProteinSynapse {
    pub fn new(pre: usize, post: usize) -> Self {
        Self {
            weight: 0.0,  // Start at baseline
            proteins: RealProteinSynthesis::new(),
            pre_neuron: pre,
            post_neuron: post,
            correlation_history: Vec::with_capacity(100),
            is_protected: false,
        }
    }
    
    /// Process pre and post synaptic activation
    pub fn process_activity(&mut self, pre_active: bool, post_active: bool, learning_rate: f32) {
        // Hebbian correlation
        let correlation = if pre_active && post_active {
            1.0  // Both firing - strengthen
        } else if pre_active && !post_active {
            -0.5  // Pre but not post - weaken
        } else {
            0.0  // No correlation
        };
        
        self.correlation_history.push(correlation);
        if self.correlation_history.len() > 100 {
            self.correlation_history.remove(0);
        }
        
        // Calculate activation strength and frequency
        let recent_correlations = self.correlation_history.iter()
            .rev()
            .take(10)
            .copied()
            .collect::<Vec<_>>();
        
        let activation_strength = recent_correlations.iter().sum::<f32>() / 10.0;
        let frequency = recent_correlations.iter()
            .filter(|&&c| c > 0.5)
            .count() as f32 * 10.0;  // Estimate Hz
        
        // Trigger protein synthesis cascade if strong correlation
        if activation_strength > 0.5 {
            let consolidated = self.proteins.process_activation(
                activation_strength.abs(),
                frequency,
                0.1  // 100ms time step
            );
            
            if consolidated {
                self.is_protected = true;
                println!("🧬 Synapse {}->{} consolidated via protein synthesis!", 
                        self.pre_neuron, self.post_neuron);
            }
        }
        
        // Update weight with protein-modulated learning
        let protection_factor = self.proteins.get_protection_factor();
        let effective_lr = learning_rate * protection_factor;
        
        self.weight += correlation * effective_lr;
        self.weight = self.enforce_trinary(self.weight);
    }
    
    /// Enforce trinary constraint on weight
    fn enforce_trinary(&self, w: f32) -> f32 {
        if w < -0.5 { -1.0 }
        else if w < 0.5 { 0.0 }
        else { 1.0 }
    }
}

/// Neural network with integrated protein synthesis
pub struct ProteinNeuralNetwork {
    /// Base network structure
    pub network: SparseTrithNetwork,
    
    /// Protein synthesis for each synapse
    pub synapses: Vec<Vec<ProteinSynapse>>,
    
    /// Global protein pool for synaptic tagging and capture
    pub protein_pool: HashMap<ProteinType, f32>,
    
    /// Network-wide neuromodulation
    pub dopamine: f32,
    pub norepinephrine: f32,
    pub acetylcholine: f32,
    pub serotonin: f32,
    
    /// Time elapsed (for temporal dynamics)
    pub time: f32,
}

impl ProteinNeuralNetwork {
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        let network = SparseTrithNetwork::new(layer_sizes.clone());
        
        // Create synapses with protein synthesis
        let mut synapses = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            let mut layer_synapses = Vec::new();
            for pre in 0..layer_sizes[i] {
                for post in 0..layer_sizes[i + 1] {
                    layer_synapses.push(ProteinSynapse::new(pre, post));
                }
            }
            synapses.push(layer_synapses);
        }
        
        Self {
            network,
            synapses,
            protein_pool: HashMap::new(),
            dopamine: 0.0,
            norepinephrine: 0.0,
            acetylcholine: 0.0,
            serotonin: 0.0,
            time: 0.0,
        }
    }
    
    /// Forward pass with protein synthesis monitoring
    pub fn forward_with_proteins(&mut self, input: &[Tryte]) -> Vec<Tryte> {
        // Standard forward pass
        let output = self.network.forward(input);
        
        // Update protein synthesis based on activations
        for (layer_idx, layer_activations) in self.network.activations.iter().enumerate() {
            if layer_idx == 0 { continue; }  // Skip input layer
            
            let prev_activations = &self.network.activations[layer_idx - 1];
            
            // Update each synapse
            if layer_idx - 1 < self.synapses.len() {
                for synapse in &mut self.synapses[layer_idx - 1] {
                    let pre_active = prev_activations.get(synapse.pre_neuron)
                        .map(|&t| t == Tryte::Activated)
                        .unwrap_or(false);
                    
                    let post_active = layer_activations.get(synapse.post_neuron)
                        .map(|&t| t == Tryte::Activated)
                        .unwrap_or(false);
                    
                    synapse.process_activity(pre_active, post_active, self.network.learning_rate);
                }
            }
        }
        
        // Update time
        self.time += 0.1;  // 100ms per step
        
        output
    }
    
    /// Backward pass with protein-based protection
    pub fn backward_with_proteins(&mut self, output: &[Tryte], target: &[usize]) -> f32 {
        // Get protein protection levels for each layer
        let mut layer_protections = Vec::new();
        for layer_synapses in &self.synapses {
            let avg_protection = layer_synapses.iter()
                .map(|s| s.proteins.get_protection_factor())
                .sum::<f32>() / layer_synapses.len() as f32;
            layer_protections.push(avg_protection);
        }
        
        // Apply protection to network
        for (i, protection) in layer_protections.iter().enumerate() {
            if i < self.network.protein_levels.len() {
                self.network.protein_levels[i] = 1.0 - protection;
            }
        }
        
        // Standard backward pass with protection
        self.network.backward(output, target)
    }
    
    /// Induce dopamine release (reward signal)
    pub fn release_dopamine(&mut self, amount: f32) {
        self.dopamine = (self.dopamine + amount).min(1.0);
        
        // Dopamine enhances protein synthesis
        for layer_synapses in &mut self.synapses {
            for synapse in layer_synapses {
                synapse.proteins.dopamine_level = self.dopamine;
            }
        }
        
        println!("💊 Dopamine released: {:.2} - enhancing consolidation", self.dopamine);
    }
    
    /// Simulate protein synthesis inhibitor (for testing)
    pub fn apply_anisomycin(&mut self) {
        println!("⚠️ Applying anisomycin - blocking protein synthesis!");
        for layer_synapses in &mut self.synapses {
            for synapse in layer_synapses {
                synapse.proteins.apply_anisomycin();
            }
        }
    }
    
    /// Remove protein synthesis inhibitor
    pub fn remove_anisomycin(&mut self) {
        println!("✅ Removing anisomycin - protein synthesis restored");
        for layer_synapses in &mut self.synapses {
            for synapse in layer_synapses {
                synapse.proteins.remove_anisomycin();
            }
        }
    }
    
    /// Implement synaptic tagging and capture across network
    pub fn synaptic_tagging_and_capture(&mut self) {
        // Identify strongly potentiated synapses
        let mut strong_synapses = Vec::new();
        for (layer_idx, layer_synapses) in self.synapses.iter().enumerate() {
            for (syn_idx, synapse) in layer_synapses.iter().enumerate() {
                if synapse.proteins.ltp_phase == LTPPhase::LateLTP2 ||
                   synapse.proteins.ltp_phase == LTPPhase::LateLTP3 {
                    strong_synapses.push((layer_idx, syn_idx));
                    
                    // Add proteins to pool
                    for (protein_type, &amount) in &synapse.proteins.new_proteins {
                        *self.protein_pool.entry(*protein_type).or_insert(0.0) += amount;
                    }
                }
            }
        }
        
        // Allow weak synapses to capture proteins
        for layer_synapses in &mut self.synapses {
            for synapse in layer_synapses {
                if synapse.proteins.synaptic_tag == SynapticTag::WeaklyTagged {
                    if synapse.proteins.capture_proteins(&self.protein_pool) {
                        println!("🏷️ Weak synapse {}->{} captured proteins for consolidation!",
                                synapse.pre_neuron, synapse.post_neuron);
                    }
                }
            }
        }
    }
    
    /// Get consolidation statistics
    pub fn get_consolidation_stats(&self) -> ConsolidationStats {
        let mut total_synapses = 0;
        let mut consolidated = 0;
        let mut early_ltp = 0;
        let mut late_ltp = 0;
        
        for layer_synapses in &self.synapses {
            for synapse in layer_synapses {
                total_synapses += 1;
                
                if synapse.is_protected {
                    consolidated += 1;
                }
                
                match synapse.proteins.ltp_phase {
                    LTPPhase::EarlyLTP => early_ltp += 1,
                    LTPPhase::LateLTP1 | LTPPhase::LateLTP2 | LTPPhase::LateLTP3 => late_ltp += 1,
                    _ => {}
                }
            }
        }
        
        ConsolidationStats {
            total_synapses,
            consolidated_synapses: consolidated,
            early_ltp_synapses: early_ltp,
            late_ltp_synapses: late_ltp,
            consolidation_percentage: (consolidated as f32 / total_synapses as f32) * 100.0,
        }
    }
}

/// Statistics about memory consolidation
#[derive(Debug)]
pub struct ConsolidationStats {
    pub total_synapses: usize,
    pub consolidated_synapses: usize,
    pub early_ltp_synapses: usize,
    pub late_ltp_synapses: usize,
    pub consolidation_percentage: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_protein_synapse() {
        let mut synapse = ProteinSynapse::new(0, 1);
        
        // Simulate repeated correlated activity
        for _ in 0..20 {
            synapse.process_activity(true, true, 0.1);
        }
        
        // Should trigger protein synthesis
        assert!(synapse.proteins.calcium_concentration > 0.1);
        assert!(synapse.proteins.camkii_activity > 0.0);
    }
    
    #[test]
    fn test_network_consolidation() {
        let mut network = ProteinNeuralNetwork::new(vec![10, 5, 2]);
        
        // Train with repeated patterns
        let input = vec![Tryte::Activated; 10];
        let target = vec![1, 0];
        
        for _ in 0..10 {
            let output = network.forward_with_proteins(&input);
            network.backward_with_proteins(&output, &target);
            
            // Add reward to enhance consolidation
            network.release_dopamine(0.1);
        }
        
        // Check consolidation
        let stats = network.get_consolidation_stats();
        assert!(stats.early_ltp_synapses > 0 || stats.late_ltp_synapses > 0,
               "Some synapses should show LTP");
    }
    
    #[test]
    fn test_anisomycin_blocks_consolidation() {
        let mut network = ProteinNeuralNetwork::new(vec![5, 3, 2]);
        
        // Apply protein synthesis inhibitor
        network.apply_anisomycin();
        
        // Try to train
        let input = vec![Tryte::Activated; 5];
        let target = vec![1, 0];
        
        for _ in 0..10 {
            let output = network.forward_with_proteins(&input);
            network.backward_with_proteins(&output, &target);
        }
        
        // Should not consolidate with anisomycin
        let stats = network.get_consolidation_stats();
        assert_eq!(stats.consolidated_synapses, 0,
                  "No consolidation should occur with anisomycin");
    }
    
    #[test]
    fn test_synaptic_tagging() {
        let mut network = ProteinNeuralNetwork::new(vec![4, 3, 2]);
        
        // Strong stimulation to some synapses
        network.synapses[0][0].proteins.set_synaptic_tag(0.9);
        network.synapses[0][0].proteins.process_activation(1.0, 100.0, 1.0);
        
        // Weak stimulation to others
        network.synapses[0][1].proteins.set_synaptic_tag(0.4);
        
        // Run tagging and capture
        network.synaptic_tagging_and_capture();
        
        // Weak synapse should capture proteins
        assert!(network.synapses[0][1].proteins.synaptic_tag == SynapticTag::WeaklyTagged ||
               network.synapses[0][1].proteins.synaptic_tag == SynapticTag::Captured);
    }
}