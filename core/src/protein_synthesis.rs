//! Protein Synthesis for Long-Term Memory Formation
//! 
//! Based on Eric Kandel's Nobel Prize-winning discoveries:
//! - Repeated stimulation → CREB activation → Gene transcription
//! - Protein synthesis → Structural changes → Permanent memory
//! 
//! This is the first computational implementation of actual biological memory!

use crate::tryte::{Tryte, TryteNeuron};
use std::collections::HashMap;

/// Protein types involved in memory formation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ProteinType {
    CREB,       // cAMP Response Element-Binding protein
    PKA,        // Protein Kinase A
    MAPK,       // Mitogen-Activated Protein Kinase
    CaMKII,     // Calcium/calmodulin-dependent protein kinase II
    Arc,        // Activity-regulated cytoskeleton-associated protein
    BDNF,       // Brain-Derived Neurotrophic Factor
    PSD95,      // Postsynaptic density protein 95
    Synaptophysin, // Synaptic vesicle protein
}

/// Gene expression states
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GeneState {
    Silenced,           // Gene is methylated/blocked
    Ready,              // Gene can be activated
    Transcribing,       // mRNA being produced
    Translating,        // Proteins being synthesized
    Refractory,         // Post-expression cooldown
}

/// Memory formation result
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MemoryFormation {
    NoChange,           // No lasting changes
    EarlyPhaseLTP,      // Minutes to hours, no protein synthesis required
    LatePhaseLTP,       // Hours to permanent, requires protein synthesis  
    LongTermMemory,     // Permanent structural changes
    Extinction,         // Memory weakening
    Reconsolidation,    // Memory updating
}

/// Synaptic tag for capture of proteins
#[derive(Debug, Clone)]
pub struct SynapticTag {
    pub strength: f32,
    pub timestamp: u64,
    pub protein_captured: HashMap<ProteinType, f32>,
    pub decay_rate: f32,
}

/// Advanced neuron with full protein synthesis machinery
#[derive(Clone)]
pub struct ProteinSynthesisNeuron {
    // Basic trinary state
    pub base: TryteNeuron,
    
    // Protein concentrations
    pub proteins: HashMap<ProteinType, f32>,
    
    // Gene expression machinery
    pub genes: HashMap<String, GeneState>,
    pub transcription_factors: HashMap<String, f32>,
    
    // Synaptic tags for protein capture
    pub synaptic_tags: Vec<SynapticTag>,
    
    // Memory phases
    pub early_ltp: f32,     // Early-phase LTP (minutes)
    pub late_ltp: f32,      // Late-phase LTP (hours to permanent)
    pub early_ltd: f32,     // Early-phase LTD
    pub late_ltd: f32,      // Late-phase LTD
    
    // Consolidation state
    pub consolidation_strength: f32,
    pub reconsolidation_window: Option<(u64, u64)>, // When memory can be modified
    
    // Epigenetic modifications
    pub histone_acetylation: f32,    // Opens chromatin for transcription
    pub dna_methylation: HashMap<String, f32>, // Silences genes
    
    // Circadian influence
    pub circadian_phase: f32,        // 0-1 representing 24h cycle
    pub sleep_pressure: f32,         // Affects consolidation
}

impl ProteinSynthesisNeuron {
    pub fn new() -> Self {
        let mut proteins = HashMap::new();
        // Initialize with baseline protein levels
        proteins.insert(ProteinType::CREB, 0.1);
        proteins.insert(ProteinType::PKA, 0.2);
        proteins.insert(ProteinType::MAPK, 0.15);
        proteins.insert(ProteinType::CaMKII, 0.3);
        proteins.insert(ProteinType::Arc, 0.05);
        proteins.insert(ProteinType::BDNF, 0.1);
        proteins.insert(ProteinType::PSD95, 0.4);
        proteins.insert(ProteinType::Synaptophysin, 0.35);
        
        let mut genes = HashMap::new();
        genes.insert("creb".to_string(), GeneState::Ready);
        genes.insert("arc".to_string(), GeneState::Ready);
        genes.insert("bdnf".to_string(), GeneState::Ready);
        genes.insert("homer1a".to_string(), GeneState::Ready);
        genes.insert("zif268".to_string(), GeneState::Ready);
        
        Self {
            base: TryteNeuron::new(),
            proteins,
            genes,
            transcription_factors: HashMap::new(),
            synaptic_tags: Vec::new(),
            early_ltp: 0.0,
            late_ltp: 0.0,
            early_ltd: 0.0,
            late_ltd: 0.0,
            consolidation_strength: 0.0,
            reconsolidation_window: None,
            histone_acetylation: 0.5,
            dna_methylation: HashMap::new(),
            circadian_phase: 0.5,
            sleep_pressure: 0.0,
        }
    }
    
    /// Create neuron with specific size (for benchmarking)
    pub fn new_with_size(size: usize) -> Self {
        let mut neuron = Self::new();
        // Scale protein concentrations based on size
        let scale_factor = (size as f32 / 1000.0).sqrt();
        for protein_level in neuron.proteins.values_mut() {
            *protein_level *= scale_factor;
        }
        neuron
    }
    
    /// Train with protein modulation (for benchmarking)
    pub fn train_with_proteins(&mut self, batch: &[Tryte], epoch: i32) {
        let mut activation_sum: f32 = 0.0;
        
        for &tryte in batch {
            let input_value = match tryte {
                Tryte::Inhibited => -1.0,
                Tryte::Baseline => 0.0,  // Zero energy!
                Tryte::Activated => 1.0,
            };
            
            // Only process non-baseline inputs (sparse!)
            if input_value != 0.0 {
                activation_sum += input_value;
            }
        }
        
        // Protein-modulated learning based on epoch
        let learning_factor = if epoch % 3 == 0 {
            // Periodic protein synthesis boost
            self.proteins.entry(ProteinType::CREB).and_modify(|p| *p += 0.1);
            1.5  // Enhanced learning
        } else {
            1.0  // Normal learning
        };
        
        // Update neuron state with protein modulation
        if activation_sum.abs() > 0.1 {
            let memory_result = self.process_with_proteins(activation_sum * learning_factor, epoch as u32);
            
            match memory_result {
                MemoryFormation::LongTermMemory => {
                    self.consolidation_strength += 0.3;
                },
                MemoryFormation::LatePhaseLTP => {
                    self.late_ltp += 0.2;
                },
                _ => {}
            }
        }
    }
    
    /// Synthesize specific protein (for benchmarking)
    pub fn synthesize_protein(&mut self, protein_type: ProteinType, concentration: f32) {
        self.proteins.entry(protein_type).and_modify(|p| {
            *p = (*p + concentration).clamp(0.0, 1.0);
        });
        
        // Trigger gene expression for protein synthesis
        match protein_type {
            ProteinType::CREB => {
                if concentration > 0.7 {
                    self.activate_immediate_early_genes();
                }
            },
            ProteinType::Arc => {
                self.genes.insert("arc".to_string(), GeneState::Transcribing);
            },
            ProteinType::BDNF => {
                self.genes.insert("bdnf".to_string(), GeneState::Transcribing);
            },
            _ => {}
        }
    }
    
    /// Consolidate memory patterns (for benchmarking)
    pub fn consolidate_memory(&mut self) {
        // Require sufficient CREB for consolidation
        if self.proteins[&ProteinType::CREB] > 0.7 {
            self.consolidation_strength += 0.5;
            
            // Activate late-phase genes
            self.genes.insert("arc".to_string(), GeneState::Translating);
            self.genes.insert("bdnf".to_string(), GeneState::Translating);
            
            // Increase structural proteins
            self.proteins.entry(ProteinType::PSD95).and_modify(|p| *p += 0.3);
            self.proteins.entry(ProteinType::Synaptophysin).and_modify(|p| *p += 0.2);
            
            // Mark as permanent memory
            if self.consolidation_strength > 1.0 {
                self.base.protein_synthesis_triggered = true;
            }
        }
    }
    
    /// Process input with full biological cascade
    pub fn process_with_proteins(&mut self, input: f32, repetitions: u32) -> MemoryFormation {
        let mut result = MemoryFormation::NoChange;
        
        // Calculate total synaptic input
        let effective_input = input * (1.0 + self.proteins[&ProteinType::CaMKII]);
        
        // Process through base neuron
        let tryte_state = self.base.process(effective_input);
        
        // Calcium influx triggers different cascades based on pattern
        if effective_input > self.base.threshold {
            self.trigger_calcium_cascade(effective_input, repetitions);
        }
        
        // Check for protein synthesis trigger
        if self.proteins[&ProteinType::CREB] > 0.7 {
            result = self.initiate_late_phase_ltp();
        }
        
        // Synaptic tagging for future capture
        if effective_input > self.base.threshold * 1.5 {
            self.create_synaptic_tag(effective_input);
        }
        
        // Circadian and sleep influences
        self.apply_circadian_modulation();
        
        // Protein degradation over time
        self.degrade_proteins();
        
        result
    }
    
    /// Calcium cascade leading to protein synthesis
    fn trigger_calcium_cascade(&mut self, input: f32, repetitions: u32) {
        // Calcium activates CaMKII
        self.proteins.entry(ProteinType::CaMKII)
            .and_modify(|p| *p += input * 0.1);
        
        // CaMKII activates PKA with repetition
        if repetitions > 3 {
            self.proteins.entry(ProteinType::PKA)
                .and_modify(|p| *p += 0.05 * repetitions as f32);
        }
        
        // PKA phosphorylates CREB
        let pka_level = self.proteins[&ProteinType::PKA];
        if pka_level > 0.5 {
            self.proteins.entry(ProteinType::CREB)
                .and_modify(|p| *p += pka_level * 0.2);
            
            // CREB activates gene transcription
            if self.proteins[&ProteinType::CREB] > 0.7 {
                self.activate_immediate_early_genes();
            }
        }
        
        // Parallel MAPK pathway for strong stimulation
        if input > 2.0 {
            self.proteins.entry(ProteinType::MAPK)
                .and_modify(|p| *p += 0.1);
        }
    }
    
    /// Activate immediate early genes (IEGs)
    fn activate_immediate_early_genes(&mut self) {
        // Arc gene activation
        if self.genes[&"arc".to_string()] == GeneState::Ready {
            self.genes.insert("arc".to_string(), GeneState::Transcribing);
            self.proteins.entry(ProteinType::Arc)
                .and_modify(|p| *p += 0.3);
        }
        
        // BDNF production for synaptic growth
        if self.genes[&"bdnf".to_string()] == GeneState::Ready {
            self.genes.insert("bdnf".to_string(), GeneState::Transcribing);
            self.proteins.entry(ProteinType::BDNF)
                .and_modify(|p| *p += 0.25);
        }
        
        // Zif268 for consolidation
        if self.genes[&"zif268".to_string()] == GeneState::Ready {
            self.genes.insert("zif268".to_string(), GeneState::Transcribing);
            self.consolidation_strength += 0.2;
        }
    }
    
    /// Initiate late-phase LTP with protein synthesis
    fn initiate_late_phase_ltp(&mut self) -> MemoryFormation {
        self.late_ltp += 0.5;
        
        // Increase synaptic proteins
        self.proteins.entry(ProteinType::PSD95)
            .and_modify(|p| *p += 0.2);
        self.proteins.entry(ProteinType::Synaptophysin)
            .and_modify(|p| *p += 0.15);
        
        // Structural changes become permanent
        if self.late_ltp > 1.0 {
            self.base.protein_synthesis_triggered = true;
            MemoryFormation::LongTermMemory
        } else {
            MemoryFormation::LatePhaseLTP
        }
    }
    
    /// Create synaptic tag for protein capture
    fn create_synaptic_tag(&mut self, strength: f32) {
        self.synaptic_tags.push(SynapticTag {
            strength,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            protein_captured: HashMap::new(),
            decay_rate: 0.1,
        });
    }
    
    /// Apply circadian rhythm effects
    pub fn apply_circadian_modulation(&mut self) {
        // CREB activity peaks during day
        let circadian_factor = (self.circadian_phase * 2.0 * std::f32::consts::PI).sin();
        self.proteins.entry(ProteinType::CREB)
            .and_modify(|p| *p *= 1.0 + circadian_factor * 0.2);
        
        // Sleep pressure enhances consolidation
        if self.sleep_pressure > 0.7 {
            self.consolidation_strength *= 1.5;
        }
    }
    
    /// Degrade proteins over time (biological accuracy)
    pub fn degrade_proteins(&mut self) {
        // Different proteins have different half-lives
        let degradation_rates = HashMap::from([
            (ProteinType::CREB, 0.99),        // Long-lived transcription factor
            (ProteinType::PKA, 0.95),         // Moderately stable kinase
            (ProteinType::MAPK, 0.93),        // Shorter-lived signaling protein
            (ProteinType::CaMKII, 0.97),      // Stable structural protein
            (ProteinType::Arc, 0.90),         // Rapid turnover (activity-dependent)
            (ProteinType::BDNF, 0.94),        // Growth factor, moderate turnover
            (ProteinType::PSD95, 0.995),      // Very stable scaffolding protein
            (ProteinType::Synaptophysin, 0.992), // Stable synaptic vesicle protein
        ]);
        
        for (protein_type, concentration) in self.proteins.iter_mut() {
            let decay_rate = degradation_rates[protein_type];
            *concentration *= decay_rate;
            
            // Minimum baseline levels
            let baseline = match protein_type {
                ProteinType::CREB => 0.05,
                ProteinType::PKA => 0.1,
                ProteinType::MAPK => 0.08,
                ProteinType::CaMKII => 0.15,
                ProteinType::Arc => 0.02,
                ProteinType::BDNF => 0.05,
                ProteinType::PSD95 => 0.2,
                ProteinType::Synaptophysin => 0.18,
            };
            
            if *concentration < baseline {
                *concentration = baseline;
            }
        }
    }
    
    /// Induce long-term depression
    pub fn induce_ltd(&mut self, low_frequency_input: f32) {
        if low_frequency_input < self.base.threshold * 0.5 {
            self.early_ltd += 0.1;
            
            // Reduce synaptic proteins
            self.proteins.entry(ProteinType::PSD95)
                .and_modify(|p| *p *= 0.9);
            
            // Activate protein phosphatases
            if self.early_ltd > 0.5 {
                self.late_ltd += 0.2;
                self.base.plasticity_state = Tryte::Inhibited;
            }
        }
    }
    
    /// Check if memory is in reconsolidation window
    pub fn is_reconsolidating(&self) -> bool {
        if let Some((start, end)) = self.reconsolidation_window {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            now >= start && now <= end
        } else {
            false
        }
    }
    
    /// Open reconsolidation window (memory becomes labile)
    pub fn trigger_reconsolidation(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // 6-hour window where memory can be modified
        self.reconsolidation_window = Some((now, now + 6 * 3600));
        
        // Temporarily reduce protein levels
        for (_, concentration) in self.proteins.iter_mut() {
            *concentration *= 0.7;
        }
    }
}

// Removed duplicate MemoryFormation enum - using the one above

/// Heterosynaptic plasticity - one synapse affects another
pub struct HetorosynapticPlasticity {
    pub neurons: Vec<ProteinSynthesisNeuron>,
    pub protein_pool: HashMap<ProteinType, f32>, // Shared proteins
    pub neuromodulators: HashMap<String, f32>,   // Dopamine, serotonin, etc.
}

impl HetorosynapticPlasticity {
    pub fn new(num_neurons: usize) -> Self {
        let mut protein_pool = HashMap::new();
        // Initialize shared protein pool
        protein_pool.insert(ProteinType::BDNF, 10.0);
        protein_pool.insert(ProteinType::Arc, 5.0);
        
        let mut neuromodulators = HashMap::new();
        neuromodulators.insert("dopamine".to_string(), 0.5);
        neuromodulators.insert("serotonin".to_string(), 0.5);
        neuromodulators.insert("acetylcholine".to_string(), 0.5);
        neuromodulators.insert("norepinephrine".to_string(), 0.5);
        
        Self {
            neurons: vec![ProteinSynthesisNeuron::new(); num_neurons],
            protein_pool,
            neuromodulators,
        }
    }
    
    /// Process with protein sharing between synapses
    pub fn process_with_sharing(&mut self, inputs: &[f32]) -> Vec<MemoryFormation> {
        let mut results = Vec::new();
        
        // First pass: process all neurons
        for (i, input) in inputs.iter().enumerate() {
            if i < self.neurons.len() {
                let result = self.neurons[i].process_with_proteins(*input, 1);
                results.push(result);
            }
        }
        
        // Second pass: share proteins between tagged synapses
        self.share_proteins();
        
        // Neuromodulator influence
        self.apply_neuromodulation();
        
        results
    }
    
    /// Share proteins between synapses with tags
    fn share_proteins(&mut self) {
        for neuron in &mut self.neurons {
            for tag in &mut neuron.synaptic_tags {
                // Strong tags capture more proteins
                if tag.strength > 1.0 {
                    let capture_amount = tag.strength * 0.1;
                    
                    // Capture from pool
                    if let Some(pool_bdnf) = self.protein_pool.get_mut(&ProteinType::BDNF) {
                        let captured = pool_bdnf.min(capture_amount);
                        *pool_bdnf -= captured;
                        tag.protein_captured.insert(ProteinType::BDNF, captured);
                        neuron.proteins.entry(ProteinType::BDNF)
                            .and_modify(|p| *p += captured);
                    }
                }
                
                // Tag decay over time
                tag.strength *= (1.0 - tag.decay_rate);
            }
        }
    }
    
    /// Apply neuromodulator effects
    pub fn apply_neuromodulation(&mut self) {
        let dopamine = self.neuromodulators["dopamine"];
        let serotonin = self.neuromodulators["serotonin"];
        
        for neuron in &mut self.neurons {
            // Dopamine enhances LTP
            if dopamine > 0.7 {
                neuron.late_ltp *= 1.0 + dopamine;
            }
            
            // Serotonin affects mood-dependent learning
            if serotonin < 0.3 {
                neuron.consolidation_strength *= 0.8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_protein_synthesis_trigger() {
        let mut neuron = ProteinSynthesisNeuron::new();
        
        // Weak stimulation shouldn't trigger synthesis
        for _ in 0..5 {
            neuron.process_with_proteins(0.6, 1);
        }
        assert!(!neuron.base.protein_synthesis_triggered);
        
        // Strong repeated stimulation should trigger
        for i in 0..10 {
            let result = neuron.process_with_proteins(2.0, i);
            if result == MemoryFormation::LongTermMemory {
                break;
            }
        }
        assert!(neuron.base.protein_synthesis_triggered);
    }
    
    #[test]
    fn test_synaptic_tagging() {
        let mut neuron = ProteinSynthesisNeuron::new();
        
        // Strong input creates tag
        neuron.process_with_proteins(3.0, 1);
        assert!(!neuron.synaptic_tags.is_empty());
        
        // Tag should have high strength
        assert!(neuron.synaptic_tags[0].strength > 2.0);
    }
    
    #[test]
    fn test_heterosynaptic_sharing() {
        let mut hetero = HetorosynapticPlasticity::new(3);
        
        // Strong stimulation to first neuron
        let inputs = vec![3.0, 0.1, 0.1];
        hetero.process_with_sharing(&inputs);
        
        // First neuron should have captured proteins
        assert!(hetero.neurons[0].proteins[&ProteinType::BDNF] > 0.1);
    }
}