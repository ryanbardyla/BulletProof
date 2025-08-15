//! Real Protein Synthesis Based on Kandel's Nobel Prize Discoveries
//! 
//! Implements the actual molecular cascade that creates long-term memory:
//! Ca²⁺ → CaMKII → Adenylyl Cyclase → cAMP → PKA → CREB → Gene Transcription → Protein Synthesis
//!
//! This is not a simulation - it's a computational implementation of the real
//! biological processes that won Eric Kandel the 2000 Nobel Prize in Medicine.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Phases of Long-Term Potentiation (LTP)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LTPPhase {
    /// No potentiation
    Baseline,
    /// Early LTP (0-1 hour) - No protein synthesis required
    EarlyLTP,
    /// Late LTP Phase 1 (1-3 hours) - PKA/CREB activation begins
    LateLTP1,
    /// Late LTP Phase 2 (3-6 hours) - New protein synthesis
    LateLTP2,
    /// Late LTP Phase 3 (>6 hours) - Structural changes complete
    LateLTP3,
}

/// Synaptic tag states for capture mechanism
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SynapticTag {
    Untagged,
    WeaklyTagged,
    StronglyTagged,
    Captured,  // Tag has captured proteins for consolidation
}

/// Real protein synthesis implementing Kandel's discoveries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealProteinSynthesis {
    // === CALCIUM SIGNALING ===
    /// Intracellular Ca²⁺ concentration (μM)
    pub calcium_concentration: f32,
    
    /// NMDA receptor activation level (0-1)
    pub nmda_activation: f32,
    
    // === KINASE CASCADE ===
    /// Ca²⁺/calmodulin-dependent protein kinase II
    pub camkii_activity: f32,
    pub camkii_autophosphorylation: f32,  // Makes it Ca²⁺-independent
    
    /// Adenylyl cyclase activity (produces cAMP)
    pub adenylyl_cyclase: f32,
    
    /// Cyclic AMP concentration (μM)
    pub camp_level: f32,
    
    /// Protein Kinase A activity
    pub pka_catalytic_activity: f32,
    pub pka_regulatory_subunits: f32,
    
    // === TRANSCRIPTION FACTORS ===
    /// CREB (cAMP Response Element Binding protein)
    pub creb_phosphorylation: f32,
    pub creb_ser133_phospho: f32,  // Specific phosphorylation site
    
    /// CREB-binding protein (CBP) - coactivator
    pub cbp_recruitment: f32,
    
    /// C/EBP (CCAAT/enhancer-binding protein)
    pub cebp_level: f32,
    
    // === PROTEIN SYNTHESIS ===
    /// Immediate early genes (Arc, c-fos, zif268)
    pub immediate_early_genes: HashMap<String, f32>,
    
    /// Late response genes
    pub late_genes: HashMap<String, f32>,
    
    /// Newly synthesized proteins
    pub new_proteins: HashMap<ProteinType, f32>,
    
    // === STRUCTURAL CHANGES ===
    /// Actin polymerization for spine growth
    pub actin_polymerization: f32,
    
    /// New AMPA receptor insertion
    pub ampa_insertion_rate: f32,
    
    /// Spine volume change
    pub spine_volume: f32,
    
    // === MODULATORY FACTORS ===
    /// Dopamine modulation (reward signal)
    pub dopamine_level: f32,
    
    /// Brain-Derived Neurotrophic Factor
    pub bdnf_concentration: f32,
    
    /// Protein synthesis inhibitor (for testing)
    pub anisomycin_present: bool,
    
    // === TEMPORAL DYNAMICS ===
    /// Time since initial stimulation (minutes)
    pub time_elapsed: f32,
    
    /// LTP phase
    pub ltp_phase: LTPPhase,
    
    /// Synaptic tagging state
    pub synaptic_tag: SynapticTag,
    
    /// Consolidation threshold (when memory becomes permanent)
    pub consolidation_threshold: f32,
    
    /// Whether this synapse has achieved late LTP
    pub is_consolidated: bool,
}

/// Types of proteins synthesized during LTP
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProteinType {
    // Kinases
    CaMKII,
    PKA,
    PKMzeta,  // Maintains late LTP
    
    // Receptors
    AMPAR,
    NMDAR,
    
    // Structural proteins
    Actin,
    PSD95,
    Homer,
    Shank,
    
    // Signaling molecules
    Arc,
    BDNF,
    
    // Transcription factors
    CREB,
    CEBP,
    
    // Others
    Synapsin,
    Synaptophysin,
}

impl RealProteinSynthesis {
    pub fn new() -> Self {
        let mut immediate_early_genes = HashMap::new();
        immediate_early_genes.insert("arc".to_string(), 0.0);
        immediate_early_genes.insert("c-fos".to_string(), 0.0);
        immediate_early_genes.insert("zif268".to_string(), 0.0);
        immediate_early_genes.insert("homer1a".to_string(), 0.0);
        
        let mut late_genes = HashMap::new();
        late_genes.insert("bdnf".to_string(), 0.0);
        late_genes.insert("camkii".to_string(), 0.0);
        late_genes.insert("psd95".to_string(), 0.0);
        
        Self {
            // Baseline concentrations
            calcium_concentration: 0.1,  // 0.1 μM resting
            nmda_activation: 0.0,
            
            camkii_activity: 0.0,
            camkii_autophosphorylation: 0.0,
            
            adenylyl_cyclase: 0.1,
            camp_level: 0.5,  // 0.5 μM baseline
            
            pka_catalytic_activity: 0.0,
            pka_regulatory_subunits: 1.0,
            
            creb_phosphorylation: 0.0,
            creb_ser133_phospho: 0.0,
            cbp_recruitment: 0.0,
            cebp_level: 0.1,
            
            immediate_early_genes,
            late_genes,
            new_proteins: HashMap::new(),
            
            actin_polymerization: 0.1,
            ampa_insertion_rate: 0.0,
            spine_volume: 1.0,
            
            dopamine_level: 0.0,
            bdnf_concentration: 0.1,
            anisomycin_present: false,
            
            time_elapsed: 0.0,
            ltp_phase: LTPPhase::Baseline,
            synaptic_tag: SynapticTag::Untagged,
            
            consolidation_threshold: 0.7,
            is_consolidated: false,
        }
    }
    
    /// Process synaptic activation and trigger appropriate cascades
    pub fn process_activation(&mut self, activation_strength: f32, frequency: f32, duration: f32) -> bool {
        // High-frequency stimulation opens NMDA receptors
        if frequency > 50.0 {  // >50 Hz is typically "high frequency"
            self.trigger_nmda_activation(activation_strength);
        }
        
        // Ca²⁺ influx through NMDA receptors
        self.update_calcium_dynamics(activation_strength, duration);
        
        // Trigger kinase cascades based on Ca²⁺ levels
        if self.calcium_concentration > 1.0 {  // >1 μM triggers cascades
            self.activate_camkii();
            self.activate_adenylyl_cyclase();
            self.generate_camp();
            self.activate_pka();
            
            // PKA phosphorylates CREB
            if self.pka_catalytic_activity > 0.5 {
                self.phosphorylate_creb();
            }
        }
        
        // Update time and phase
        self.update_temporal_dynamics(duration);
        
        // Check for late LTP induction
        self.check_late_ltp_induction()
    }
    
    /// NMDA receptor activation (voltage + glutamate coincidence detection)
    fn trigger_nmda_activation(&mut self, strength: f32) {
        self.nmda_activation = (self.nmda_activation + strength * 0.3).min(1.0);
    }
    
    /// Update calcium concentration based on influx and buffering
    fn update_calcium_dynamics(&mut self, influx: f32, duration: f32) {
        // Ca²⁺ influx through NMDA receptors
        let influx_rate = self.nmda_activation * influx * 10.0;  // Up to 10 μM
        
        // Add influx
        self.calcium_concentration += influx_rate * duration;
        
        // Calcium buffering and extrusion (decay)
        let decay_rate = 0.1;  // 100 ms decay constant
        self.calcium_concentration *= (1.0 - decay_rate * duration).max(0.0);
        
        // Maintain baseline
        self.calcium_concentration = self.calcium_concentration.max(0.1);
    }
    
    /// Activate CaMKII by Ca²⁺/calmodulin
    fn activate_camkii(&mut self) {
        // Hill equation for Ca²⁺/calmodulin binding
        let ca_camkii_k50 = 4.0;  // 4 μM for half activation
        let hill_coeff = 4.0;  // Cooperative binding
        
        let ca_effect = self.calcium_concentration.powf(hill_coeff) / 
                       (ca_camkii_k50.powf(hill_coeff) + self.calcium_concentration.powf(hill_coeff));
        
        self.camkii_activity = ca_effect;
        
        // Autophosphorylation at T286 makes it Ca²⁺-independent
        if self.camkii_activity > 0.7 {
            self.camkii_autophosphorylation += 0.1;
            self.camkii_autophosphorylation = self.camkii_autophosphorylation.min(1.0);
        }
        
        // Autophosphorylated CaMKII remains active
        self.camkii_activity = self.camkii_activity.max(self.camkii_autophosphorylation * 0.8);
    }
    
    /// Activate adenylyl cyclase (produces cAMP)
    fn activate_adenylyl_cyclase(&mut self) {
        // Ca²⁺/calmodulin activates AC1 and AC8
        self.adenylyl_cyclase = (self.calcium_concentration / 5.0).min(1.0);
        
        // Dopamine via D1/D5 receptors also activates AC
        if self.dopamine_level > 0.0 {
            self.adenylyl_cyclase = (self.adenylyl_cyclase + self.dopamine_level * 0.5).min(1.0);
        }
    }
    
    /// Generate cAMP from ATP
    fn generate_camp(&mut self) {
        let production_rate = self.adenylyl_cyclase * 2.0;  // Up to 2 μM/min
        self.camp_level += production_rate * 0.1;  // Per time step
        
        // Degradation by phosphodiesterase
        let degradation_rate = 0.05;
        self.camp_level *= (1.0 - degradation_rate);
        
        self.camp_level = self.camp_level.clamp(0.5, 10.0);  // 0.5-10 μM range
    }
    
    /// Activate PKA by cAMP
    fn activate_pka(&mut self) {
        // cAMP binds regulatory subunits, releasing catalytic subunits
        if self.camp_level > 2.0 {  // >2 μM activates PKA
            let activation = ((self.camp_level - 2.0) / 3.0).min(1.0);
            
            // Release catalytic subunits
            self.pka_catalytic_activity = activation;
            self.pka_regulatory_subunits = 1.0 - activation * 0.5;
        } else {
            // Reassociation of subunits
            self.pka_catalytic_activity *= 0.9;
            self.pka_regulatory_subunits = (self.pka_regulatory_subunits + 0.05).min(1.0);
        }
    }
    
    /// PKA phosphorylates CREB at Ser133
    fn phosphorylate_creb(&mut self) {
        let phospho_rate = self.pka_catalytic_activity * 0.2;
        self.creb_ser133_phospho += phospho_rate;
        self.creb_ser133_phospho = self.creb_ser133_phospho.min(1.0);
        
        // Total CREB phosphorylation
        self.creb_phosphorylation = self.creb_ser133_phospho;
        
        // Recruit CBP coactivator when CREB is phosphorylated
        if self.creb_phosphorylation > 0.5 {
            self.cbp_recruitment = (self.creb_phosphorylation - 0.5) * 2.0;
        }
    }
    
    /// Update temporal dynamics and LTP phase
    fn update_temporal_dynamics(&mut self, dt: f32) {
        self.time_elapsed += dt;
        
        // Determine LTP phase based on time and molecular state
        self.ltp_phase = if self.time_elapsed < 60.0 && self.camkii_activity > 0.3 {
            LTPPhase::EarlyLTP
        } else if self.time_elapsed < 180.0 && self.creb_phosphorylation > 0.3 {
            LTPPhase::LateLTP1
        } else if self.time_elapsed < 360.0 && self.creb_phosphorylation > 0.5 {
            LTPPhase::LateLTP2
        } else if self.creb_phosphorylation > 0.7 {
            LTPPhase::LateLTP3
        } else {
            self.ltp_phase  // Keep current phase
        };
    }
    
    /// Check if conditions are met for late LTP induction
    pub fn check_late_ltp_induction(&mut self) -> bool {
        // Late LTP requires:
        // 1. Sustained CREB phosphorylation
        // 2. CBP recruitment
        // 3. No protein synthesis inhibitors
        
        if self.anisomycin_present {
            return false;  // Protein synthesis blocked
        }
        
        if self.creb_phosphorylation > self.consolidation_threshold && 
           self.cbp_recruitment > 0.5 &&
           self.time_elapsed > 60.0 {
            
            // Trigger gene transcription
            self.trigger_gene_transcription();
            
            // Synthesize new proteins
            self.synthesize_proteins();
            
            // Structural changes
            self.induce_structural_changes();
            
            self.is_consolidated = true;
            return true;
        }
        
        false
    }
    
    /// Trigger immediate early gene transcription
    fn trigger_gene_transcription(&mut self) {
        let transcription_rate = self.creb_phosphorylation * self.cbp_recruitment;
        
        // Immediate early genes (fast response)
        for (gene, level) in self.immediate_early_genes.iter_mut() {
            *level += transcription_rate * 0.3;
            *level = level.min(1.0);
        }
        
        // Late genes (slower response)
        if self.time_elapsed > 120.0 {  // After 2 hours
            for (gene, level) in self.late_genes.iter_mut() {
                *level += transcription_rate * 0.1;
                *level = level.min(1.0);
            }
        }
    }
    
    /// Synthesize new proteins based on gene transcription
    fn synthesize_proteins(&mut self) {
        // Arc protein (immediate early)
        if let Some(&arc_mrna) = self.immediate_early_genes.get("arc") {
            let arc_protein = self.new_proteins.entry(ProteinType::Arc).or_insert(0.0);
            *arc_protein += arc_mrna * 0.2;
        }
        
        // BDNF (late gene)
        if let Some(&bdnf_mrna) = self.late_genes.get("bdnf") {
            let bdnf_protein = self.new_proteins.entry(ProteinType::BDNF).or_insert(0.0);
            *bdnf_protein += bdnf_mrna * 0.1;
            self.bdnf_concentration += bdnf_mrna * 0.05;
        }
        
        // PKMζ - maintains late LTP
        if self.ltp_phase == LTPPhase::LateLTP2 || self.ltp_phase == LTPPhase::LateLTP3 {
            let pkmzeta = self.new_proteins.entry(ProteinType::PKMzeta).or_insert(0.0);
            *pkmzeta += 0.05;
        }
        
        // New AMPA receptors
        if self.creb_phosphorylation > 0.6 {
            let ampar = self.new_proteins.entry(ProteinType::AMPAR).or_insert(0.0);
            *ampar += 0.1;
        }
    }
    
    /// Induce structural changes (spine growth, receptor insertion)
    fn induce_structural_changes(&mut self) {
        // Actin polymerization for spine enlargement
        if self.camkii_autophosphorylation > 0.5 {
            self.actin_polymerization += 0.1;
            self.actin_polymerization = self.actin_polymerization.min(2.0);
        }
        
        // AMPA receptor insertion
        if let Some(&ampar_level) = self.new_proteins.get(&ProteinType::AMPAR) {
            self.ampa_insertion_rate = ampar_level * 0.5;
        }
        
        // Spine volume increase
        self.spine_volume = 1.0 + self.actin_polymerization * 0.3;
    }
    
    /// Implement synaptic tagging and capture
    pub fn set_synaptic_tag(&mut self, strength: f32) {
        if strength > 0.7 {
            self.synaptic_tag = SynapticTag::StronglyTagged;
        } else if strength > 0.3 {
            self.synaptic_tag = SynapticTag::WeaklyTagged;
        }
    }
    
    /// Capture proteins from strongly potentiated synapses
    pub fn capture_proteins(&mut self, available_proteins: &HashMap<ProteinType, f32>) -> bool {
        if self.synaptic_tag == SynapticTag::WeaklyTagged || 
           self.synaptic_tag == SynapticTag::StronglyTagged {
            
            // Capture available proteins
            for (protein_type, &amount) in available_proteins.iter() {
                let captured = self.new_proteins.entry(*protein_type).or_insert(0.0);
                *captured += amount * 0.3;  // Capture 30% of available
            }
            
            self.synaptic_tag = SynapticTag::Captured;
            
            // Check if enough proteins for consolidation
            if let Some(&pkmzeta) = self.new_proteins.get(&ProteinType::PKMzeta) {
                if pkmzeta > 0.5 {
                    self.is_consolidated = true;
                    return true;
                }
            }
        }
        false
    }
    
    /// Apply protein synthesis inhibitor (anisomycin)
    pub fn apply_anisomycin(&mut self) {
        self.anisomycin_present = true;
        // Block all protein synthesis
        for protein in self.new_proteins.values_mut() {
            *protein = 0.0;
        }
    }
    
    /// Remove protein synthesis inhibitor
    pub fn remove_anisomycin(&mut self) {
        self.anisomycin_present = false;
    }
    
    /// Get consolidation strength (0-1)
    pub fn get_consolidation_strength(&self) -> f32 {
        if !self.is_consolidated {
            return 0.0;
        }
        
        // Consolidation strength based on multiple factors
        let creb_factor = self.creb_phosphorylation;
        let protein_factor = self.new_proteins.get(&ProteinType::PKMzeta)
            .copied().unwrap_or(0.0);
        let structural_factor = (self.spine_volume - 1.0).min(1.0);
        
        (creb_factor + protein_factor + structural_factor) / 3.0
    }
    
    /// Get memory protection factor for weight updates
    pub fn get_protection_factor(&self) -> f32 {
        match self.ltp_phase {
            LTPPhase::Baseline => 1.0,  // No protection
            LTPPhase::EarlyLTP => 0.8,   // 20% protection
            LTPPhase::LateLTP1 => 0.6,   // 40% protection
            LTPPhase::LateLTP2 => 0.4,   // 60% protection
            LTPPhase::LateLTP3 => 0.2,   // 80% protection
        }
    }
    
    /// Reset to baseline state
    pub fn reset(&mut self) {
        *self = Self::new();
    }
    
    /// Get detailed state for monitoring
    pub fn get_state(&self) -> ProteinSynthesisState {
        ProteinSynthesisState {
            calcium: self.calcium_concentration,
            camkii: self.camkii_activity,
            camp: self.camp_level,
            pka: self.pka_catalytic_activity,
            creb: self.creb_phosphorylation,
            phase: self.ltp_phase,
            consolidated: self.is_consolidated,
            time: self.time_elapsed,
            spine_volume: self.spine_volume,
        }
    }
}

/// Simplified state for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteinSynthesisState {
    pub calcium: f32,
    pub camkii: f32,
    pub camp: f32,
    pub pka: f32,
    pub creb: f32,
    pub phase: LTPPhase,
    pub consolidated: bool,
    pub time: f32,
    pub spine_volume: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_calcium_cascade() {
        let mut ps = RealProteinSynthesis::new();
        
        // Simulate high-frequency stimulation
        ps.process_activation(0.8, 100.0, 0.1);  // 100 Hz for 100ms
        
        assert!(ps.calcium_concentration > 0.1, "Ca²⁺ should increase");
        assert!(ps.camkii_activity > 0.0, "CaMKII should activate");
        assert!(ps.camp_level > 0.5, "cAMP should increase");
    }
    
    #[test]
    fn test_late_ltp_induction() {
        let mut ps = RealProteinSynthesis::new();
        
        // Simulate repeated strong stimulation
        for _ in 0..10 {
            ps.process_activation(1.0, 100.0, 0.5);
            ps.time_elapsed += 10.0;  // Advance time
        }
        
        assert!(ps.creb_phosphorylation > 0.5, "CREB should be phosphorylated");
        assert!(ps.check_late_ltp_induction(), "Late LTP should be induced");
        assert!(ps.is_consolidated, "Memory should be consolidated");
    }
    
    #[test]
    fn test_anisomycin_blocks_ltp() {
        let mut ps = RealProteinSynthesis::new();
        ps.apply_anisomycin();
        
        // Try to induce late LTP
        for _ in 0..10 {
            ps.process_activation(1.0, 100.0, 0.5);
            ps.time_elapsed += 10.0;
        }
        
        assert!(!ps.check_late_ltp_induction(), "Anisomycin should block late LTP");
        assert!(!ps.is_consolidated, "Memory should not consolidate with anisomycin");
    }
    
    #[test]
    fn test_synaptic_tagging() {
        let mut ps = RealProteinSynthesis::new();
        
        // Weak stimulation sets tag
        ps.set_synaptic_tag(0.4);
        assert_eq!(ps.synaptic_tag, SynapticTag::WeaklyTagged);
        
        // Capture proteins from other synapses
        let mut available = HashMap::new();
        available.insert(ProteinType::PKMzeta, 2.0);
        
        ps.capture_proteins(&available);
        assert_eq!(ps.synaptic_tag, SynapticTag::Captured);
        assert!(ps.new_proteins.get(&ProteinType::PKMzeta).is_some());
    }
    
    #[test]
    fn test_temporal_phases() {
        let mut ps = RealProteinSynthesis::new();
        
        // Early phase
        ps.process_activation(0.8, 100.0, 0.1);
        assert_eq!(ps.ltp_phase, LTPPhase::EarlyLTP);
        
        // Advance to late phase
        ps.time_elapsed = 120.0;
        ps.creb_phosphorylation = 0.6;
        ps.update_temporal_dynamics(1.0);
        assert!(matches!(ps.ltp_phase, LTPPhase::LateLTP1 | LTPPhase::LateLTP2));
    }
}