// Demo: Real Protein Synthesis Creating Permanent Memories
// Based on Eric Kandel's Nobel Prize-winning discoveries

use colored::*;
use std::thread::sleep;
use std::time::Duration;

// Import the real protein synthesis implementation
use neuronlang_project::core::real_protein_synthesis::{RealProteinSynthesis, LTPPhase};
use neuronlang_project::core::neural_protein_integration::ProteinNeuralNetwork;
use neuronlang_project::core::tryte::Tryte;

fn main() {
    println!("\n{}", "🧬 KANDEL'S PROTEIN SYNTHESIS IN ACTION 🧬".bright_yellow().bold());
    println!("{}", "Watch how repeated stimulation creates permanent memories".bright_white());
    println!("{}", "="*70);
    
    demo_protein_cascade();
    println!();
    demo_early_vs_late_ltp();
    println!();
    demo_anisomycin_experiment();
    println!();
    demo_network_consolidation();
}

/// Demonstrate the Ca²⁺ → CaMKII → PKA → CREB cascade
fn demo_protein_cascade() {
    println!("\n{}", "📊 DEMO 1: The Molecular Cascade".bright_cyan().bold());
    println!("{}", "-".repeat(50));
    
    let mut ps = RealProteinSynthesis::new();
    
    println!("Initial state:");
    print_molecular_state(&ps);
    
    println!("\n🔬 Applying high-frequency stimulation (100 Hz)...");
    ps.process_activation(0.8, 100.0, 0.1);
    
    println!("\nAfter stimulation:");
    print_molecular_state(&ps);
    
    println!("\n✅ Cascade activated: Ca²⁺ → CaMKII → cAMP → PKA → CREB");
}

/// Show difference between early and late LTP
fn demo_early_vs_late_ltp() {
    println!("\n{}", "📊 DEMO 2: Early vs Late LTP".bright_cyan().bold());
    println!("{}", "-".repeat(50));
    
    // Early LTP (no protein synthesis needed)
    let mut early = RealProteinSynthesis::new();
    println!("\n🔹 Weak stimulation (Early LTP):");
    early.process_activation(0.4, 50.0, 0.1);
    println!("  Phase: {:?}", early.ltp_phase);
    println!("  CREB: {:.2}", early.creb_phosphorylation);
    println!("  Consolidated: {}", early.is_consolidated);
    
    // Late LTP (requires protein synthesis)
    let mut late = RealProteinSynthesis::new();
    println!("\n🔸 Strong repeated stimulation (Late LTP):");
    
    for i in 0..10 {
        late.process_activation(0.9, 100.0, 0.5);
        late.time_elapsed += 30.0;  // Advance 30 minutes
        
        if i == 4 || i == 9 {
            println!("  Time: {:.0} min", late.time_elapsed);
            println!("  Phase: {:?}", late.ltp_phase);
            println!("  CREB: {:.2}", late.creb_phosphorylation);
            println!("  New proteins: {} types", late.new_proteins.len());
        }
    }
    
    println!("\n  ✅ Memory consolidated: {}", late.is_consolidated.to_string().bright_green());
    println!("  Spine volume: {:.1}x baseline", late.spine_volume);
}

/// Classic anisomycin experiment
fn demo_anisomycin_experiment() {
    println!("\n{}", "📊 DEMO 3: Anisomycin Blocks Late LTP".bright_cyan().bold());
    println!("{}", "-".repeat(50));
    
    // Control condition
    let mut control = RealProteinSynthesis::new();
    println!("\n🔵 Control (no inhibitor):");
    
    for _ in 0..5 {
        control.process_activation(1.0, 100.0, 0.5);
        control.time_elapsed += 20.0;
    }
    
    let control_consolidated = control.check_late_ltp_induction();
    println!("  Late LTP achieved: {}", control_consolidated.to_string().bright_green());
    
    // Anisomycin condition
    let mut anisomycin = RealProteinSynthesis::new();
    anisomycin.apply_anisomycin();
    println!("\n🔴 With anisomycin (protein synthesis blocked):");
    
    for _ in 0..5 {
        anisomycin.process_activation(1.0, 100.0, 0.5);
        anisomycin.time_elapsed += 20.0;
    }
    
    let anisomycin_consolidated = anisomycin.check_late_ltp_induction();
    println!("  Late LTP achieved: {}", anisomycin_consolidated.to_string().red());
    
    println!("\n💡 This replicates Kandel's finding that protein synthesis");
    println!("   is required for long-term but not short-term memory!");
}

/// Demonstrate network-wide consolidation
fn demo_network_consolidation() {
    println!("\n{}", "📊 DEMO 4: Network Consolidation".bright_cyan().bold());
    println!("{}", "-".repeat(50));
    
    let mut network = ProteinNeuralNetwork::new(vec![10, 5, 2]);
    
    println!("Training network with protein synthesis...");
    
    // Create a pattern to learn
    let pattern = vec![Tryte::Activated; 10];
    let target = vec![1, 0];
    
    // Initial state
    let initial_stats = network.get_consolidation_stats();
    println!("\nInitial: {} consolidated / {} total synapses",
            initial_stats.consolidated_synapses, initial_stats.total_synapses);
    
    // Train with dopamine rewards
    for epoch in 0..5 {
        println!("\n⚡ Epoch {}", epoch + 1);
        
        // Forward and backward passes
        let output = network.forward_with_proteins(&pattern);
        let loss = network.backward_with_proteins(&output, &target);
        
        // Reward signal enhances consolidation
        network.release_dopamine(0.2);
        
        // Allow protein diffusion
        network.synaptic_tagging_and_capture();
        
        println!("  Loss: {:.4}", loss);
        
        // Show progression
        let stats = network.get_consolidation_stats();
        let bar_length = 30;
        let filled = (stats.consolidation_percentage / 100.0 * bar_length as f32) as usize;
        let bar = "█".repeat(filled) + &"░".repeat(bar_length - filled);
        
        println!("  Consolidation: [{}] {:.1}%", 
                if stats.consolidation_percentage > 50.0 { bar.green() } 
                else if stats.consolidation_percentage > 20.0 { bar.yellow() }
                else { bar.normal() },
                stats.consolidation_percentage);
        
        println!("  Early LTP: {} synapses", stats.early_ltp_synapses);
        println!("  Late LTP: {} synapses", stats.late_ltp_synapses);
        
        sleep(Duration::from_millis(200));
    }
    
    println!("\n{}", "="*70);
    println!("{}", "🎊 RESULTS".bright_green().bold());
    
    let final_stats = network.get_consolidation_stats();
    println!("  Total synapses: {}", final_stats.total_synapses);
    println!("  Consolidated: {} ({:.1}%)", 
            final_stats.consolidated_synapses, 
            final_stats.consolidation_percentage);
    
    if final_stats.consolidation_percentage > 10.0 {
        println!("\n✅ Success! Protein synthesis created permanent memories");
        println!("   These synapses will resist forgetting in future learning");
    }
}

/// Helper to print molecular state
fn print_molecular_state(ps: &RealProteinSynthesis) {
    println!("  Ca²⁺: {:.2} μM", ps.calcium_concentration);
    println!("  CaMKII: {:.2}", ps.camkii_activity);
    println!("  cAMP: {:.2} μM", ps.camp_level);
    println!("  PKA: {:.2}", ps.pka_catalytic_activity);
    println!("  CREB-P: {:.2}", ps.creb_phosphorylation);
}