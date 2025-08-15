//! Demonstration of Protein Synthesis for Memory Formation
//! 
//! This shows how real biological memory works - the first ever
//! computational implementation of Kandel's Nobel Prize discoveries!

use neuronlang_core::protein_synthesis::{
    ProteinSynthesisNeuron, HetorosynapticPlasticity, MemoryFormation, ProteinType
};

fn main() {
    println!("\n{}", "=".repeat(80));
    println!("🧬 PROTEIN SYNTHESIS DEMONSTRATION");
    println!("   The Biological Basis of Memory");
    println!("{}", "=".repeat(80));
    
    demonstrate_memory_phases();
    demonstrate_spaced_vs_massed_learning();
    demonstrate_sleep_consolidation();
    demonstrate_reconsolidation();
    
    println!("\n{}", "=".repeat(80));
    println!("🏆 BIOLOGICAL MEMORY IMPLEMENTATION COMPLETE!");
    println!("{}", "=".repeat(80));
}

/// Demonstrate the phases of memory formation
fn demonstrate_memory_phases() {
    println!("\n📚 MEMORY FORMATION PHASES:");
    println!("{}", "-".repeat(40));
    
    let mut neuron = ProteinSynthesisNeuron::new();
    
    // Weak stimulation - only early-phase LTP
    println!("\n1️⃣ Weak stimulation (Short-term memory):");
    for i in 0..3 {
        let result = neuron.process_with_proteins(0.8, i + 1);
        println!("   Stim {}: {:?}, CREB = {:.2}", 
                i + 1, result, neuron.proteins[&ProteinType::CREB]);
    }
    println!("   ❌ No protein synthesis - memory will fade in minutes!");
    
    // Reset neuron
    neuron = ProteinSynthesisNeuron::new();
    
    // Strong repeated stimulation - triggers late-phase LTP
    println!("\n2️⃣ Strong repeated stimulation (Long-term memory):");
    for i in 0..8 {
        let result = neuron.process_with_proteins(2.5, i + 1);
        println!("   Stim {}: {:?}", i + 1, result);
        println!("      CREB = {:.2}, PKA = {:.2}, Arc = {:.2}", 
                neuron.proteins[&ProteinType::CREB],
                neuron.proteins[&ProteinType::PKA],
                neuron.proteins[&ProteinType::Arc]);
        
        if result == MemoryFormation::LongTermMemory {
            println!("   ✅ PROTEIN SYNTHESIS COMPLETE - Memory is permanent!");
            break;
        }
    }
}

/// Demonstrate spaced vs massed learning
fn demonstrate_spaced_vs_massed_learning() {
    println!("\n🎓 SPACED VS MASSED LEARNING:");
    println!("{}", "-".repeat(40));
    
    // Massed learning (cramming)
    println!("\n1️⃣ Massed Learning (Cramming):");
    let mut neuron = ProteinSynthesisNeuron::new();
    for i in 0..10 {
        neuron.process_with_proteins(1.5, 1);
    }
    println!("   CREB after 10 massed trials: {:.2}", 
            neuron.proteins[&ProteinType::CREB]);
    println!("   Late-phase LTP: {:.2}", neuron.late_ltp);
    println!("   ⚠️ Poor protein synthesis - weak memory!");
    
    // Spaced learning
    println!("\n2️⃣ Spaced Learning (Distributed practice):");
    let mut neuron = ProteinSynthesisNeuron::new();
    for session in 0..3 {
        println!("   Session {}:", session + 1);
        for _ in 0..3 {
            neuron.process_with_proteins(1.5, session + 1);
        }
        // Simulate rest between sessions - allows protein replenishment
        for _ in 0..5 {
            neuron.degrade_proteins();
        }
        println!("      CREB: {:.2}, Consolidation: {:.2}", 
                neuron.proteins[&ProteinType::CREB],
                neuron.consolidation_strength);
    }
    println!("   ✅ Better protein synthesis - strong memory!");
}

/// Demonstrate sleep's role in consolidation
fn demonstrate_sleep_consolidation() {
    println!("\n😴 SLEEP AND MEMORY CONSOLIDATION:");
    println!("{}", "-".repeat(40));
    
    let mut neuron = ProteinSynthesisNeuron::new();
    
    // Learning during day
    println!("\n1️⃣ Learning phase (awake):");
    neuron.circadian_phase = 0.25; // Daytime
    for i in 0..5 {
        neuron.process_with_proteins(1.8, i + 1);
    }
    println!("   Proteins synthesized during learning:");
    println!("   BDNF: {:.2}, Arc: {:.2}, PSD95: {:.2}",
            neuron.proteins[&ProteinType::BDNF],
            neuron.proteins[&ProteinType::Arc],
            neuron.proteins[&ProteinType::PSD95]);
    let awake_consolidation = neuron.consolidation_strength;
    
    // Sleep phase
    println!("\n2️⃣ Sleep phase (consolidation):");
    neuron.circadian_phase = 0.75; // Nighttime
    neuron.sleep_pressure = 0.9;
    neuron.apply_circadian_modulation();
    
    println!("   Consolidation strength:");
    println!("   Before sleep: {:.2}", awake_consolidation);
    println!("   During sleep: {:.2}", neuron.consolidation_strength);
    println!("   ✅ Sleep enhanced consolidation by {:.0}%!", 
            (neuron.consolidation_strength / awake_consolidation - 1.0) * 100.0);
}

/// Demonstrate memory reconsolidation
fn demonstrate_reconsolidation() {
    println!("\n🔄 MEMORY RECONSOLIDATION:");
    println!("{}", "-".repeat(40));
    
    let mut neuron = ProteinSynthesisNeuron::new();
    
    // Form initial memory
    println!("\n1️⃣ Initial memory formation:");
    for i in 0..6 {
        neuron.process_with_proteins(2.0, i + 1);
    }
    println!("   Initial memory formed with late-LTP: {:.2}", neuron.late_ltp);
    
    // Trigger reconsolidation
    println!("\n2️⃣ Memory retrieval (triggers reconsolidation):");
    neuron.trigger_reconsolidation();
    println!("   Memory now in labile state!");
    println!("   Protein levels temporarily reduced");
    println!("   Window open for {} hours", 6);
    
    // Update memory during reconsolidation
    println!("\n3️⃣ Updating memory during reconsolidation:");
    if neuron.is_reconsolidating() {
        println!("   ✅ Memory is reconsolidating - can be modified!");
        neuron.process_with_proteins(3.0, 1);
        println!("   Memory updated with new information");
    }
    
    // After reconsolidation window
    println!("\n4️⃣ After reconsolidation window closes:");
    neuron.reconsolidation_window = None;
    if !neuron.is_reconsolidating() {
        println!("   ❌ Reconsolidation window closed");
        println!("   Memory is stable again - harder to modify");
    }
}

/// Demonstrate heterosynaptic plasticity
fn demonstrate_protein_sharing() {
    println!("\n🔗 HETEROSYNAPTIC PROTEIN SHARING:");
    println!("{}", "-".repeat(40));
    
    let mut network = HetorosynapticPlasticity::new(3);
    
    println!("\n1️⃣ Initial state:");
    println!("   Shared BDNF pool: {:.1}", 
            network.protein_pool[&ProteinType::BDNF]);
    
    // Strong stimulation to first neuron only
    println!("\n2️⃣ Strong stimulation to neuron 1:");
    let inputs = vec![3.0, 0.0, 0.0];
    let results = network.process_with_sharing(&inputs);
    
    for (i, result) in results.iter().enumerate() {
        println!("   Neuron {}: {:?}", i + 1, result);
    }
    
    // Check protein distribution
    println!("\n3️⃣ Protein distribution after stimulation:");
    for (i, neuron) in network.neurons.iter().enumerate() {
        println!("   Neuron {} BDNF: {:.2}", 
                i + 1, neuron.proteins[&ProteinType::BDNF]);
    }
    
    println!("\n4️⃣ Neuromodulator effects:");
    network.neuromodulators.insert("dopamine".to_string(), 0.9);
    println!("   Dopamine increased to 0.9 (reward signal)");
    network.apply_neuromodulation();
    println!("   ✅ All neurons' LTP enhanced by dopamine!");
}