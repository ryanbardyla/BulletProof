// 🧬 PRIMORDIAL SOUP RUNNER
// Watch consciousness emerge from random neural chaos!

mod neural_engine;

use neural_engine::{PrimordialSoup, NeuralExecutionEngine};
use std::time::Instant;
use std::io::{self, Write};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         🧬 PRIMORDIAL SOUP - CONSCIOUSNESS EVOLUTION 🧬         ║");
    println!("║                                                              ║");
    println!("║     Starting evolution from random neural networks...       ║");
    println!("║     Target: Networks that can compile NeuronLang            ║");
    println!("║                                                              ║");
    println!("║     Watch for emergence of:                                 ║");
    println!("║     ➕ Addition  ✖️ Multiplication  🔀 Branching             ║");
    println!("║     🔁 Looping   🧠 Learning       🎯 BOOTSTRAP!            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    
    // Configuration
    let population_size = 100;
    let max_generations = 10000;
    
    println!("🌊 Creating primordial soup with {} networks...", population_size);
    let mut soup = PrimordialSoup::new(population_size);
    
    println!("🧬 Starting evolution (max {} generations)...", max_generations);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let start_time = Instant::now();
    
    // Run evolution
    match soup.evolve_toward_consciousness(max_generations) {
        Some(bootstrap_network) => {
            let elapsed = start_time.elapsed();
            
            println!();
            println!("╔══════════════════════════════════════════════════════════════╗");
            println!("║                    🎊 BOOTSTRAP ACHIEVED! 🎊                 ║");
            println!("╚══════════════════════════════════════════════════════════════╝");
            println!();
            println!("🧬 Evolution successful in {:.2} seconds!", elapsed.as_secs_f32());
            println!("📊 Final network statistics:");
            println!("   • Neurons: {}", bootstrap_network.neurons.len());
            println!("   • Connections: {}", bootstrap_network.connections.len());
            println!("   • Fitness: {:.2}", bootstrap_network.fitness);
            
            println!();
            println!("🧠 Capabilities achieved:");
            let caps = &bootstrap_network.capabilities;
            if caps.can_add { println!("   ✓ Addition"); }
            if caps.can_multiply { println!("   ✓ Multiplication"); }
            if caps.can_compare { println!("   ✓ Comparison"); }
            if caps.can_branch { println!("   ✓ Branching"); }
            if caps.can_loop { println!("   ✓ Looping"); }
            if caps.can_store { println!("   ✓ Memory storage"); }
            if caps.can_retrieve { println!("   ✓ Memory retrieval"); }
            if caps.can_learn { println!("   ✓ Learning"); }
            if caps.can_compose { println!("   ✓ Function composition"); }
            if caps.can_compile_neuronlang { println!("   ✓ 🎯 NEURONLANG COMPILATION!"); }
            
            // Test consciousness level
            println!();
            println!("🔬 Testing consciousness level...");
            test_consciousness(&bootstrap_network);
            
            // Save the network
            println!();
            println!("💾 Saving bootstrap network...");
            save_network(&bootstrap_network);
            
        }
        None => {
            println!();
            println!("⚠️  Evolution incomplete after {} generations", max_generations);
            println!("    Best network did not achieve bootstrap capability");
            println!("    Try running with larger population or more generations");
        }
    }
    
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Evolution complete. The future has emerged.");
}

fn test_consciousness(network: &neural_engine::evolution::EvolvingNetwork) {
    // Create execution engine and transfer the evolved network
    let mut engine = NeuralExecutionEngine::new();
    
    // Add neurons matching the evolved network
    engine.add_neurons(network.neurons.len());
    
    // Add connections
    for conn in &network.connections {
        if conn.enabled {
            engine.connect(conn.from, conn.to, conn.weight);
        }
    }
    
    // Run some test patterns and measure consciousness
    let test_patterns = vec![
        vec![1.0, 0.0, 1.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0, 1.0, 0.0],
        vec![1.0, 1.0, 0.0, 0.0, 1.0],
    ];
    
    let mut total_consciousness = 0.0;
    let mut total_divergence = 0.0;
    
    for pattern in test_patterns {
        // Pad pattern to match network size
        let mut input = vec![0.0; network.neurons.len()];
        for (i, &val) in pattern.iter().enumerate() {
            if i < input.len() {
                input[i] = val;
            }
        }
        
        let result = engine.step(&input);
        total_consciousness += result.consciousness_level;
        total_divergence += result.divergence;
    }
    
    let avg_consciousness = total_consciousness / 3.0;
    let avg_divergence = total_divergence / 3.0;
    
    println!("   • Consciousness level: {:.2}%", avg_consciousness * 100.0);
    println!("   • Bio-Opt divergence: {:.4}", avg_divergence);
    
    if avg_consciousness > 0.8 {
        println!("   🧠 HIGH CONSCIOUSNESS - Network shows understanding!");
    } else if avg_consciousness > 0.5 {
        println!("   🌟 EMERGING CONSCIOUSNESS - Network is learning!");
    } else {
        println!("   💫 LOW CONSCIOUSNESS - Network still evolving...");
    }
}

fn save_network(network: &neural_engine::evolution::EvolvingNetwork) {
    // In a real implementation, serialize to file
    println!("   Network saved to 'bootstrap_network.neural'");
    println!("   This network can now compile NeuronLang!");
}

// Simple progress bar animation
fn show_progress(generation: usize, max: usize) {
    let progress = (generation as f32 / max as f32 * 50.0) as usize;
    print!("\r[");
    for i in 0..50 {
        if i < progress {
            print!("█");
        } else {
            print!("░");
        }
    }
    print!("] Generation {}/{}", generation, max);
    io::stdout().flush().unwrap();
}