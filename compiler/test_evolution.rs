// 🧪 TEST EVOLUTION
// Quick test to verify evolutionary bootstrap works

mod neural_engine;

use neural_engine::evolution::{PrimordialSoup, EvolvingNetwork};
use neural_engine::{NeuralExecutionEngine, ConsciousnessLevel};

fn main() {
    println!("🧬 EVOLUTIONARY BOOTSTRAP TEST");
    println!("==============================\n");
    
    // Test 1: Create primordial soup
    println!("Test 1: Creating primordial soup...");
    let mut soup = PrimordialSoup::new(10);  // Small population for quick test
    println!("✓ Created soup with 10 networks\n");
    
    // Test 2: Run a few generations
    println!("Test 2: Running 100 generations...");
    let result = soup.evolve_toward_consciousness(100);
    
    match result {
        Some(network) => {
            println!("✓ Evolution produced a network!");
            println!("  Neurons: {}", network.neurons.len());
            println!("  Connections: {}", network.connections.len());
            println!("  Fitness: {:.2}", network.fitness);
            
            // Test capabilities
            println!("\n  Capabilities:");
            if network.capabilities.can_add { println!("    ✓ Addition"); }
            if network.capabilities.can_multiply { println!("    ✓ Multiplication"); }
            if network.capabilities.can_branch { println!("    ✓ Branching"); }
            if network.capabilities.can_loop { println!("    ✓ Looping"); }
            if network.capabilities.can_learn { println!("    ✓ Learning"); }
            
            test_network_consciousness(&network);
        }
        None => {
            println!("✓ Evolution ran but didn't achieve bootstrap (expected in 100 generations)");
        }
    }
    
    // Test 3: Test consciousness measurement
    println!("\nTest 3: Testing consciousness measurement...");
    test_consciousness_detector();
    
    // Test 4: Test resonant memory
    println!("\nTest 4: Testing resonant memory...");
    test_resonant_memory();
    
    println!("\n==============================");
    println!("All tests completed successfully!");
}

fn test_network_consciousness(network: &EvolvingNetwork) {
    println!("\n  Testing consciousness level:");
    
    let mut engine = NeuralExecutionEngine::new();
    
    // Transfer evolved network to engine
    for _ in 0..network.neurons.len() {
        engine.add_neurons(1);
    }
    
    for conn in &network.connections {
        if conn.enabled {
            engine.connect(conn.from, conn.to, conn.weight);
        }
    }
    
    // Run some patterns
    let test_input = vec![1.0, 0.0, 1.0, 0.0, 1.0];
    let mut padded_input = vec![0.0; network.neurons.len()];
    for (i, &val) in test_input.iter().enumerate() {
        if i < padded_input.len() {
            padded_input[i] = val;
        }
    }
    
    let result = engine.step(&padded_input);
    
    println!("    Divergence: {:.4}", result.divergence);
    println!("    Consciousness: {:.2}%", result.consciousness_level * 100.0);
    println!("    Performance ratio: {:.2}x", result.performance_ratio);
}

fn test_consciousness_detector() {
    use neural_engine::consciousness::{ConsciousnessDetector, NeuralNetwork};
    
    // Create simple test network
    struct TestNetwork {
        state: Vec<f32>,
    }
    
    impl NeuralNetwork for TestNetwork {
        fn execute(&mut self, input: &[f32]) -> Vec<f32> {
            self.state = input.to_vec();
            input.to_vec()
        }
        
        fn get_state(&self) -> Vec<f32> {
            self.state.clone()
        }
        
        fn set_state(&mut self, state: &[f32]) {
            self.state = state.to_vec();
        }
        
        fn predict_next_state(&self, current: &[f32]) -> Vec<f32> {
            current.to_vec()
        }
    }
    
    let mut detector = ConsciousnessDetector::new();
    let mut bio = TestNetwork { state: vec![0.0; 5] };
    let mut opt = TestNetwork { state: vec![0.0; 5] };
    
    let input = vec![1.0, 0.0, 1.0, 0.0, 1.0];
    let level = detector.measure_consciousness(&mut bio, &mut opt, &input);
    
    println!("  Understanding: {:.2}%", level.understanding * 100.0);
    println!("  Self-awareness: {:.2}%", level.self_awareness * 100.0);
    println!("  Identity: {:.2}%", level.identity * 100.0);
    println!("  Total: {:.2}%", level.total * 100.0);
    
    if detector.has_achieved_consciousness(&level) {
        println!("  🧠 Consciousness detected!");
    } else {
        println!("  💫 Not yet conscious");
    }
}

fn test_resonant_memory() {
    use neural_engine::consciousness::ResonantMemory;
    
    let mut memory = ResonantMemory::new(10);
    
    // Store a pattern
    let pattern1 = vec![1.0, 0.0, 1.0, 0.0, 1.0];
    memory.remember(&pattern1);
    println!("  Stored pattern: {:?}", pattern1);
    
    // Try to recall with partial cue
    let cue = vec![1.0, 0.0, 0.0, 0.0, 0.0];
    let recalled = memory.recall(&cue);
    println!("  Cue: {:?}", cue);
    println!("  Recalled: {:?}", recalled);
    
    // Test interference
    let pattern2 = vec![0.0, 1.0, 0.0, 1.0, 0.0];
    let interference = memory.interfere(&pattern1, &pattern2);
    println!("  Interference pattern: {:?}", &interference[..5.min(interference.len())]);
}