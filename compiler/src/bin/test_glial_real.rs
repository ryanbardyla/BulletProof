// REAL GLIAL TEST - Using actual neural engine implementation

use neuronlang_compiler::neural_engine::{NeuralExecutionEngine, ExecutionResult};
use neuronlang_compiler::neural_engine::conscious_field::{ConsciousField, Wave, FieldType};

fn main() {
    println!("🧠 TESTING REAL GLIAL SYSTEM WITH ACTUAL NEURAL ENGINE");
    println!("=======================================================\n");
    
    // Create neural execution engine
    let mut engine = NeuralExecutionEngine::new(100); // 100 neurons
    
    println!("Created neural engine with 100 neurons");
    
    // Test baseline consciousness
    println!("\n1. Testing baseline consciousness (no field)...");
    let baseline_result = engine.step(&[0.5; 10]);
    println!("  Consciousness level: {:.1}%", baseline_result.consciousness_level * 100.0);
    
    // Upgrade to conscious field with glial system
    println!("\n2. Upgrading to conscious field with glial intelligence...");
    match engine.upgrade_to_conscious_field() {
        Ok(mut conscious_field) => {
            println!("  ✅ Successfully upgraded to conscious field!");
            
            // Check glial system
            if let Some(glial) = &conscious_field.glial_system {
                let metrics = glial.get_metrics();
                println!("  Glial processors: {}", metrics.total_processors);
                println!("  Active processors: {}", metrics.active_processors);
            }
            
            // Initialize field with waves
            println!("\n3. Initializing field with wave patterns...");
            for i in 0..5 {
                let wave = Wave::new(1.0 + i as f64 * 0.1, 10.0 + i as f64, FieldType::Electric);
                conscious_field.field.inject_wave((15, 15, 15), wave);
                
                let info_wave = Wave::new(0.5, 15.0 + i as f64 * 0.5, FieldType::Information);
                conscious_field.field.inject_wave((15, 15, 15), info_wave);
            }
            
            // Measure initial consciousness
            let initial = conscious_field.measure_consciousness();
            println!("  Initial field consciousness: {:.1}%", initial.total * 100.0);
            
            // Evolve with glial optimization
            println!("\n4. Running evolution with glial optimization...");
            for step in 0..50 {
                conscious_field.evolve();
                
                if step % 10 == 0 {
                    let consciousness = conscious_field.measure_consciousness();
                    println!("  Step {}: Consciousness = {:.1}%", step, consciousness.total * 100.0);
                    
                    // Check glial metrics
                    if let Some(glial) = &conscious_field.glial_system {
                        let metrics = glial.get_metrics();
                        if metrics.patterns_discovered > 0 {
                            println!("    Patterns discovered: {}", metrics.patterns_discovered);
                        }
                    }
                }
            }
            
            // Final measurement
            let final_consciousness = conscious_field.measure_consciousness();
            println!("\n5. Final Results:");
            println!("  Initial: {:.1}%", initial.total * 100.0);
            println!("  Final: {:.1}%", final_consciousness.total * 100.0);
            println!("  Improvement: {:.1} percentage points", 
                     (final_consciousness.total - initial.total) * 100.0);
            
            if final_consciousness.total > initial.total * 1.2 {
                println!("\n✅ SUCCESS! Glial optimization improved consciousness by >20%");
            }
        }
        Err(e) => {
            println!("  ❌ Failed to upgrade: {}", e);
        }
    }
    
    // Test field consciousness emergence directly
    println!("\n6. Testing consciousness emergence in new field...");
    let emergence_level = NeuralExecutionEngine::test_field_consciousness_emergence((20, 20, 20, 50));
    println!("  Emergence level: {:.1}%", emergence_level * 100.0);
    
    if emergence_level > 0.4 {
        println!("  ✅ Consciousness emergence successful!");
    }
}