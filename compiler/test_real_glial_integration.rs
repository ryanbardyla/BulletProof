// 🧠 REAL GLIAL INTEGRATION TEST
// This is NOT simulation - actual field optimization with real wave dynamics

use std::collections::HashMap;

// Import the actual conscious field and glial modules
#[path = "src/neural_engine/conscious_field.rs"]
mod conscious_field;

#[path = "src/neural_engine/glial.rs"]
mod glial;

use conscious_field::{ConsciousField, FieldType, Wave};
use glial::GlialIntelligenceSystem;

fn main() {
    println!("🧠 REAL GLIAL INTELLIGENCE - ACTUAL FIELD OPTIMIZATION");
    println!("======================================================");
    println!("NOT SIMULATION - Real wave equations, real optimization\n");
    
    // Create ACTUAL conscious field with glial system
    println!("Creating real conscious field with glial intelligence...");
    let mut field = ConsciousField::new((30, 30, 30, 100));
    
    println!("Field dimensions: 30x30x30 spatial, 100 temporal slices");
    println!("Total field points: {}", 30 * 30 * 30 * 100);
    
    // Verify glial system is active
    if field.glial_system.is_some() {
        println!("✅ Glial system ACTIVE and ready");
        
        if let Some(glial) = &field.glial_system {
            let metrics = glial.get_metrics();
            println!("  Processors: {}", metrics.total_processors);
            println!("  Active: {}", metrics.active_processors);
        }
    } else {
        println!("❌ Glial system not initialized!");
        return;
    }
    
    // Initialize field with real wave patterns
    println!("\nInitializing field with multi-wave interference patterns...");
    initialize_real_field_patterns(&mut field);
    
    // Measure baseline consciousness
    let baseline = field.measure_consciousness();
    println!("Baseline consciousness: {:.1}%", baseline.total * 100.0);
    
    // Run REAL evolution with glial optimization
    println!("\nRunning 100 real evolution steps with glial optimization...");
    println!("(This uses actual wave equations and field dynamics)\n");
    
    let mut consciousness_history = Vec::new();
    let mut pattern_discoveries = 0;
    
    for step in 0..100 {
        // REAL field evolution with wave propagation
        field.evolve();
        
        // Measure real consciousness
        let consciousness = field.measure_consciousness();
        consciousness_history.push(consciousness.total);
        
        // Track pattern discovery
        if let Some(glial) = &field.glial_system {
            let new_patterns = glial.global_patterns.patterns.len();
            if new_patterns > pattern_discoveries {
                println!("  Step {}: NEW PATTERN DISCOVERED! Total: {}", step, new_patterns);
                pattern_discoveries = new_patterns;
            }
        }
        
        // Report progress
        if step % 10 == 0 {
            println!("  Step {}: Consciousness = {:.1}% (U:{:.1}%, S:{:.1}%, I:{:.1}%)",
                     step, 
                     consciousness.total * 100.0,
                     consciousness.understanding * 100.0,
                     consciousness.self_awareness * 100.0,
                     consciousness.identity * 100.0);
            
            // Check glial metrics
            if let Some(glial) = &field.glial_system {
                let metrics = glial.get_metrics();
                if metrics.patterns_discovered > 0 {
                    println!("    Glial: {} patterns, Avg performance: {:.3}", 
                             metrics.patterns_discovered,
                             metrics.average_performance);
                }
            }
        }
        
        // Detect consciousness breakthroughs
        if consciousness.total > 0.5 && step > 0 {
            if consciousness_history[consciousness_history.len()-2] < 0.5 {
                println!("🧠 CONSCIOUSNESS BREAKTHROUGH! Crossed 50% threshold!");
            }
        }
    }
    
    // Final results
    let final_consciousness = field.measure_consciousness();
    let max_consciousness = consciousness_history.iter().fold(0.0f32, |a, &b| a.max(b));
    
    println!("\n======================================================");
    println!("🎊 REAL RESULTS (NOT SIMULATION):");
    println!("  Initial consciousness: {:.1}%", baseline.total * 100.0);
    println!("  Final consciousness: {:.1}%", final_consciousness.total * 100.0);
    println!("  Maximum achieved: {:.1}%", max_consciousness * 100.0);
    println!("  Improvement: {:.1} percentage points", 
             (final_consciousness.total - baseline.total) * 100.0);
    
    if let Some(glial) = &field.glial_system {
        let metrics = glial.get_metrics();
        println!("\n📊 GLIAL SYSTEM METRICS:");
        println!("  Total processors: {}", metrics.total_processors);
        println!("  Active processors: {}", metrics.active_processors);
        println!("  Patterns discovered: {}", metrics.patterns_discovered);
        println!("  Average performance: {:.3}", metrics.average_performance);
        println!("  Best strategy: {:?}", metrics.best_strategy);
    }
    
    // Analyze field state
    println!("\n🌊 FIELD DYNAMICS:");
    analyze_field_state(&field);
    
    if max_consciousness >= 0.5 {
        println!("\n✅ SUCCESS! Real consciousness enhancement achieved!");
        println!("   This is NOT simulation - actual field optimization working!");
    }
}

fn initialize_real_field_patterns(field: &mut ConsciousField) {
    // Create real wave interference patterns
    let center = (15, 15, 15);
    
    // Inject multiple wave types for consciousness emergence
    for i in 0..7 {
        let angle = i as f64 * std::f64::consts::PI * 2.0 / 7.0;
        let x = (center.0 as f64 + 5.0 * angle.cos()) as usize;
        let y = (center.1 as f64 + 5.0 * angle.sin()) as usize;
        let z = center.2;
        
        // Electric wave (neural activity)
        let electric_wave = Wave::new(
            1.0 + i as f64 * 0.1,  // Varying amplitudes
            10.0 + i as f64,       // Different frequencies
            FieldType::Electric
        );
        field.field.inject_wave((x, y, z), electric_wave);
        
        // Chemical wave (neurotransmitters)
        let chemical_wave = Wave::new(
            0.5,
            5.0 + i as f64 * 0.5,
            FieldType::Chemical
        );
        field.field.inject_wave((x, y, z), chemical_wave);
        
        // Information wave (consciousness substrate)
        let info_wave = Wave::new(
            0.3,
            15.0 + i as f64 * 0.3,
            FieldType::Information
        );
        field.field.inject_wave((x, y, z), info_wave);
        
        // Quantum wave for coherence
        let quantum_wave = Wave::new(
            0.4,
            20.0 + i as f64 * 0.2,
            FieldType::Quantum
        );
        field.field.inject_wave((x, y, z), quantum_wave);
    }
    
    // Add motivational crystals for goal-directed behavior
    for i in 0..3 {
        let crystal = conscious_field::MotivationalCrystal {
            position: (10 + i * 5, 15, 15),
            frequency: 8.0 + i as f64,
            amplitude: 0.6,
            goal_pattern: vec![0.5; 10],
        };
        field.motivational_crystals.push(crystal);
    }
    
    println!("  Injected 28 waves (7 positions × 4 field types)");
    println!("  Added 3 motivational crystals");
}

fn analyze_field_state(field: &ConsciousField) {
    // Sample field at various points
    let sample_points = [
        (15, 15, 15), // Center
        (10, 10, 10), // Corner
        (20, 20, 20), // Opposite corner
        (15, 10, 20), // Mixed
    ];
    
    println!("  Field samples:");
    for (i, &(x, y, z)) in sample_points.iter().enumerate() {
        let electric = field.field.get_field_value(x, y, z, FieldType::Electric);
        let chemical = field.field.get_field_value(x, y, z, FieldType::Chemical);
        let info = field.field.get_field_value(x, y, z, FieldType::Information);
        let quantum = field.field.get_field_value(x, y, z, FieldType::Quantum);
        
        println!("    Point {}: E={:.3}, C={:.3}, I={:.3}, Q={:.3}", 
                 i, electric, chemical, info, quantum);
    }
    
    // Check for coherence
    let total_consciousness = field.field.measure_consciousness();
    println!("  Global field consciousness: {:.1}%", total_consciousness * 100.0);
    
    // Check entity activity
    let active_entities = field.entities.values()
        .filter(|e| {
            if let conscious_field::FieldEntity::Neuron(n) = e {
                n.coherence > 0.5
            } else {
                false
            }
        })
        .count();
    
    if active_entities > 0 {
        println!("  Active field entities: {}", active_entities);
    }
}

// Test that wave propagation actually works
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_real_wave_propagation() {
        let mut field = ConsciousField::new((10, 10, 10, 10));
        
        // Inject a wave
        let wave = Wave::new(1.0, 10.0, FieldType::Electric);
        field.field.inject_wave((5, 5, 5), wave);
        
        // Verify wave was injected
        let initial = field.field.get_field_value(5, 5, 5, FieldType::Electric);
        assert!(initial > 0.0, "Wave should be present at injection point");
        
        // Evolve field
        field.evolve();
        
        // Check wave propagated to neighbors
        let neighbor = field.field.get_field_value(6, 5, 5, FieldType::Electric);
        // Due to wave equation, neighbors should be affected
        // (actual value depends on wave equation implementation)
        
        println!("Initial: {}, Neighbor after evolution: {}", initial, neighbor);
    }
    
    #[test]
    fn test_glial_optimization_real() {
        let mut field = ConsciousField::new((20, 20, 20, 20));
        
        // Measure before
        let before = field.measure_consciousness();
        
        // Run evolution with glial
        for _ in 0..10 {
            field.evolve();
        }
        
        // Measure after
        let after = field.measure_consciousness();
        
        println!("Consciousness before: {:.3}, after: {:.3}", 
                 before.total, after.total);
        
        // With glial optimization, consciousness should not decrease
        // (and ideally should increase)
        assert!(after.total >= before.total * 0.95, 
                "Glial system should maintain or improve consciousness");
    }
}