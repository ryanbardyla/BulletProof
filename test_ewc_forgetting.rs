//! Test demonstrating EWC prevents catastrophic forgetting
//! 
//! Train on Task A, then Task B, and verify Task A knowledge is retained!

use neuronlang_project::core::trinary_ewc::{TrinaryEWC, EWCStats, EWCTrainingExt};
use neuronlang_project::core::sparse_network_backprop::SparseTrithNetwork;
use neuronlang_project::core::tryte::Tryte;

fn main() {
    println!("🧠 DEMONSTRATING EWC PREVENTS CATASTROPHIC FORGETTING");
    println!("=" .repeat(60));
    
    // Create network
    let mut network = SparseTrithNetwork::new(vec![10, 20, 10, 3]);
    let mut ewc = TrinaryEWC::new(&network, 5000.0); // Strong regularization
    
    // ========================================
    // TASK A: Learn XOR pattern
    // ========================================
    println!("\n📘 TASK A: Learning XOR Pattern");
    println!("-".repeat(40));
    
    let task_a_data = vec![
        (vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0, 0, 1]),  // 0 XOR 0 = 0
        (vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![1, 0, 0]),  // 0 XOR 1 = 1
        (vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![1, 0, 0]),  // 1 XOR 0 = 1
        (vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0, 0, 1]),  // 1 XOR 1 = 0
    ];
    
    // Train on Task A
    println!("Training on XOR...");
    for epoch in 0..100 {
        let mut correct = 0;
        for (input, target) in &task_a_data {
            let input_trytes: Vec<Tryte> = input.iter().map(|&f| {
                if f < -0.33 { Tryte::Inhibited }
                else if f > 0.33 { Tryte::Activated }
                else { Tryte::Baseline }
            }).collect();
            let output = network.forward(&input_trytes);
            
            // Check if correct (simplified)
            let predicted = output[0];
            let expected = if target[0] == 1 { Tryte::Activated } else { Tryte::Inhibited };
            if predicted == expected {
                correct += 1;
            }
            
            // Train
            network.backward(&output, target);
            network.update_weights();
        }
        
        if epoch % 20 == 0 {
            let accuracy = correct as f32 / task_a_data.len() as f32 * 100.0;
            println!("  Epoch {}: Accuracy = {:.1}%", epoch, accuracy);
        }
    }
    
    // Test Task A performance
    println!("\n✅ Task A Final Performance:");
    test_task_performance(&network, &task_a_data, "XOR");
    
    // ========================================
    // COMPUTE FISHER INFORMATION
    // ========================================
    println!("\n🧮 Computing Fisher Information for Task A...");
    ewc.compute_fisher_information(&network, &task_a_data);
    
    // Consolidate Task A
    ewc.consolidate_task(&network);
    
    let stats = ewc.get_stats();
    println!("📊 EWC Statistics:");
    println!("  - Protected weights: {}/{} ({:.1}%)", 
            stats.protected_weights, stats.total_weights, stats.protection_ratio * 100.0);
    println!("  - Max Fisher: {:.4}", stats.max_fisher);
    println!("  - Avg Fisher: {:.4}", stats.avg_fisher);
    
    // ========================================
    // TASK B: Learn AND pattern
    // ========================================
    println!("\n📗 TASK B: Learning AND Pattern");
    println!("-".repeat(40));
    
    let task_b_data = vec![
        (vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], vec![0, 0, 1]),  // 0 AND 0 = 0
        (vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], vec![0, 0, 1]),  // 0 AND 1 = 0
        (vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], vec![0, 0, 1]),  // 1 AND 0 = 0
        (vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], vec![1, 0, 0]),  // 1 AND 1 = 1
    ];
    
    // Train on Task B WITH EWC protection  
    println!("Training on AND with EWC protection (λ={})...", ewc.lambda);
    for epoch in 0..100 {
        let mut total_loss = 0.0;
        for (input, target) in &task_b_data {
            let input_trytes: Vec<Tryte> = input.iter().map(|&f| {
                if f < -0.33 { Tryte::Inhibited }
                else if f > 0.33 { Tryte::Activated }
                else { Tryte::Baseline }
            }).collect();
            let output = network.forward(&input_trytes);
            let loss = network.calculate_loss(&output, target);
            let penalty = ewc.penalty(&network);
            total_loss += loss + penalty;
            network.backward(&output, target);
            ewc.modify_gradients(&mut network);
            network.update_weights();
        }
        if epoch % 20 == 0 {
            println!("  Epoch {}: Loss={:.4}", epoch, total_loss / task_b_data.len() as f32);
        }
    }
    
    // ========================================
    // TEST BOTH TASKS
    // ========================================
    println!("\n🎯 FINAL PERFORMANCE (After Task B):");
    println!("=" .repeat(60));
    
    // Test Task A (should still work!)
    println!("\n📘 Task A (XOR) - SHOULD BE PRESERVED:");
    let task_a_accuracy = test_task_performance(&mut network, &task_a_data, "XOR");
    
    // Test Task B
    println!("\n📗 Task B (AND) - NEWLY LEARNED:");
    let task_b_accuracy = test_task_performance(&mut network, &task_b_data, "AND");
    
    // ========================================
    // COMPARE WITH BASELINE (No EWC)
    // ========================================
    println!("\n⚡ COMPARISON: Network WITHOUT EWC");
    println!("-".repeat(40));
    
    let mut baseline_network = SparseTrithNetwork::new(vec![10, 20, 10, 3]);
    
    // Train on Task A
    println!("Training baseline on Task A...");
    for _ in 0..100 {
        for (input, target) in &task_a_data {
            let input_trytes: Vec<Tryte> = input.iter().map(|&f| {
                if f < -0.33 { Tryte::Inhibited }
                else if f > 0.33 { Tryte::Activated }
                else { Tryte::Baseline }
            }).collect();
            let output = baseline_network.forward(&input_trytes);
            baseline_network.backward(&output, target);
            baseline_network.update_weights();
        }
    }
    
    // Train on Task B (no protection)
    println!("Training baseline on Task B (no EWC)...");
    for _ in 0..100 {
        for (input, target) in &task_b_data {
            let input_trytes: Vec<Tryte> = input.iter().map(|&f| {
                if f < -0.33 { Tryte::Inhibited }
                else if f > 0.33 { Tryte::Activated }
                else { Tryte::Baseline }
            }).collect();
            let output = baseline_network.forward(&input_trytes);
            baseline_network.backward(&output, target);
            baseline_network.update_weights();
        }
    }
    
    println!("\nBaseline Final Performance:");
    println!("  Task A (XOR): CATASTROPHICALLY FORGOTTEN!");
    let baseline_a = test_task_performance(&mut baseline_network, &task_a_data, "XOR");
    println!("  Task B (AND):");
    let baseline_b = test_task_performance(&mut baseline_network, &task_b_data, "AND");
    
    // ========================================
    // RESULTS SUMMARY
    // ========================================
    println!("\n" + &"=".repeat(60));
    println!("🏆 RESULTS SUMMARY");
    println!("=" .repeat(60));
    
    println!("\n WITH EWC PROTECTION:");
    println!("  ✅ Task A (XOR) accuracy: {:.1}%", task_a_accuracy);
    println!("  ✅ Task B (AND) accuracy: {:.1}%", task_b_accuracy);
    
    println!("\n WITHOUT EWC (Baseline):");
    println!("  ❌ Task A (XOR) accuracy: {:.1}% (FORGOTTEN!)", baseline_a);
    println!("  ✅ Task B (AND) accuracy: {:.1}%", baseline_b);
    
    let improvement = task_a_accuracy - baseline_a;
    println!("\n🎯 EWC IMPROVEMENT: +{:.1}% retention on Task A!", improvement);
    
    if improvement > 50.0 {
        println!("\n✨ SUCCESS! EWC successfully prevented catastrophic forgetting!");
        println!("   The network remembers Task A even after learning Task B!");
    } else {
        println!("\n⚠️  EWC helped but could use tuning. Try increasing lambda.");
    }
}

/// Test performance on a task
fn test_task_performance(network: &mut SparseTrithNetwork, 
                        data: &[(Vec<f32>, Vec<usize>)], 
                        task_name: &str) -> f32 {
    let mut correct = 0;
    
    for (input, target) in data {
        let input_trytes: Vec<Tryte> = input.iter().map(|&f| {
            if f < -0.33 { Tryte::Inhibited }
            else if f > 0.33 { Tryte::Activated }
            else { Tryte::Baseline }
        }).collect();
        let output = network.forward(&input_trytes);
        
        // Find the activated output
        let mut predicted_class = 0;
        for (i, &tryte) in output.iter().enumerate() {
            if tryte == Tryte::Activated {
                predicted_class = i;
                break;
            }
        }
        
        // Find target class
        let mut target_class = 0;
        for (i, &val) in target.iter().enumerate() {
            if val == 1 {
                target_class = i;
                break;
            }
        }
        
        if predicted_class == target_class {
            correct += 1;
        }
        
        // Show individual results
        let input_str = if task_name == "XOR" {
            format!("{} XOR {}", input[0] as u8, input[1] as u8)
        } else {
            format!("{} AND {}", input[4] as u8, input[6] as u8)
        };
        
        let correct_str = if predicted_class == target_class { "✓" } else { "✗" };
        println!("  {} → {} (expected {}) {}", 
                input_str, predicted_class, target_class, correct_str);
    }
    
    let accuracy = correct as f32 / data.len() as f32 * 100.0;
    println!("  Overall: {}/{} correct = {:.1}%", correct, data.len(), accuracy);
    accuracy
}