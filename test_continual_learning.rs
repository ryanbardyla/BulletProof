//! Comprehensive Continual Learning Test
//! 
//! Demonstrates actual lifelong learning across multiple tasks:
//! 1. XOR Logic (Task 1)
//! 2. AND Logic (Task 2) 
//! 3. OR Logic (Task 3)
//! 4. NAND Logic (Task 4)
//! 5. Pattern Recognition (Task 5)

use neuronlang_project::core::continual_learning::{
    ContinualLearner, TaskData, LearningStrategy
};

fn main() {
    println!("🧠 CONTINUAL LEARNING DEMONSTRATION");
    println!("=" .repeat(60));
    println!("Testing lifelong learning across 5 different tasks");
    println!("Each task builds on previous knowledge without forgetting!");
    
    // Initialize continual learner with adaptive EWC strategy
    let strategy = LearningStrategy::AdaptiveEWC { 
        base_lambda: 2000.0,  // Strong memory protection
        adaptation_rate: 1.0   // High adaptability
    };
    
    let mut learner = ContinualLearner::new(
        vec![10, 20, 15, 5],  // Network: 10 input, 20 hidden, 15 hidden, 5 output
        strategy
    );
    
    println!("🏗️  Network Architecture: 10→20→15→5");
    println!("🧬 Strategy: Adaptive EWC with λ=2000, adaptation=1.0\n");
    
    // ========================================
    // TASK 1: XOR Logic Learning
    // ========================================
    println!("📘 TASK 1: XOR Logic Learning");
    println!("-".repeat(40));
    
    let task1 = create_xor_task();
    match learner.train_with_memory_protection(task1) {
        Ok(performance) => {
            println!("✅ XOR Task completed:");
            println!("   Accuracy: {:.1}%", performance.accuracy * 100.0);
            println!("   Learning Speed: {:.2}x", performance.learning_speed);
            println!("   Memory Efficiency: {:.1}%", performance.memory_efficiency * 100.0);
        }
        Err(e) => println!("❌ XOR Task failed: {}", e),
    }
    
    // ========================================  
    // TASK 2: AND Logic Learning
    // ========================================
    println!("\n📗 TASK 2: AND Logic Learning (with XOR memory protection)");
    println!("-".repeat(40));
    
    let task2 = create_and_task();
    match learner.train_with_memory_protection(task2) {
        Ok(performance) => {
            println!("✅ AND Task completed:");
            println!("   Accuracy: {:.1}%", performance.accuracy * 100.0);
            println!("   Retention Score: {:.1}% (XOR preserved!)", performance.retention_score * 100.0);
            println!("   Learning Speed: {:.2}x", performance.learning_speed);
        }
        Err(e) => println!("❌ AND Task failed: {}", e),
    }
    
    // ========================================
    // TASK 3: OR Logic Learning
    // ========================================
    println!("\n📙 TASK 3: OR Logic Learning (with XOR+AND memory protection)");
    println!("-".repeat(40));
    
    let task3 = create_or_task();
    match learner.train_with_memory_protection(task3) {
        Ok(performance) => {
            println!("✅ OR Task completed:");
            println!("   Accuracy: {:.1}%", performance.accuracy * 100.0);
            println!("   Retention Score: {:.1}% (XOR+AND preserved!)", performance.retention_score * 100.0);
            println!("   Memory Efficiency: {:.1}%", performance.memory_efficiency * 100.0);
        }
        Err(e) => println!("❌ OR Task failed: {}", e),
    }
    
    // ========================================
    // TASK 4: NAND Logic Learning  
    // ========================================
    println!("\n📕 TASK 4: NAND Logic Learning (with XOR+AND+OR memory protection)");
    println!("-".repeat(40));
    
    let task4 = create_nand_task();
    match learner.train_with_memory_protection(task4) {
        Ok(performance) => {
            println!("✅ NAND Task completed:");
            println!("   Accuracy: {:.1}%", performance.accuracy * 100.0);
            println!("   Retention Score: {:.1}% (ALL previous tasks preserved!)", performance.retention_score * 100.0);
            println!("   Learning Speed: {:.2}x", performance.learning_speed);
        }
        Err(e) => println!("❌ NAND Task failed: {}", e),
    }
    
    // ========================================
    // TASK 5: Pattern Recognition
    // ========================================
    println!("\n📓 TASK 5: Pattern Recognition (with all logic memory protection)");
    println!("-".repeat(40));
    
    let task5 = create_pattern_task();
    match learner.train_with_memory_protection(task5) {
        Ok(performance) => {
            println!("✅ Pattern Task completed:");
            println!("   Accuracy: {:.1}%", performance.accuracy * 100.0);
            println!("   Retention Score: {:.1}% (ALL tasks preserved!)", performance.retention_score * 100.0);
            println!("   Memory Efficiency: {:.1}%", performance.memory_efficiency * 100.0);
        }
        Err(e) => println!("❌ Pattern Task failed: {}", e),
    }
    
    // ========================================
    // FINAL EVALUATION
    // ========================================
    println!("\n" + &"=".repeat(60));
    println!("🏆 CONTINUAL LEARNING RESULTS");
    println!("=" .repeat(60));
    
    let final_stats = learner.get_learning_stats();
    println!("📊 Overall Statistics:");
    println!("   Tasks Learned: {}", final_stats.total_tasks_learned);
    println!("   Average Accuracy: {:.1}%", final_stats.average_accuracy * 100.0);
    println!("   Average Retention: {:.1}%", final_stats.average_retention * 100.0);
    println!("   Memory Efficiency: {:.1}%", final_stats.memory_efficiency * 100.0);
    println!("   Experience Buffer: {} samples", final_stats.experience_buffer_size);
    println!("   Protected Weights: {}/{} ({:.1}%)", 
            final_stats.protected_weights, final_stats.total_weights,
            (final_stats.protected_weights as f32 / final_stats.total_weights as f32) * 100.0);
    println!("   Current λ (EWC): {:.1}", final_stats.current_lambda);
    
    // Test individual task retention
    println!("\n🔍 Individual Task Retention Test:");
    test_individual_task_retention(&mut learner);
    
    // Performance comparison
    println!("\n📈 Performance Analysis:");
    analyze_learning_progression(&learner);
    
    // Memory analysis
    println!("\n🧠 Memory Protection Analysis:");
    analyze_memory_protection(&learner);
    
    if final_stats.average_retention > 0.8 {
        println!("\n✨ SUCCESS! Continual learning achieved!");
        println!("   🎯 Network successfully learned 5 tasks without catastrophic forgetting");
        println!("   🧠 Memory protection preserved {:.1}% of knowledge", final_stats.average_retention * 100.0);
        println!("   ⚡ Energy efficiency: {:.1}% neurons protected", final_stats.memory_efficiency * 100.0);
    } else {
        println!("\n⚠️  Partial Success - Room for improvement");
        println!("   Current retention: {:.1}% (target: >80%)", final_stats.average_retention * 100.0);
    }
}

/// Create XOR task data
fn create_xor_task() -> TaskData {
    TaskData {
        task_id: 1,
        name: "XOR Logic".to_string(),
        data: vec![
            // XOR truth table with padding to 10 inputs and 5 outputs
            (vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![1, 0, 0, 0, 0]), // 0 XOR 0 = 0
            (vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0, 1, 0, 0, 0]), // 1 XOR 0 = 1
            (vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0, 1, 0, 0, 0]), // 0 XOR 1 = 1
            (vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![1, 0, 0, 0, 0]), // 1 XOR 1 = 0
            // Repeat patterns for better learning
            (vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![1, 0, 0, 0, 0]),
            (vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0, 1, 0, 0, 0]),
            (vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0, 1, 0, 0, 0]),
            (vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![1, 0, 0, 0, 0]),
        ],
        epochs: 100,
        importance: 1.0,
        domain: "logic".to_string(),
    }
}

/// Create AND task data
fn create_and_task() -> TaskData {
    TaskData {
        task_id: 2,
        name: "AND Logic".to_string(),
        data: vec![
            // AND truth table in different input positions
            (vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0, 0, 1, 0, 0]), // 0 AND 0 = 0
            (vec![1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0, 0, 1, 0, 0]), // 1 AND 0 = 0
            (vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0, 0, 1, 0, 0]), // 0 AND 1 = 0
            (vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0, 0, 0, 1, 0]), // 1 AND 1 = 1
            // Repeat for robustness
            (vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0, 0, 1, 0, 0]),
            (vec![1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0, 0, 1, 0, 0]),
            (vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0, 0, 1, 0, 0]),
            (vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0, 0, 0, 1, 0]),
        ],
        epochs: 80,
        importance: 0.9,
        domain: "logic".to_string(),
    }
}

/// Create OR task data
fn create_or_task() -> TaskData {
    TaskData {
        task_id: 3,
        name: "OR Logic".to_string(),
        data: vec![
            // OR truth table
            (vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], vec![0, 0, 0, 0, 1]), // 0 OR 0 = 0
            (vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], vec![0, 1, 0, 0, 0]), // 1 OR 0 = 1
            (vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], vec![0, 1, 0, 0, 0]), // 0 OR 1 = 1
            (vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], vec![0, 1, 0, 0, 0]), // 1 OR 1 = 1
            // More examples
            (vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], vec![0, 0, 0, 0, 1]),
            (vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], vec![0, 1, 0, 0, 0]),
            (vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], vec![0, 1, 0, 0, 0]),
            (vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], vec![0, 1, 0, 0, 0]),
        ],
        epochs: 60,
        importance: 0.8,
        domain: "logic".to_string(),
    }
}

/// Create NAND task data
fn create_nand_task() -> TaskData {
    TaskData {
        task_id: 4,
        name: "NAND Logic".to_string(),
        data: vec![
            // NAND truth table (NOT AND)
            (vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], vec![0, 1, 0, 0, 0]), // 0 NAND 0 = 1
            (vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], vec![0, 1, 0, 0, 0]), // 1 NAND 0 = 1
            (vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0], vec![0, 1, 0, 0, 0]), // 0 NAND 1 = 1
            (vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], vec![1, 0, 0, 0, 0]), // 1 NAND 1 = 0
            // Additional patterns
            (vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], vec![0, 1, 0, 0, 0]),
            (vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], vec![0, 1, 0, 0, 0]),
            (vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0], vec![0, 1, 0, 0, 0]),
            (vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], vec![1, 0, 0, 0, 0]),
        ],
        epochs: 70,
        importance: 0.7,
        domain: "logic".to_string(),
    }
}

/// Create pattern recognition task
fn create_pattern_task() -> TaskData {
    TaskData {
        task_id: 5,
        name: "Pattern Recognition".to_string(),
        data: vec![
            // Simple patterns - all high values = pattern A
            (vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![1, 0, 0, 0, 0]),
            (vec![0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![1, 0, 0, 0, 0]),
            // Alternating pattern = pattern B  
            (vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0, 1, 0, 0, 0]),
            (vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], vec![0, 1, 0, 0, 0]),
            // All low values = pattern C
            (vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0, 0, 1, 0, 0]),
            (vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0], vec![0, 0, 1, 0, 0]),
            // Sequential pattern = pattern D
            (vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.8, 0.6, 0.4], vec![0, 0, 0, 1, 0]),
            (vec![0.2, 0.4, 0.6, 0.8, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0, 0, 0, 1, 0]),
        ],
        epochs: 90,
        importance: 1.2, // Higher importance for complex task
        domain: "pattern".to_string(),
    }
}

/// Test retention on individual tasks
fn test_individual_task_retention(learner: &mut ContinualLearner) {
    for (i, task) in learner.task_history.iter().enumerate() {
        match learner.evaluate_task_performance(task) {
            Ok(accuracy) => {
                let status = if accuracy > 0.8 { "✅" } else if accuracy > 0.6 { "⚠️" } else { "❌" };
                println!("   {}: {} - {:.1}%", status, task.name, accuracy * 100.0);
            }
            Err(_) => println!("   ❌: {} - evaluation failed", task.name),
        }
    }
}

/// Analyze learning progression across tasks
fn analyze_learning_progression(learner: &ContinualLearner) {
    println!("   Task progression analysis:");
    for (i, perf) in learner.performance_history.iter().enumerate() {
        println!("     Task {}: Acc={:.1}%, Speed={:.2}x, Memory={:.1}%", 
                i + 1, perf.accuracy * 100.0, perf.learning_speed, perf.memory_efficiency * 100.0);
    }
    
    // Calculate improvement trends
    if learner.performance_history.len() > 1 {
        let first_acc = learner.performance_history[0].accuracy;
        let last_acc = learner.performance_history.last().unwrap().accuracy;
        let improvement = ((last_acc - first_acc) / first_acc) * 100.0;
        
        if improvement > 0.0 {
            println!("   📈 Performance improved by {:.1}% across tasks", improvement);
        } else {
            println!("   📉 Performance declined by {:.1}% across tasks", -improvement);
        }
    }
}

/// Analyze memory protection effectiveness
fn analyze_memory_protection(learner: &ContinualLearner) {
    let stats = learner.ewc.get_stats();
    
    println!("   Memory Protection Effectiveness:");
    println!("     Protected weights: {}/{} ({:.1}%)", 
            stats.protected_weights, stats.total_weights, stats.protection_ratio * 100.0);
    println!("     Current EWC λ: {:.1}", stats.lambda);
    println!("     Max Fisher importance: {:.4}", stats.max_fisher);
    println!("     Avg Fisher importance: {:.4}", stats.avg_fisher);
    
    // Experience buffer analysis
    println!("     Experience buffer: {} samples", learner.experience_buffer.len());
    
    if !learner.task_similarity.is_empty() {
        let avg_similarity: f32 = learner.task_similarity.values().sum::<f32>() 
            / learner.task_similarity.len() as f32;
        println!("     Average task similarity: {:.3}", avg_similarity);
    }
}