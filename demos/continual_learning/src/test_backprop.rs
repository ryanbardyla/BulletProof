// Test to verify REAL backpropagation is working
use neuronlang_project::core::sparse_network_backprop::SparseTrithNetwork;
use neuronlang_project::core::tryte::Tryte;
use colored::*;

fn main() {
    println!("\n{}", "🧬 TESTING REAL BACKPROPAGATION 🧬".bright_yellow().bold());
    println!("{}", "="*50);
    
    // Create a trinary network
    let mut network = SparseTrithNetwork::new(vec![784, 256, 128, 10]);
    println!("✅ Created network: 784 → 256 → 128 → 10");
    
    // Verify initial sparsity
    let initial_sparsity = network.get_sparsity();
    println!("📊 Initial sparsity: {:.1}%", initial_sparsity * 100.0);
    
    // Create training data (simple pattern)
    let mut input = vec![Tryte::Baseline; 784];
    // Set some pixels to activated (simulate digit "1")
    for i in 390..410 {
        input[i] = Tryte::Activated;
    }
    
    // Target: class 1 (one-hot)
    let target = vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0];
    
    println!("\n{}", "Training for 10 epochs...".bright_cyan());
    
    let mut losses = Vec::new();
    for epoch in 0..10 {
        // Forward pass
        let output = network.forward(&input);
        
        // Backward pass with REAL gradients
        let loss = network.backward(&output, &target);
        losses.push(loss);
        
        // Update protein synthesis
        network.update_protein_synthesis(0.1);
        
        // Check prediction
        let predicted_class = output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                network.tryte_to_float(**a)
                    .partial_cmp(&network.tryte_to_float(**b))
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        println!("  Epoch {}: Loss = {:.4}, Predicted = {}, Target = 1", 
                 epoch + 1, loss, predicted_class);
    }
    
    // Verify learning occurred
    let final_sparsity = network.get_sparsity();
    let loss_decreased = losses.first().unwrap() > losses.last().unwrap();
    
    println!("\n{}", "="*50);
    println!("{}", "📈 RESULTS".bright_green().bold());
    println!("  Initial loss: {:.4}", losses.first().unwrap());
    println!("  Final loss: {:.4}", losses.last().unwrap());
    println!("  Loss decreased: {}", 
             if loss_decreased { "✅ YES".green() } else { "❌ NO".red() });
    println!("  Final sparsity: {:.1}%", final_sparsity * 100.0);
    println!("  Protein levels active: {}", 
             if network.protein_levels.iter().any(|&p| p > 0.0) { 
                 "✅ YES".green() 
             } else { 
                 "❌ NO".red() 
             });
    
    // Verify weights are still trinary
    let mut trinary_count = 0;
    let mut total_weights = 0;
    for layer in &network.layers {
        for row in &layer.weights {
            for &w in row {
                total_weights += 1;
                if w == -1.0 || w == 0.0 || w == 1.0 {
                    trinary_count += 1;
                }
            }
        }
    }
    
    println!("  Trinary constraint maintained: {}/{} ({:.1}%)", 
             trinary_count, total_weights, 
             trinary_count as f32 / total_weights as f32 * 100.0);
    
    if loss_decreased && trinary_count == total_weights {
        println!("\n{}", "🎊 SUCCESS: Real backpropagation with trinary constraints! 🎊".bright_green().bold());
    } else {
        println!("\n{}", "⚠️ Issues detected - check implementation".yellow());
    }
}