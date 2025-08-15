// Test the forward pass with real MNIST data
use anyhow::Result;
use neuronlang_project::core::sparse_network_backprop::SparseTrithNetwork;

mod data_loader;

fn main() -> Result<()> {
    println!("🧪 Testing forward pass with real MNIST data...");
    
    // Load a small sample of MNIST data
    let (train_data, _) = data_loader::load_real_mnist()?;
    
    println!("✅ Loaded {} training samples", train_data.len());
    
    // Create a small network for testing (784 -> 128 -> 10)
    let mut network = SparseTrithNetwork::new(vec![784, 128, 10]);
    
    // Test with a batch of 4 samples
    let test_batch: Vec<Vec<f32>> = train_data[0..4].iter()
        .map(|(input, _)| input.to_vec())
        .collect();
    
    println!("📊 Input batch shape: {} samples × {} features", test_batch.len(), test_batch[0].len());
    
    // Run forward pass
    let outputs = network.forward_batch(&test_batch);
    
    println!("✅ Forward pass completed!");
    println!("📤 Output batch shape: {} samples × {} classes", outputs.len(), outputs[0].len());
    
    // Check outputs
    for (i, output) in outputs.iter().enumerate() {
        println!("Sample {}: {:?}", i, output);
        
        // Find predicted class
        let predicted_class = output.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
            
        println!("  → Predicted class: {}", predicted_class);
    }
    
    println!("🎉 Forward pass test successful!");
    Ok(())
}