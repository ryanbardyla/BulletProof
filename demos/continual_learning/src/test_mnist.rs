// Simple test to verify MNIST loading works
use anyhow::Result;

mod data_loader;

fn main() -> Result<()> {
    println!("🧪 Testing MNIST data loading...");
    
    let (train_data, test_data) = data_loader::load_real_mnist()?;
    
    println!("✅ Successfully loaded MNIST!");
    println!("   Training samples: {}", train_data.len());
    println!("   Test samples: {}", test_data.len());
    
    if !train_data.is_empty() {
        println!("   Input dimensions: {}", train_data[0].0.len());
        println!("   Output dimensions: {}", train_data[0].1.len());
        
        // Check the trinary conversion worked
        let sample_input = &train_data[0].0;
        let unique_values: std::collections::HashSet<_> = sample_input.iter()
            .map(|&x| (x * 100.0) as i32)  // Multiply to avoid float precision issues
            .collect();
        
        println!("   Unique trinary values: {:?}", unique_values);
        println!("   Sample label: {:?}", train_data[0].1);
    }
    
    Ok(())
}