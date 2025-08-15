// Simple test to verify MNIST loading works
use anyhow::Result;

mod data_loader;

fn main() -> Result<()> {
    println!("Testing MNIST data loading...");
    
    let (train_data, test_data) = data_loader::load_real_mnist()?;
    
    println!("✅ Successfully loaded MNIST!");
    println!("   Training samples: {}", train_data.len());
    println!("   Test samples: {}", test_data.len());
    
    if !train_data.is_empty() {
        println!("   Input dimensions: {}", train_data[0].0.len());
        println!("   Output dimensions: {}", train_data[0].1.len());
    }
    
    Ok(())
}