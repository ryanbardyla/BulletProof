//! Live Brain Training Binary
//! 
//! Runs the brain continuously, learning from real Redis data
//! while we build the native NeuronLang compiler.

use neuronlang_core::live_training_runner::start_live_training;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 === NEURONLANG LIVE BRAIN ===");
    println!("📡 Starting continuous learning from Redis...");
    println!("🔬 The brain will discover patterns while we build the language!");
    println!("⚡ Press Ctrl+C to stop and save progress\n");
    
    // Run the training loop
    match start_live_training().await {
        Ok(_) => {
            println!("\n✅ Training completed successfully!");
        }
        Err(e) => {
            println!("\n❌ Training error: {}", e);
            println!("💾 Checkpoint saved before exit");
        }
    }
    
    Ok(())
}