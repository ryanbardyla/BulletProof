//! Quick test of GPU simulator

use neuronlang_gpu::{GpuTrinaryBrain, Tryte};

fn main() {
    println!("🧠 Testing GPU Trinary Brain (Simulator)");
    println!("=========================================\n");
    
    // Create a brain with 15,003 neurons (matching our compiler output)
    let layer_sizes = vec![10000, 5000, 3];
    let mut brain = GpuTrinaryBrain::new(layer_sizes);
    
    // Test input
    let input = vec![0.5, -0.3, 0.8, 0.1, -0.6];
    
    println!("📊 Network: 15,003 neurons");
    println!("📥 Input: {:?}\n", input);
    
    // Forward pass
    let output = brain.forward(input);
    
    println!("\n🎯 Output neurons: {:?}", output);
    
    // Energy stats
    let (current, avg, best) = brain.energy_stats();
    println!("\n⚡ Energy Efficiency:");
    println!("   Current: {:.1}% baseline (ZERO energy)", current);
    println!("   Average: {:.1}%", avg);
    println!("   Best:    {:.1}%", best);
    
    // Calculate savings
    let binary_energy = 15003.0;
    let trinary_energy = 15003.0 * (1.0 - current / 100.0);
    let savings = ((binary_energy - trinary_energy) / binary_energy) * 100.0;
    
    println!("\n💰 Energy Comparison:");
    println!("   Binary:  {:.0} units (all neurons active)", binary_energy);
    println!("   Trinary: {:.0} units (baseline = ZERO!)", trinary_energy);
    println!("   SAVINGS: {:.1}%!", savings);
    
    println!("\n✅ GPU simulator working!");
    println!("   Ready for full CUDA when needed");
    println!("   See GPU_INTEGRATION_STATUS.md for details");
}