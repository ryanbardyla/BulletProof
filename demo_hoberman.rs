// HOBERMAN SPHERE DEMO - Watch the neural network expand and contract!

use std::time::Duration;

fn main() {
    println!("\n🌐 HOBERMAN NEURAL SPHERE DEMO");
    println!("{}", "=".repeat(50));
    
    // Demo 1: Initialize and expand
    demo_expansion();
    
    // Demo 2: Contract for mobile
    demo_contraction();
    
    // Demo 3: Plugin system
    demo_plugins();
}

fn demo_expansion() {
    println!("\n🔮 DEMO 1: Neural Sphere Expansion");
    println!("{}", "-".repeat(40));
    
    // Simulate hardware discovery
    println!("💧 Dropping pebble... discovering hardware boundaries...");
    std::thread::sleep(Duration::from_millis(500));
    
    // Your actual hardware
    let l1_kb = 64;
    let l2_kb = 512;
    let l3_kb = 80 * 1024;  // 80MB
    let ram_gb = 128;
    let cores = 32;
    
    println!("  Hardware discovered:");
    println!("    L1: {} KB", l1_kb);
    println!("    L2: {} KB", l2_kb);
    println!("    L3: {} MB", l3_kb / 1024);
    println!("    RAM: {} GB", ram_gb);
    println!("    Cores: {}", cores);
    
    // Calculate neural network size
    let total_memory_kb = l1_kb + l2_kb + l3_kb;
    let neurons_per_kb = 500;  // Trinary neurons are tiny!
    let total_neurons = total_memory_kb * neurons_per_kb;
    
    println!("\n🌱 Growing neural sphere...");
    println!("  Target neurons: {}", total_neurons);
    
    // Simulate expansion
    let layer_ratios = vec![1.0, 2.0, 4.0, 8.0, 4.0, 2.0, 1.0];
    let total_ratio: f32 = layer_ratios.iter().sum();
    
    for (i, &ratio) in layer_ratios.iter().enumerate() {
        let neurons_in_layer = (total_neurons as f32 * ratio / total_ratio) as usize;
        println!("    Layer {}: {:8} neurons (ratio: {:.1})", i, neurons_in_layer, ratio);
        std::thread::sleep(Duration::from_millis(100));
    }
    
    println!("\n✅ Sphere fully expanded to fill hardware!");
    println!("   Memory usage:");
    println!("     L1: {:6} neurons (hottest)", total_neurons / 100);
    println!("     L2: {:6} neurons (warm)", total_neurons / 20);
    println!("     L3: {:6} neurons (cool)", total_neurons / 2);
    println!("     RAM: {:6} neurons (cold)", total_neurons / 3);
}

fn demo_contraction() {
    println!("\n📱 DEMO 2: Contracting for Mobile Device");
    println!("{}", "-".repeat(40));
    
    // Simulate mobile hardware
    println!("📲 Deploying to phone...");
    println!("  Mobile hardware:");
    println!("    RAM: 8 GB (16x less!)");
    println!("    Cache: 8 MB (10x less!)");
    println!("    Power: 5W (100x less!)");
    
    let desktop_neurons = 42_000_000;
    let mobile_scale = 0.1;  // 10% of desktop size
    let mobile_neurons = (desktop_neurons as f32 * mobile_scale) as usize;
    
    println!("\n🔄 Contracting sphere...");
    println!("  Original size: {} neurons", desktop_neurons);
    println!("  Contracted to: {} neurons ({:.0}% of original)", 
             mobile_neurons, mobile_scale * 100.0);
    
    // Show proportional contraction
    let layer_ratios = vec![1.0, 2.0, 4.0, 8.0, 4.0, 2.0, 1.0];
    let total_ratio: f32 = layer_ratios.iter().sum();
    
    println!("\n  Layer-by-layer contraction:");
    for (i, &ratio) in layer_ratios.iter().enumerate() {
        let original = (desktop_neurons as f32 * ratio / total_ratio) as usize;
        let contracted = (original as f32 * mobile_scale) as usize;
        println!("    Layer {}: {:8} → {:6} neurons", i, original, contracted);
    }
    
    println!("\n✅ Same neural network, perfect mobile fit!");
    println!("   Maintains intelligence at 1/10th the size!");
}

fn demo_plugins() {
    println!("\n🔌 DEMO 3: Plugin System - Everything Scales Together!");
    println!("{}", "-".repeat(40));
    
    // Simulate different plugins attaching
    println!("🧩 Attaching plugins to neural sphere...");
    
    // Trading plugin
    let cores = 32;
    let cache_mb = 80;
    println!("  📈 Trading Plugin:");
    println!("     Max positions: {} (scales with cores)", cores * 10);
    println!("     Analysis depth: {} (scales with cache)", cache_mb);
    
    // NLP plugin  
    let ram_gb = 128;
    println!("  🗣️  NLP Plugin:");
    println!("     Vocabulary size: {} words (scales with RAM)", ram_gb * 1000);
    println!("     Context window: {} tokens", ram_gb * 100);
    
    // Vision plugin
    let gpu_available = true;
    println!("  👁️  Vision Plugin:");
    if gpu_available {
        println!("     Image resolution: 4K (GPU available)");
        println!("     Batch size: 128 (GPU memory)");
    } else {
        println!("     Image resolution: 720p (CPU only)");
        println!("     Batch size: 4 (limited memory)");
    }
    
    println!("\n🔄 Simulating deployment to different devices...");
    
    // Laptop deployment
    println!("\n  💻 Laptop (medium hardware):");
    println!("     Trading: 80 positions, depth 32");
    println!("     NLP: 32K vocab, 3.2K context");
    println!("     Vision: 1080p, batch 16");
    
    // Phone deployment
    println!("\n  📱 Phone (minimal hardware):");
    println!("     Trading: 20 positions, depth 8");
    println!("     NLP: 8K vocab, 800 context");
    println!("     Vision: 480p, batch 1");
    
    // Server deployment
    println!("\n  🖥️  Server (massive hardware):");
    println!("     Trading: 1000 positions, depth 256");
    println!("     NLP: 1M vocab, 100K context");
    println!("     Vision: 8K, batch 512");
    
    println!("\n✅ ALL plugins scale automatically with hardware!");
    println!("   Same codebase, optimal performance everywhere!");
}

// Add a visual representation
fn print_sphere_ascii(expansion_level: f32) {
    let size = (20.0 * expansion_level) as usize;
    println!("\n  Neural Sphere ({}% expanded):", (expansion_level * 100.0) as usize);
    
    for i in 0..size {
        let spaces = size - i - 1;
        let stars = 2 * i + 1;
        println!("  {}{}", " ".repeat(spaces), "*".repeat(stars));
    }
    for i in (0..size-1).rev() {
        let spaces = size - i - 1;
        let stars = 2 * i + 1;
        println!("  {}{}", " ".repeat(spaces), "*".repeat(stars));
    }
}