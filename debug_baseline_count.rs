// DEBUG BASELINE COUNTING - FIND THE 94.1% LOCK
// Minimal test to understand the exact calculation

fn main() {
    let l1_size = 10000;   // 1% 
    let l2_size = 50000;   // 5%
    let l3_size = 500000;  // 50%
    let ram_size = 440000; // 44%
    let total_neurons = l1_size + l2_size + l3_size + ram_size; // 1,000,000
    
    println!("🔍 DEBUGGING BULLETPROOF EFFICIENCY CALCULATION");
    println!("Network composition:");
    println!("  L1:  {:6} ({:.1}%)", l1_size, (l1_size as f64 / total_neurons as f64) * 100.0);
    println!("  L2:  {:6} ({:.1}%)", l2_size, (l2_size as f64 / total_neurons as f64) * 100.0);
    println!("  L3:  {:6} ({:.1}%)", l3_size, (l3_size as f64 / total_neurons as f64) * 100.0);
    println!("  RAM: {:6} ({:.1}%)", ram_size, (ram_size as f64 / total_neurons as f64) * 100.0);
    println!("  TOTAL: {}", total_neurons);
    
    // Test different baseline scenarios
    println!("\n🧮 TESTING BASELINE SCENARIOS:");
    
    // Scenario 1: Everything starts at baseline (should be 100%)
    test_scenario("All baseline", l1_size, l2_size, l3_size, ram_size, total_neurons);
    
    // Scenario 2: Some L1 active (10% of L1)
    test_scenario("10% L1 active", l1_size - 1000, l2_size, l3_size, ram_size, total_neurons);
    
    // Scenario 3: Some L1+L2 active 
    test_scenario("10% L1+L2 active", l1_size - 1000, l2_size - 5000, l3_size, ram_size, total_neurons);
    
    // Scenario 4: What creates exactly 94.1%?
    let target_baseline = (941000.0 / 10.0) as usize; // 94.1% of 1M
    let non_baseline = total_neurons - target_baseline;
    println!("\n🎯 For exactly 94.1% efficiency:");
    println!("  Need {} baseline neurons", target_baseline);
    println!("  Need {} active neurons", non_baseline);
    println!("  941000 / 1000000 = {:.1}%", (target_baseline as f64 / total_neurons as f64) * 100.0);
    
    // Now test what our current logic produces
    println!("\n🔍 CURRENT ALGORITHM SIMULATION:");
    
    // Simulate current L3 logic: only processes 100, rest counted as baseline
    let l3_processed = 100;
    let l3_unprocessed = l3_size - l3_processed;
    
    println!("L3 Analysis:");
    println!("  Processed: {} neurons", l3_processed);
    println!("  Unprocessed (auto-baseline): {} neurons", l3_unprocessed);
    println!("  Unprocessed percentage: {:.1}%", (l3_unprocessed as f64 / total_neurons as f64) * 100.0);
    
    // If L3 unprocessed is mostly baseline...
    let l3_baseline_auto = l3_unprocessed; // All unprocessed = baseline
    let efficiency_from_l3_alone = (l3_baseline_auto as f64 / total_neurons as f64) * 100.0;
    println!("  Efficiency from L3 alone: {:.1}%", efficiency_from_l3_alone);
    
    // Add RAM (all baseline in original logic)
    let total_auto_baseline = l3_baseline_auto + ram_size;
    let efficiency_l3_plus_ram = (total_auto_baseline as f64 / total_neurons as f64) * 100.0;
    println!("  L3 + RAM baseline efficiency: {:.1}%", efficiency_l3_plus_ram);
    
    // This should reveal the 94.1% source!
}

fn test_scenario(name: &str, l1_base: usize, l2_base: usize, l3_base: usize, ram_base: usize, total: usize) {
    let total_baseline = l1_base + l2_base + l3_base + ram_base;
    let efficiency = (total_baseline * 10000) / total; // Same calc as main code
    let efficiency_pct = efficiency as f64 / 100.0;
    
    println!("  {}: {} baseline = {:.1}% efficiency", name, total_baseline, efficiency_pct);
}