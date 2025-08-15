// LIVE PERFORMANCE ANALYSIS - HOBERMAN SPHERE METRICS
// Analyze the real-time performance data to optimize the foundation

use std::fs;
use std::collections::HashMap;

#[derive(Debug)]
struct PerformanceMetrics {
    timestamp: u64,
    operations: u64,
    errors: u64,
    l1_baseline: usize,
    l2_baseline: usize,
    l3_baseline: usize,
    ram_baseline: usize,
    efficiency: f64,
    latency_ns: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 HOBERMAN SPHERE LIVE PERFORMANCE ANALYSIS");
    println!("{}", "=".repeat(60));
    
    // Load metrics data
    let metrics = load_metrics("live_hoberman_metrics.csv")?;
    println!("📈 Loaded {} performance records", metrics.len());
    
    if metrics.is_empty() {
        println!("❌ No metrics data found!");
        return Ok(());
    }
    
    // Analysis 1: Performance trends
    analyze_performance_trends(&metrics);
    
    // Analysis 2: Energy efficiency patterns
    analyze_energy_efficiency(&metrics);
    
    // Analysis 3: Latency analysis
    analyze_latency_patterns(&metrics);
    
    // Analysis 4: Memory tier utilization
    analyze_memory_tiers(&metrics);
    
    // Analysis 5: System stability
    analyze_system_stability(&metrics);
    
    // Analysis 6: Optimization recommendations
    provide_optimization_recommendations(&metrics);
    
    Ok(())
}

fn load_metrics(filename: &str) -> Result<Vec<PerformanceMetrics>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(filename)?;
    let mut metrics = Vec::new();
    
    for (line_num, line) in content.lines().enumerate() {
        if line_num == 0 { continue; } // Skip header
        
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 9 {
            continue;
        }
        
        let metric = PerformanceMetrics {
            timestamp: parts[0].parse().unwrap_or(0),
            operations: parts[1].parse().unwrap_or(0),
            errors: parts[2].parse().unwrap_or(0),
            l1_baseline: parts[3].parse().unwrap_or(0),
            l2_baseline: parts[4].parse().unwrap_or(0),
            l3_baseline: parts[5].parse().unwrap_or(0),
            ram_baseline: parts[6].parse().unwrap_or(0),
            efficiency: parts[7].parse().unwrap_or(0) as f64 / 100.0,
            latency_ns: parts[8].parse().unwrap_or(0),
        };
        
        metrics.push(metric);
    }
    
    Ok(metrics)
}

fn analyze_performance_trends(metrics: &[PerformanceMetrics]) {
    println!("\n🚀 PERFORMANCE TRENDS ANALYSIS");
    println!("{}", "-".repeat(40));
    
    if metrics.len() < 2 {
        println!("❌ Need at least 2 data points for trend analysis");
        return;
    }
    
    let first = &metrics[0];
    let last = &metrics[metrics.len() - 1];
    
    let runtime_seconds = last.timestamp - first.timestamp;
    let total_operations = last.operations - first.operations;
    let ops_per_second = if runtime_seconds > 0 {
        total_operations as f64 / runtime_seconds as f64
    } else {
        0.0
    };
    
    println!("⏱️  Runtime: {} seconds", runtime_seconds);
    println!("🔄 Total operations: {}", total_operations);
    println!("📈 Average ops/sec: {:.1}", ops_per_second);
    println!("❌ Total errors: {}", last.errors);
    println!("📊 Error rate: {:.6}%", (last.errors as f64 / total_operations as f64) * 100.0);
    
    // Throughput stability
    let mut recent_ops = Vec::new();
    let recent_count = std::cmp::min(10, metrics.len());
    
    for i in (metrics.len() - recent_count)..metrics.len() {
        if i > 0 {
            let ops_diff = metrics[i].operations - metrics[i-1].operations;
            recent_ops.push(ops_diff);
        }
    }
    
    if !recent_ops.is_empty() {
        let avg_recent = recent_ops.iter().sum::<u64>() as f64 / recent_ops.len() as f64;
        let variance: f64 = recent_ops.iter()
            .map(|&x| (x as f64 - avg_recent).powi(2))
            .sum::<f64>() / recent_ops.len() as f64;
        let std_dev = variance.sqrt();
        
        println!("🎯 Recent throughput stability:");
        println!("   Average: {:.1} ops/interval", avg_recent);
        println!("   Std dev: {:.1} ({}% variance)", std_dev, (std_dev / avg_recent * 100.0) as i32);
        
        if std_dev / avg_recent < 0.05 {
            println!("   ✅ EXCELLENT stability (<5% variance)");
        } else if std_dev / avg_recent < 0.1 {
            println!("   ✅ Good stability (<10% variance)");
        } else {
            println!("   ⚠️  High variance detected");
        }
    }
}

fn analyze_energy_efficiency(metrics: &[PerformanceMetrics]) {
    println!("\n🔋 ENERGY EFFICIENCY ANALYSIS");
    println!("{}", "-".repeat(40));
    
    let total_neurons = 1_000_000; // Our network size
    
    // Find min, max, average efficiency
    let efficiencies: Vec<f64> = metrics.iter().map(|m| m.efficiency).collect();
    let min_eff = efficiencies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_eff = efficiencies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let avg_eff = efficiencies.iter().sum::<f64>() / efficiencies.len() as f64;
    
    println!("📊 Energy Efficiency Distribution:");
    println!("   Minimum: {:.2}%", min_eff);
    println!("   Maximum: {:.2}%", max_eff);
    println!("   Average: {:.2}%", avg_eff);
    println!("   Range: {:.2}%", max_eff - min_eff);
    
    // Baseline neuron analysis
    if let Some(sample) = metrics.last() {
        let total_baseline = sample.l1_baseline + sample.l2_baseline + 
                            sample.l3_baseline + sample.ram_baseline;
        let active_neurons = total_neurons - total_baseline;
        
        println!("\n🧠 Neural Activity Breakdown:");
        println!("   Total neurons: {}", total_neurons);
        println!("   Baseline (0 energy): {} ({:.1}%)", total_baseline, 
                (total_baseline as f64 / total_neurons as f64) * 100.0);
        println!("   Active neurons: {} ({:.1}%)", active_neurons,
                (active_neurons as f64 / total_neurons as f64) * 100.0);
        
        println!("\n🏗️  Memory Tier Distribution:");
        println!("   L1 baseline: {} / 10000 ({:.1}%)", sample.l1_baseline,
                (sample.l1_baseline as f64 / 10000.0) * 100.0);
        println!("   L2 baseline: {} / 50000 ({:.1}%)", sample.l2_baseline,
                (sample.l2_baseline as f64 / 50000.0) * 100.0);
        println!("   L3 baseline: {} / 500000 ({:.1}%)", sample.l3_baseline,
                (sample.l3_baseline as f64 / 500000.0) * 100.0);
        println!("   RAM baseline: {} / 440000 ({:.1}%)", sample.ram_baseline,
                (sample.ram_baseline as f64 / 440000.0) * 100.0);
    }
    
    // Energy savings calculation
    let binary_energy = total_neurons as f64; // All neurons active in binary
    let trinary_energy = total_neurons as f64 * (1.0 - avg_eff / 100.0);
    let savings = ((binary_energy - trinary_energy) / binary_energy) * 100.0;
    
    println!("\n💰 Energy Savings vs Binary:");
    println!("   Binary system energy: {:.0} units", binary_energy);
    println!("   Trinary system energy: {:.0} units", trinary_energy);
    println!("   TOTAL SAVINGS: {:.1}%", savings);
    println!("   Power reduction: {:.1}x less energy!", binary_energy / trinary_energy);
}

fn analyze_latency_patterns(metrics: &[PerformanceMetrics]) {
    println!("\n⚡ LATENCY ANALYSIS");
    println!("{}", "-".repeat(40));
    
    let latencies: Vec<u64> = metrics.iter().map(|m| m.latency_ns).collect();
    
    // Basic statistics
    let min_lat = *latencies.iter().min().unwrap_or(&0);
    let max_lat = *latencies.iter().max().unwrap_or(&0);
    let avg_lat = latencies.iter().sum::<u64>() as f64 / latencies.len() as f64;
    
    // Percentiles (approximate)
    let mut sorted_latencies = latencies.clone();
    sorted_latencies.sort();
    let p50 = sorted_latencies[sorted_latencies.len() / 2];
    let p95 = sorted_latencies[(sorted_latencies.len() * 95) / 100];
    let p99 = sorted_latencies[(sorted_latencies.len() * 99) / 100];
    
    println!("📊 Latency Distribution:");
    println!("   Minimum: {}ns ({:.2}µs)", min_lat, min_lat as f64 / 1000.0);
    println!("   Average: {:.0}ns ({:.2}µs)", avg_lat, avg_lat / 1000.0);
    println!("   Maximum: {}ns ({:.2}µs)", max_lat, max_lat as f64 / 1000.0);
    println!("   P50: {}ns ({:.2}µs)", p50, p50 as f64 / 1000.0);
    println!("   P95: {}ns ({:.2}µs)", p95, p95 as f64 / 1000.0);
    println!("   P99: {}ns ({:.2}µs)", p99, p99 as f64 / 1000.0);
    
    // Performance categorization
    if avg_lat < 100_000.0 { // < 100µs
        println!("   ✅ EXCELLENT: Sub-100µs latency!");
    } else if avg_lat < 1_000_000.0 { // < 1ms
        println!("   ✅ GOOD: Sub-millisecond latency");
    } else {
        println!("   ⚠️  HIGH: >1ms latency detected");
    }
    
    // Latency stability
    let variance: f64 = latencies.iter()
        .map(|&x| (x as f64 - avg_lat).powi(2))
        .sum::<f64>() / latencies.len() as f64;
    let std_dev = variance.sqrt();
    let cv = std_dev / avg_lat; // Coefficient of variation
    
    println!("\n🎯 Latency Stability:");
    println!("   Standard deviation: {:.0}ns", std_dev);
    println!("   Coefficient of variation: {:.3}", cv);
    
    if cv < 0.1 {
        println!("   ✅ EXCELLENT stability (CV < 0.1)");
    } else if cv < 0.2 {
        println!("   ✅ Good stability (CV < 0.2)");
    } else {
        println!("   ⚠️  High latency variance detected");
    }
}

fn analyze_memory_tiers(metrics: &[PerformanceMetrics]) {
    println!("\n🏗️  MEMORY TIER ANALYSIS");
    println!("{}", "-".repeat(40));
    
    if let Some(latest) = metrics.last() {
        let l1_active = 10000 - latest.l1_baseline;
        let l2_active = 50000 - latest.l2_baseline;
        let l3_active = 500000 - latest.l3_baseline;
        let ram_active = 440000 - latest.ram_baseline;
        
        println!("🔥 Current Activity per Tier:");
        println!("   L1 (ultra-fast): {} active / {} total ({:.1}% utilization)", 
                l1_active, 10000, (l1_active as f64 / 10000.0) * 100.0);
        println!("   L2 (fast):       {} active / {} total ({:.1}% utilization)", 
                l2_active, 50000, (l2_active as f64 / 50000.0) * 100.0);
        println!("   L3 (medium):     {} active / {} total ({:.1}% utilization)", 
                l3_active, 500000, (l3_active as f64 / 500000.0) * 100.0);
        println!("   RAM (storage):   {} active / {} total ({:.1}% utilization)", 
                ram_active, 440000, (ram_active as f64 / 440000.0) * 100.0);
        
        // Memory hierarchy efficiency
        let total_active = l1_active + l2_active + l3_active + ram_active;
        println!("\n🎯 Memory Hierarchy Efficiency:");
        println!("   L1 share of activity: {:.1}%", (l1_active as f64 / total_active as f64) * 100.0);
        println!("   L2 share of activity: {:.1}%", (l2_active as f64 / total_active as f64) * 100.0);
        println!("   L3 share of activity: {:.1}%", (l3_active as f64 / total_active as f64) * 100.0);
        println!("   RAM share of activity: {:.1}%", (ram_active as f64 / total_active as f64) * 100.0);
        
        // Ideal would be more activity in faster tiers
        let fast_tier_activity = l1_active + l2_active;
        let fast_percentage = (fast_tier_activity as f64 / total_active as f64) * 100.0;
        
        if fast_percentage > 20.0 {
            println!("   ✅ Good fast-tier utilization: {:.1}%", fast_percentage);
        } else {
            println!("   💡 Opportunity: Only {:.1}% in fast tiers", fast_percentage);
        }
    }
}

fn analyze_system_stability(metrics: &[PerformanceMetrics]) {
    println!("\n🛡️  SYSTEM STABILITY ANALYSIS");
    println!("{}", "-".repeat(40));
    
    let error_rate = if let Some(last) = metrics.last() {
        if last.operations > 0 {
            (last.errors as f64 / last.operations as f64) * 100.0
        } else {
            0.0
        }
    } else {
        0.0
    };
    
    println!("🚨 Error Analysis:");
    println!("   Total errors: {}", metrics.last().map(|m| m.errors).unwrap_or(0));
    println!("   Error rate: {:.6}%", error_rate);
    
    if error_rate == 0.0 {
        println!("   ✅ PERFECT: Zero errors detected!");
    } else if error_rate < 0.01 {
        println!("   ✅ EXCELLENT: <0.01% error rate");
    } else if error_rate < 0.1 {
        println!("   ✅ Good: <0.1% error rate");
    } else {
        println!("   ⚠️  High error rate detected");
    }
    
    // Check for efficiency stability
    let recent_efficiencies: Vec<f64> = metrics.iter()
        .rev()
        .take(100)
        .map(|m| m.efficiency)
        .collect();
    
    if recent_efficiencies.len() > 1 {
        let eff_variance: f64 = {
            let mean = recent_efficiencies.iter().sum::<f64>() / recent_efficiencies.len() as f64;
            recent_efficiencies.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / recent_efficiencies.len() as f64
        };
        let eff_std_dev = eff_variance.sqrt();
        
        println!("\n⚡ Energy Efficiency Stability:");
        println!("   Standard deviation: {:.3}%", eff_std_dev);
        
        if eff_std_dev < 0.1 {
            println!("   ✅ ROCK SOLID: <0.1% efficiency variation");
        } else if eff_std_dev < 1.0 {
            println!("   ✅ Stable: <1% efficiency variation");
        } else {
            println!("   ⚠️  Efficiency fluctuation detected");
        }
    }
}

fn provide_optimization_recommendations(metrics: &[PerformanceMetrics]) {
    println!("\n🎯 OPTIMIZATION RECOMMENDATIONS");
    println!("{}", "-".repeat(40));
    
    if let Some(latest) = metrics.last() {
        let latencies: Vec<u64> = metrics.iter().map(|m| m.latency_ns).collect();
        let avg_latency = latencies.iter().sum::<u64>() as f64 / latencies.len() as f64;
        
        println!("Based on live performance data analysis:");
        println!();
        
        // Latency optimizations
        if avg_latency > 200_000.0 { // > 200µs
            println!("🚀 LATENCY OPTIMIZATION:");
            println!("   • Consider SIMD vectorization for trinary operations");
            println!("   • Implement lock-free data structures");
            println!("   • Move hot neurons to L1 cache tier");
        } else {
            println!("✅ LATENCY: Excellent performance ({:.0}ns average)", avg_latency);
        }
        
        // Memory tier optimization
        let l1_utilization = (10000 - latest.l1_baseline) as f64 / 10000.0;
        if l1_utilization < 0.5 {
            println!("\n💾 MEMORY TIER OPTIMIZATION:");
            println!("   • L1 cache underutilized ({:.1}% active)", l1_utilization * 100.0);
            println!("   • Move frequently accessed neurons to L1");
            println!("   • Implement adaptive tier migration");
        } else {
            println!("\n✅ MEMORY TIERS: Good utilization pattern");
        }
        
        // Energy efficiency optimization
        if latest.efficiency < 90.0 {
            println!("\n🔋 ENERGY OPTIMIZATION:");
            println!("   • Current efficiency: {:.1}%", latest.efficiency);
            println!("   • Target: >95% baseline neurons");
            println!("   • Implement more aggressive baseline decay");
        } else {
            println!("\n✅ ENERGY: Exceptional efficiency ({:.1}%)", latest.efficiency);
        }
        
        // Scaling recommendations
        let ops_per_sec = if metrics.len() > 1 {
            let first = &metrics[0];
            let runtime = latest.timestamp - first.timestamp;
            if runtime > 0 {
                (latest.operations - first.operations) as f64 / runtime as f64
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        println!("\n🎯 SCALING RECOMMENDATIONS:");
        println!("   Current throughput: {:.0} ops/sec", ops_per_sec);
        
        if ops_per_sec > 5000.0 {
            println!("   ✅ Ready for larger neural networks!");
            println!("   • Consider scaling to 10M+ neurons");
            println!("   • Add GPU acceleration for massive parallelism");
        } else {
            println!("   💡 Optimize current scale before expanding");
        }
        
        println!("\n🏆 OVERALL ASSESSMENT:");
        
        let score = calculate_performance_score(latest, avg_latency, ops_per_sec);
        println!("   Performance score: {}/100", score);
        
        if score >= 90 {
            println!("   🥇 GRADE A: Production ready, exceptional performance!");
        } else if score >= 80 {
            println!("   🥈 GRADE B: Very good, minor optimizations possible");
        } else if score >= 70 {
            println!("   🥉 GRADE C: Good foundation, optimization needed");
        } else {
            println!("   📈 GRADE D: Significant optimization required");
        }
    }
}

fn calculate_performance_score(metrics: &PerformanceMetrics, avg_latency: f64, ops_per_sec: f64) -> u32 {
    let mut score = 0u32;
    
    // Energy efficiency (40 points max)
    if metrics.efficiency >= 95.0 {
        score += 40;
    } else if metrics.efficiency >= 90.0 {
        score += 35;
    } else if metrics.efficiency >= 80.0 {
        score += 25;
    } else {
        score += 10;
    }
    
    // Latency (30 points max)
    if avg_latency < 100_000.0 { // < 100µs
        score += 30;
    } else if avg_latency < 500_000.0 { // < 500µs
        score += 20;
    } else if avg_latency < 1_000_000.0 { // < 1ms
        score += 10;
    }
    
    // Throughput (20 points max)
    if ops_per_sec > 5000.0 {
        score += 20;
    } else if ops_per_sec > 1000.0 {
        score += 15;
    } else if ops_per_sec > 100.0 {
        score += 10;
    } else {
        score += 5;
    }
    
    // Stability (10 points max)
    if metrics.errors == 0 {
        score += 10;
    } else {
        score += 5;
    }
    
    score
}