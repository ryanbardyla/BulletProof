// SIMPLE RIPPLE TEST - Let's see what the hardware tells us!

use std::time::Instant;

fn main() {
    println!("\n🌊 RIPPLE DISCOVERY - DROPPING THE PEBBLE!");
    println!("{}", "=".repeat(60));
    
    // Test 1: Find cache boundaries by timing memory access
    println!("\n💧 Finding cache boundaries...");
    find_cache_boundaries();
    
    // Test 2: Detect CPU capabilities
    println!("\n🔍 Detecting CPU features...");
    detect_cpu_features();
    
    // Test 3: Measure actual performance
    println!("\n⚡ Measuring real performance...");
    measure_performance();
}

fn find_cache_boundaries() {
    println!("  Size        Time/Access   Bandwidth     Boundary?");
    println!("  ----        -----------   ---------     ---------");
    
    let sizes = [
        32, 64, 128, 256, 512, 1024, 2048, 4096, 
        8192, 16384, 32768, 65536, 81920, 131072, 262144
    ];
    
    let mut last_time = 0.0;
    
    for &size_kb in &sizes {
        let size = size_kb * 1024;
        let mut data = vec![0u8; size];
        
        // Warm up
        for i in 0..size {
            data[i] = (i % 256) as u8;
        }
        
        // Time random access
        let iterations = 10_000_000 / (size_kb.max(1));
        let mut sum = 0u64;
        let mut index = 0;
        
        let start = Instant::now();
        for _ in 0..iterations {
            index = (index * 1103515245 + 12345) % size;
            sum += data[index] as u64;
        }
        let elapsed = start.elapsed();
        
        let ns_per_access = elapsed.as_nanos() as f64 / iterations as f64;
        let mb_per_sec = (iterations * 8) as f64 / elapsed.as_secs_f64() / 1_000_000.0;
        
        let boundary = if last_time > 0.0 && ns_per_access > last_time * 2.0 {
            " <-- BOUNDARY!"
        } else {
            ""
        };
        
        println!("  {:6} KB   {:8.2} ns   {:8.1} MB/s  {}", 
                 size_kb, ns_per_access, mb_per_sec, boundary);
        
        last_time = ns_per_access;
        
        // Prevent optimization
        if sum == 0 { println!(""); }
    }
}

fn detect_cpu_features() {
    // Detect CPU vendor
    let is_amd = std::fs::read_to_string("/proc/cpuinfo")
        .map(|s| s.contains("AuthenticAMD"))
        .unwrap_or(false);
    
    let vendor = if is_amd { "AMD" } else { "Intel" };
    println!("  Vendor: {}", vendor);
    
    // Count cores
    let physical_cores = num_cpus::get_physical();
    let logical_cores = num_cpus::get();
    println!("  Cores: {} physical, {} logical", physical_cores, logical_cores);
    
    // Detect SIMD features
    #[cfg(target_arch = "x86_64")]
    {
        println!("  SIMD Features:");
        if is_x86_feature_detected!("avx512f") {
            println!("    ✅ AVX-512 (512-bit vectors!)");
        }
        if is_x86_feature_detected!("avx2") {
            println!("    ✅ AVX2 (256-bit vectors)");
        }
        if is_x86_feature_detected!("avx") {
            println!("    ✅ AVX");
        }
        if is_x86_feature_detected!("sse4.2") {
            println!("    ✅ SSE4.2");
        }
    }
    
    // Detect memory
    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        if let Some(line) = meminfo.lines().find(|l| l.starts_with("MemTotal:")) {
            if let Some(kb_str) = line.split_whitespace().nth(1) {
                if let Ok(kb) = kb_str.parse::<u64>() {
                    println!("  RAM: {:.1} GB", kb as f64 / 1_048_576.0);
                }
            }
        }
    }
    
    // Check for GPU
    if std::path::Path::new("/dev/nvidia0").exists() {
        println!("  GPU: NVIDIA detected!");
    } else if std::path::Path::new("/dev/dri/card0").exists() {
        println!("  GPU: Graphics card detected");
    } else {
        println!("  GPU: None detected");
    }
}

fn measure_performance() {
    // Measure single-core performance
    let start = Instant::now();
    let mut x = 0u64;
    for i in 0..1_000_000_000 {
        x = x.wrapping_add(i);
    }
    let elapsed = start.elapsed();
    let gops = 1.0 / elapsed.as_secs_f64();
    println!("  Single-core: {:.2} GOPS", gops);
    
    // Measure trinary operations
    let start = Instant::now();
    let mut result = 0i8;
    for _ in 0..100_000_000 {
        let a = 1i8;
        let b = -1i8;
        result = if a == 0 || b == 0 { 0 } else { a * b };
    }
    let elapsed = start.elapsed();
    let trinary_ops = 100_000_000.0 / elapsed.as_secs_f64() / 1_000_000_000.0;
    println!("  Trinary ops: {:.2} GTOPS", trinary_ops);
    
    // Prevent optimization
    if result == x as i8 { println!(""); }
}

// Simple num_cpus implementation
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
    
    pub fn get_physical() -> usize {
        // Read from /proc/cpuinfo
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            cpuinfo.lines()
                .filter(|l| l.starts_with("physical id"))
                .collect::<std::collections::HashSet<_>>()
                .len()
                .max(1)
        } else {
            get() / 2  // Assume hyperthreading
        }
    }
}

use std::collections::HashSet;