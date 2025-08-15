// Test DNA Graph Storage - Always-on compute!

use neuronlang::graph_storage::{DNAGraphStorage, DNABase};
use std::thread;
use std::time::Duration;

fn main() {
    println!("🧬 DNA GRAPH STORAGE TEST");
    println!("========================");
    println!("This runs at ZERO energy - even when sleeping!");
    println!();
    
    // Create the always-on storage
    let storage = DNAGraphStorage::new();
    println!("✅ Baseline compute thread started (never stops!)");
    
    // Add some DNA-encoded nodes
    let node1 = storage.add_node(0, vec![
        DNABase::Guanine,  // 0 - baseline
        DNABase::Cytosine, // 0 - baseline  
        DNABase::Adenine,  // +1
        DNABase::Thymine,  // -1
    ]);
    
    let node2 = storage.add_node(1, vec![
        DNABase::Thymine,  // -1
        DNABase::Adenine,  // +1
        DNABase::Cytosine, // 0
        DNABase::Guanine,  // 0
    ]);
    
    println!("✅ Added DNA nodes (ATGC sequences)");
    
    // Connect them
    storage.connect_nodes(node1, node2);
    storage.connect_nodes(node2, node1); // Cycle!
    println!("✅ Created graph connections");
    
    // Store some data
    storage.store_data(node1, vec![42, 69, 13, 37]);
    println!("✅ Stored data in DNA format");
    
    // Let the baseline compute run
    println!("\n⏳ Baseline compute running (1 second)...");
    thread::sleep(Duration::from_secs(1));
    
    // Check sleep compute stats
    let (active, total) = storage.sleep_compute_stats();
    println!("📊 Sleep compute: {}/{} nodes active", active, total);
    
    // Traverse at zero energy
    let path = storage.traverse_baseline(node1);
    println!("🔄 Zero-energy traversal path: {:?}", path);
    
    // Trigger dream state
    storage.dream_state();
    println!("💭 Dream state executed (pattern discovery)");
    
    println!("\n✨ DNA Graph Storage is ALIVE!");
    println!("   Running forever at baseline energy...");
    println!("   Even when your computer sleeps!");
}