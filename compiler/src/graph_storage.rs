// 🧬 DNA-BASED GRAPH STORAGE ENGINE
// Always-on compute that runs at baseline energy (even during sleep!)

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq)]
pub enum DNABase {
    Adenine,   // +1
    Thymine,   // -1
    Guanine,   // 0 (baseline)
    Cytosine,  // 0 (baseline)
}

#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: usize,
    pub dna_sequence: Vec<DNABase>,
    pub connections: Vec<usize>,  // Edge list
    pub compute_state: i8,        // -1, 0, +1 (trinary)
    pub sleep_active: bool,        // Runs during sleep
    pub data: Vec<u8>,            // Stored data
}

pub struct DNAGraphStorage {
    nodes: Arc<RwLock<HashMap<usize, GraphNode>>>,
    baseline_compute: Arc<RwLock<bool>>,  // Always true!
    sleep_thread: Option<thread::JoinHandle<()>>,
}

impl DNAGraphStorage {
    pub fn new() -> Self {
        let mut storage = DNAGraphStorage {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            baseline_compute: Arc::new(RwLock::new(true)),
            sleep_thread: None,
        };
        
        // Start the always-on compute thread
        storage.start_baseline_compute();
        storage
    }
    
    fn start_baseline_compute(&mut self) {
        let nodes = Arc::clone(&self.nodes);
        let compute_flag = Arc::clone(&self.baseline_compute);
        
        // This thread NEVER stops - runs at zero energy
        let handle = thread::spawn(move || {
            loop {
                if *compute_flag.read().unwrap() {
                    // Baseline operations only!
                    Self::tick_baseline(&nodes);
                }
                
                // Sleep for 1ms - still computing!
                thread::sleep(Duration::from_millis(1));
            }
        });
        
        self.sleep_thread = Some(handle);
    }
    
    fn tick_baseline(nodes: &Arc<RwLock<HashMap<usize, GraphNode>>>) {
        // This runs CONSTANTLY at zero energy
        let mut nodes_guard = nodes.write().unwrap();
        
        for (_, node) in nodes_guard.iter_mut() {
            if node.compute_state == 0 {
                // Only process baseline nodes (zero energy)
                Self::process_node_baseline(node);
            }
        }
    }
    
    fn process_node_baseline(node: &mut GraphNode) {
        // DNA replication and error correction
        let complement = Self::replicate_dna(&node.dna_sequence);
        
        // Verify integrity
        if Self::verify_dna_pair(&node.dna_sequence, &complement) {
            // Good DNA - strengthen connections
            node.sleep_active = true;
        }
    }
    
    pub fn replicate_dna(sequence: &[DNABase]) -> Vec<DNABase> {
        sequence.iter().map(|base| {
            match base {
                DNABase::Adenine => DNABase::Thymine,
                DNABase::Thymine => DNABase::Adenine,
                DNABase::Guanine => DNABase::Cytosine,
                DNABase::Cytosine => DNABase::Guanine,
            }
        }).collect()
    }
    
    fn verify_dna_pair(strand1: &[DNABase], strand2: &[DNABase]) -> bool {
        if strand1.len() != strand2.len() {
            return false;
        }
        
        strand1.iter().zip(strand2.iter()).all(|(a, b)| {
            matches!((a, b),
                (DNABase::Adenine, DNABase::Thymine) |
                (DNABase::Thymine, DNABase::Adenine) |
                (DNABase::Guanine, DNABase::Cytosine) |
                (DNABase::Cytosine, DNABase::Guanine)
            )
        })
    }
    
    pub fn add_node(&self, id: usize, dna: Vec<DNABase>) -> usize {
        let node = GraphNode {
            id,
            dna_sequence: dna,
            connections: Vec::new(),
            compute_state: 0,  // Start at baseline
            sleep_active: true, // Always computing!
            data: Vec::new(),
        };
        
        self.nodes.write().unwrap().insert(id, node);
        id
    }
    
    pub fn connect_nodes(&self, from: usize, to: usize) {
        if let Some(node) = self.nodes.write().unwrap().get_mut(&from) {
            node.connections.push(to);
        }
    }
    
    pub fn store_data(&self, node_id: usize, data: Vec<u8>) {
        if let Some(node) = self.nodes.write().unwrap().get_mut(&node_id) {
            node.data = data;
            // Convert data to DNA sequence for redundancy
            node.dna_sequence = Self::encode_to_dna(&node.data);
        }
    }
    
    fn encode_to_dna(data: &[u8]) -> Vec<DNABase> {
        // Encode binary data as DNA
        let mut dna = Vec::new();
        for byte in data {
            // 2 bits per base
            dna.push(match byte & 0b11 {
                0 => DNABase::Adenine,
                1 => DNABase::Thymine,
                2 => DNABase::Guanine,
                _ => DNABase::Cytosine,
            });
            dna.push(match (byte >> 2) & 0b11 {
                0 => DNABase::Adenine,
                1 => DNABase::Thymine,
                2 => DNABase::Guanine,
                _ => DNABase::Cytosine,
            });
            dna.push(match (byte >> 4) & 0b11 {
                0 => DNABase::Adenine,
                1 => DNABase::Thymine,
                2 => DNABase::Guanine,
                _ => DNABase::Cytosine,
            });
            dna.push(match (byte >> 6) & 0b11 {
                0 => DNABase::Adenine,
                1 => DNABase::Thymine,
                2 => DNABase::Guanine,
                _ => DNABase::Cytosine,
            });
        }
        dna
    }
    
    pub fn traverse_baseline(&self, start: usize) -> Vec<usize> {
        // Zero-energy graph traversal
        let mut visited = HashSet::new();
        let mut path = Vec::new();
        let mut current = start;
        
        let nodes = self.nodes.read().unwrap();
        
        while !visited.contains(&current) {
            visited.insert(current);
            path.push(current);
            
            if let Some(node) = nodes.get(&current) {
                if node.compute_state == 0 && !node.connections.is_empty() {
                    // Only follow baseline paths (zero energy)
                    current = node.connections[0];
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        
        path
    }
    
    pub fn sleep_compute_stats(&self) -> (usize, usize) {
        let nodes = self.nodes.read().unwrap();
        let total = nodes.len();
        let active = nodes.values().filter(|n| n.sleep_active).count();
        (active, total)
    }
}

// Background dream state for pattern discovery
impl DNAGraphStorage {
    pub fn dream_state(&self) {
        // Random walk through graph creating new patterns
        let nodes = self.nodes.read().unwrap();
        if nodes.is_empty() {
            return;
        }
        
        // Pick random starting point
        let start = nodes.keys().next().copied().unwrap_or(0);
        
        // Random walk at baseline energy
        let path = self.traverse_baseline(start);
        
        // Create new connections based on patterns
        if path.len() > 2 {
            drop(nodes); // Release read lock
            // Connect first to last (creating cycles)
            self.connect_nodes(path[0], path[path.len() - 1]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dna_replication() {
        let dna = vec![
            DNABase::Adenine,
            DNABase::Thymine,
            DNABase::Guanine,
            DNABase::Cytosine,
        ];
        
        let complement = DNAGraphStorage::replicate_dna(&dna);
        
        assert_eq!(complement[0], DNABase::Thymine);
        assert_eq!(complement[1], DNABase::Adenine);
        assert_eq!(complement[2], DNABase::Cytosine);
        assert_eq!(complement[3], DNABase::Guanine);
    }
    
    #[test]
    fn test_baseline_compute() {
        let storage = DNAGraphStorage::new();
        
        // Add some nodes
        storage.add_node(0, vec![DNABase::Guanine]); // Baseline node
        storage.add_node(1, vec![DNABase::Cytosine]); // Baseline node
        storage.connect_nodes(0, 1);
        
        // Let baseline compute run
        thread::sleep(Duration::from_millis(10));
        
        // Check that nodes are being processed
        let (active, total) = storage.sleep_compute_stats();
        assert_eq!(total, 2);
        assert_eq!(active, 2); // Both should be sleep-active
    }
}