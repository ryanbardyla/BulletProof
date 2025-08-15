//! Real-time Pub/Sub Brain - Learns from LIVE data streams!
//!
//! This is the CORRECT way - subscribing to channels for instant data

use redis::{Client, Commands, PubSubCommands, ControlFlow};
use serde_json::Value;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

pub struct PubSubBrain {
    redis_client: Client,
    trinary_states: Arc<Mutex<HashMap<String, TrinaryState>>>,
    patterns: Arc<Mutex<Vec<Pattern>>>,
}

#[derive(Debug, Clone)]
pub enum TrinaryState {
    Activated(f32),   // +1 with confidence
    Baseline,         // 0 - ZERO ENERGY!
    Inhibited(f32),   // -1 with confidence
}

#[derive(Debug, Clone)]
pub struct Pattern {
    pub name: String,
    pub trigger: String,
    pub confidence: f32,
    pub occurrences: u32,
}

impl PubSubBrain {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        println!("🧠 === PUBSUB BRAIN INITIALIZING ===");
        println!("📡 This brain subscribes to LIVE channels!");
        
        let redis_client = Client::open("redis://192.168.1.30:6379")?;
        
        Ok(Self {
            redis_client,
            trinary_states: Arc::new(Mutex::new(HashMap::new())),
            patterns: Arc::new(Mutex::new(Vec::new())),
        })
    }
    
    pub fn start_learning(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n🚀 Subscribing to live data channels...");
        
        let mut pubsub = self.redis_client.get_connection()?.into_pubsub();
        
        // Subscribe to RAW DATA channels
        pubsub.subscribe("team:btc:raw_data")?;
        pubsub.subscribe("team:eth:raw_data")?;
        pubsub.subscribe("team:sol:raw_data")?;
        pubsub.subscribe("reddit:new_post")?;
        pubsub.subscribe("fenrisa:sentiment:*")?;
        
        println!("✅ Subscribed to 5 channels!");
        println!("📊 Processing live data streams...\n");
        
        let states = self.trinary_states.clone();
        let patterns = self.patterns.clone();
        
        // Process messages in real-time
        pubsub.on_message(|msg| {
            let channel = msg.get_channel_name();
            let payload: String = msg.get_payload().unwrap_or_default();
            
            // Parse JSON data
            if let Ok(data) = serde_json::from_str::<Value>(&payload) {
                // Determine trinary state based on channel and data
                let state = Self::compute_trinary_state(channel, &data);
                
                // Update states
                states.lock().unwrap().insert(channel.to_string(), state.clone());
                
                // Display state change
                match state {
                    TrinaryState::Activated(conf) => {
                        println!("⚡ [{}] ACTIVATED (confidence: {:.2})", channel, conf);
                        
                        // Check for patterns
                        if channel.contains("btc") && conf > 0.8 {
                            let mut p = patterns.lock().unwrap();
                            p.push(Pattern {
                                name: "btc_surge".to_string(),
                                trigger: channel.to_string(),
                                confidence: conf,
                                occurrences: 1,
                            });
                            println!("  🎯 Pattern detected: BTC SURGE!");
                        }
                    },
                    TrinaryState::Inhibited(conf) => {
                        println!("🔻 [{}] INHIBITED (confidence: {:.2})", channel, conf);
                    },
                    TrinaryState::Baseline => {
                        // Silent - ZERO energy consumption
                    }
                }
                
                // Show energy efficiency periodically
                let states_snapshot = states.lock().unwrap();
                let total = states_snapshot.len();
                if total > 0 && total % 10 == 0 {
                    let baseline_count = states_snapshot.values()
                        .filter(|s| matches!(s, TrinaryState::Baseline))
                        .count();
                    let efficiency = (baseline_count as f32 / total as f32) * 100.0;
                    println!("\n🔋 Energy Efficiency: {:.1}% baseline (ZERO energy!)\n", efficiency);
                }
            }
            
            ControlFlow::Continue
        });
        
        Ok(())
    }
    
    fn compute_trinary_state(channel: &str, data: &Value) -> TrinaryState {
        // Extract key indicators
        if channel.contains("raw_data") {
            // Market data from HyperLiquid
            if let Some(momentum) = data.get("price_momentum").and_then(|m| m.as_str()) {
                match momentum {
                    "bullish" => {
                        let volume = data.get("volume_24h")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0) as f32;
                        
                        let confidence = if volume > 10_000_000.0 { 0.9 } else { 0.5 };
                        return TrinaryState::Activated(confidence);
                    },
                    "bearish" => {
                        let spread = data.get("spread_bps")
                            .and_then(|s| s.as_f64())
                            .unwrap_or(0.0) as f32;
                        
                        let confidence = if spread > 10.0 { 0.8 } else { 0.4 };
                        return TrinaryState::Inhibited(confidence);
                    },
                    _ => {}
                }
            }
        } else if channel == "reddit:new_post" {
            // Reddit sentiment
            let upvotes = data.get("ups").and_then(|u| u.as_i64()).unwrap_or(0);
            let downvotes = data.get("downs").and_then(|d| d.as_i64()).unwrap_or(0);
            
            if upvotes > 100 {
                return TrinaryState::Activated((upvotes as f32 / 1000.0).min(1.0));
            } else if downvotes > 50 {
                return TrinaryState::Inhibited((downvotes as f32 / 100.0).min(1.0));
            }
        }
        
        // Default to baseline - uses ZERO energy!
        TrinaryState::Baseline
    }
    
    pub fn get_energy_stats(&self) -> (usize, usize, usize) {
        let states = self.trinary_states.lock().unwrap();
        let activated = states.values().filter(|s| matches!(s, TrinaryState::Activated(_))).count();
        let inhibited = states.values().filter(|s| matches!(s, TrinaryState::Inhibited(_))).count();
        let baseline = states.values().filter(|s| matches!(s, TrinaryState::Baseline)).count();
        
        (activated, baseline, inhibited)
    }
}

pub fn start_pubsub_brain() -> Result<(), Box<dyn std::error::Error>> {
    let mut brain = PubSubBrain::new()?;
    brain.start_learning()?;
    Ok(())
}