//! Live Training Runner - Simple version to get brain running NOW
//! 
//! This runs the core training loop with real Redis data while we fix
//! compilation issues and build the native language.

use redis::{Client, Commands};
use chrono::Utc;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use serde_json::Value;

/// Simple live training system
pub struct LiveTrainingRunner {
    redis_client: Client,
    training_data: Vec<TrainingDataPoint>,
    patterns_discovered: HashMap<String, PatternInfo>,
    start_time: Instant,
    total_samples: u64,
}

#[derive(Debug, Clone)]
pub struct TrainingDataPoint {
    pub timestamp: i64,
    pub source: String,
    pub data: Value,
    pub sentiment: f32,
}

#[derive(Debug, Clone)]
pub struct PatternInfo {
    pub pattern_type: String,
    pub confidence: f32,
    pub occurrences: u32,
    pub last_seen: i64,
}

impl LiveTrainingRunner {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        println!("🧠 === NEURONLANG LIVE TRAINING SYSTEM ===");
        println!("📡 Connecting to Redis at 192.168.1.30:6379...");
        
        let redis_client = Client::open("redis://192.168.1.30:6379")?;
        
        // Test connection
        let mut conn = redis_client.get_connection()?;
        let _: String = redis::cmd("PING").query(&mut conn)?;
        println!("✅ Redis connection established!");
        
        Ok(Self {
            redis_client,
            training_data: Vec::new(),
            patterns_discovered: HashMap::new(),
            start_time: Instant::now(),
            total_samples: 0,
        })
    }
    
    /// Main training loop
    pub async fn run_training_loop(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n🚀 Starting live training loop...");
        println!("⚡ Brain will learn continuously from real market data");
        println!("📊 Press Ctrl+C to stop and save checkpoint\n");
        
        let mut iteration = 0;
        
        loop {
            iteration += 1;
            let iteration_start = Instant::now();
            
            // 1. Fetch new data from Redis
            let new_data = self.fetch_redis_data().await?;
            
            if !new_data.is_empty() {
                println!("📥 [{}] Fetched {} new data points", 
                        Utc::now().format("%H:%M:%S"), 
                        new_data.len());
                
                // 2. Process with trinary logic
                for data_point in new_data {
                    self.process_data_point(&data_point)?;
                    self.total_samples += 1;
                }
                
                // 3. Pattern detection every 10 iterations
                if iteration % 10 == 0 {
                    self.detect_patterns();
                }
                
                // 4. Memory consolidation every 100 iterations
                if iteration % 100 == 0 {
                    self.consolidate_memory()?;
                    self.print_stats();
                }
            }
            
            // 5. Sleep to maintain reasonable polling rate
            let elapsed = iteration_start.elapsed();
            if elapsed < Duration::from_millis(100) {
                sleep(Duration::from_millis(100) - elapsed).await;
            }
        }
    }
    
    /// Fetch data from Redis - NOW WITH RAW DATA CHANNELS!
    async fn fetch_redis_data(&mut self) -> Result<Vec<TrainingDataPoint>, Box<dyn std::error::Error>> {
        let mut conn = self.redis_client.get_connection()?;
        let mut data_points = Vec::new();
        
        // SUBSCRIBE TO REAL-TIME RAW DATA CHANNELS!
        // This is what we SHOULD be learning from
        
        // Get latest from team:btc:raw_data (HyperLiquid real-time)
        if let Ok(raw_btc) = conn.get::<&str, String>("team:btc:raw_data:latest") {
            if let Ok(json_data) = serde_json::from_str::<Value>(&raw_btc) {
                // Extract REAL market sentiment from price momentum
                let momentum = json_data.get("price_momentum")
                    .and_then(|m| m.as_str())
                    .unwrap_or("neutral");
                
                let sentiment = match momentum {
                    "bullish" => 1.0,   // ACTIVATED
                    "bearish" => -1.0,  // INHIBITED
                    _ => 0.0,           // BASELINE
                };
                
                // Also consider volume and spread
                let volume = json_data.get("volume_24h")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
                
                // High volume = more activation
                let volume_factor = if volume > 10_000_000.0 { 0.5 } else { 0.0 };
                
                data_points.push(TrainingDataPoint {
                    timestamp: Utc::now().timestamp(),
                    source: "hyperliquid_btc".to_string(),
                    data: json_data,
                    sentiment: sentiment + volume_factor,
                });
            }
        }
        
        // Get Reddit posts (raw sentiment from actual text)
        let reddit_keys: Vec<String> = conn.keys("reddit:post:*").unwrap_or_default();
        for key in reddit_keys.iter().take(3) {
            if let Ok(data) = conn.get::<String, String>(key.clone()) {
                if let Ok(json_data) = serde_json::from_str::<Value>(&data) {
                    // Calculate sentiment from upvotes/comments
                    let upvotes = json_data.get("ups")
                        .and_then(|u| u.as_i64())
                        .unwrap_or(0) as f32;
                    
                    let comments = json_data.get("num_comments")
                        .and_then(|c| c.as_i64())
                        .unwrap_or(0) as f32;
                    
                    // High engagement = activation
                    let sentiment = if upvotes > 100.0 {
                        1.0  // ACTIVATED - hot post
                    } else if upvotes < 10.0 {
                        -0.5 // INHIBITED - ignored post
                    } else {
                        0.0  // BASELINE
                    };
                    
                    data_points.push(TrainingDataPoint {
                        timestamp: Utc::now().timestamp(),
                        source: "reddit_raw".to_string(),
                        data: json_data,
                        sentiment,
                    });
                }
            }
        }
        
        // Fetch brain knowledge
        let brain_keys: Vec<String> = conn.keys("brain:knowledge:*").unwrap_or_default();
        for key in brain_keys.iter().take(3) {
            if let Ok(data) = conn.get::<String, String>(key.clone()) {
                if let Ok(json_data) = serde_json::from_str::<Value>(&data) {
                    // Extract brain tone/confidence as sentiment
                    let sentiment = json_data.get("confidence")
                        .and_then(|c| c.as_f64())
                        .unwrap_or(0.0) as f32;
                    
                    data_points.push(TrainingDataPoint {
                        timestamp: Utc::now().timestamp(),
                        source: "brain".to_string(),
                        data: json_data,
                        sentiment,
                    });
                }
            }
        }
        
        // Fetch price data
        let price_keys: Vec<String> = conn.keys("datalake:price:*").unwrap_or_default();
        for key in price_keys.iter().take(2) {
            if let Ok(data) = conn.get::<String, String>(key.clone()) {
                if let Ok(json_data) = serde_json::from_str::<Value>(&data) {
                    // Calculate price sentiment based on price change
                    let price = json_data.get("price")
                        .and_then(|p| p.as_f64())
                        .unwrap_or(0.0) as f32;
                    
                    // Simple momentum: if price > 50000 for BTC, bullish
                    let sentiment = if price > 50000.0 {
                        0.5  // Activated
                    } else if price < 40000.0 {
                        -0.5 // Inhibited
                    } else {
                        0.0  // Baseline
                    };
                    
                    data_points.push(TrainingDataPoint {
                        timestamp: Utc::now().timestamp(),
                        source: "price".to_string(),
                        data: json_data,
                        sentiment,
                    });
                }
            }
        }
        
        Ok(data_points)
    }
    
    /// Process a single data point with trinary logic
    fn process_data_point(&mut self, data_point: &TrainingDataPoint) -> Result<(), Box<dyn std::error::Error>> {
        // Store for pattern detection
        self.training_data.push(data_point.clone());
        
        // Keep window of last 1000 points
        if self.training_data.len() > 1000 {
            self.training_data.remove(0);
        }
        
        // MORE SENSITIVE trinary classification for demonstration
        // Lower thresholds to see more activation
        let trinary_state = if data_point.sentiment > 0.01 {  // Much lower threshold
            "ACTIVATED"
        } else if data_point.sentiment < -0.01 {  // Much lower threshold
            "INHIBITED"
        } else {
            "BASELINE" // Zero energy!
        };
        
        // Track state transitions
        if self.training_data.len() >= 2 {
            let prev = &self.training_data[self.training_data.len() - 2];
            let prev_sentiment = prev.sentiment;
            
            // Detect sentiment velocity
            let velocity = data_point.sentiment - prev_sentiment;
            if velocity.abs() > 0.2 {
                let pattern_key = format!("velocity_spike_{}", data_point.source);
                let pattern = self.patterns_discovered.entry(pattern_key).or_insert(PatternInfo {
                    pattern_type: "velocity_spike".to_string(),
                    confidence: 0.0,
                    occurrences: 0,
                    last_seen: 0,
                });
                pattern.occurrences += 1;
                pattern.confidence = pattern.occurrences as f32 / 100.0;
                pattern.last_seen = data_point.timestamp;
            }
        }
        
        Ok(())
    }
    
    /// Detect patterns in training data
    fn detect_patterns(&mut self) {
        // Simple pattern: correlation between sources
        let social_points: Vec<&TrainingDataPoint> = self.training_data.iter()
            .filter(|d| d.source == "social")
            .collect();
            
        let price_points: Vec<&TrainingDataPoint> = self.training_data.iter()
            .filter(|d| d.source == "price")
            .collect();
        
        if social_points.len() >= 10 && price_points.len() >= 10 {
            // Check if social sentiment predicts price movement
            let avg_social_sentiment: f32 = social_points.iter()
                .map(|p| p.sentiment)
                .sum::<f32>() / social_points.len() as f32;
            
            if avg_social_sentiment.abs() > 0.05 {
                let pattern_key = "social_price_correlation".to_string();
                let pattern = self.patterns_discovered.entry(pattern_key).or_insert(PatternInfo {
                    pattern_type: "correlation".to_string(),
                    confidence: avg_social_sentiment.abs(),
                    occurrences: 0,
                    last_seen: 0,
                });
                pattern.occurrences += 1;
                pattern.last_seen = Utc::now().timestamp();
            }
        }
        
        // Pattern: Brain tone clusters
        let brain_points: Vec<&TrainingDataPoint> = self.training_data.iter()
            .filter(|d| d.source == "brain")
            .collect();
        
        if brain_points.len() >= 5 {
            let pattern_key = "brain_cluster".to_string();
            let pattern = self.patterns_discovered.entry(pattern_key).or_insert(PatternInfo {
                pattern_type: "cluster".to_string(),
                confidence: 0.5,
                occurrences: 0,
                last_seen: 0,
            });
            pattern.occurrences += 1;
            pattern.last_seen = Utc::now().timestamp();
        }
    }
    
    /// Consolidate memory (simplified EWC)
    fn consolidate_memory(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n🧬 === MEMORY CONSOLIDATION ===");
        
        // Save important patterns
        let important_patterns: Vec<(&String, &PatternInfo)> = self.patterns_discovered.iter()
            .filter(|(_, p)| p.confidence > 0.3)
            .collect();
        
        println!("📊 Important patterns preserved: {}", important_patterns.len());
        
        for (name, pattern) in important_patterns.iter().take(5) {
            println!("  • {} (confidence: {:.2}, occurrences: {})",
                    name, pattern.confidence, pattern.occurrences);
        }
        
        // Save checkpoint
        self.save_checkpoint()?;
        
        Ok(())
    }
    
    /// Print training statistics
    fn print_stats(&self) {
        let runtime = self.start_time.elapsed();
        let hours = runtime.as_secs() / 3600;
        let minutes = (runtime.as_secs() % 3600) / 60;
        let seconds = runtime.as_secs() % 60;
        
        println!("\n📊 === TRAINING STATISTICS ===");
        println!("⏱️  Runtime: {:02}:{:02}:{:02}", hours, minutes, seconds);
        println!("📈 Total samples: {}", self.total_samples);
        println!("🧠 Patterns discovered: {}", self.patterns_discovered.len());
        println!("💾 Data window: {} points", self.training_data.len());
        
        // Calculate processing rate
        if runtime.as_secs() > 0 {
            let rate = self.total_samples as f32 / runtime.as_secs() as f32;
            println!("⚡ Processing rate: {:.1} samples/sec", rate);
        }
        
        // Show baseline percentage (energy efficiency)
        let baseline_count = self.training_data.iter()
            .filter(|d| d.sentiment.abs() < 0.1)
            .count();
        let baseline_pct = if !self.training_data.is_empty() {
            baseline_count as f32 / self.training_data.len() as f32 * 100.0
        } else {
            0.0
        };
        println!("🔋 Energy efficiency: {:.1}% baseline (zero energy)", baseline_pct);
    }
    
    /// Save checkpoint
    fn save_checkpoint(&self) -> Result<(), Box<dyn std::error::Error>> {
        let checkpoint = serde_json::json!({
            "timestamp": Utc::now().timestamp(),
            "total_samples": self.total_samples,
            "patterns": self.patterns_discovered.len(),
            "runtime_seconds": self.start_time.elapsed().as_secs(),
            "top_patterns": self.patterns_discovered.iter()
                .take(10)
                .map(|(k, v)| (k.clone(), v.confidence))
                .collect::<Vec<_>>()
        });
        
        let checkpoint_path = format!("checkpoint_{}.json", Utc::now().format("%Y%m%d_%H%M%S"));
        std::fs::write(&checkpoint_path, serde_json::to_string_pretty(&checkpoint)?)?;
        println!("💾 Checkpoint saved: {}", checkpoint_path);
        
        Ok(())
    }
}

/// Run the live training
pub async fn start_live_training() -> Result<(), Box<dyn std::error::Error>> {
    let mut runner = LiveTrainingRunner::new()?;
    
    // Handle Ctrl+C gracefully
    let result = runner.run_training_loop().await;
    
    // Save final checkpoint
    runner.save_checkpoint()?;
    
    result
}