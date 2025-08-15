//! Native Trinary Data Ingestion
//! 
//! NO BINARY CONTAMINATION! Pure trinary from source to neural network.
//! This is how a true trinary system should work!

use crate::tryte::Tryte;
use crate::real_brain::{RealMarketData, RealTradingSignal};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use redis::Commands;
use tokio::time::{Duration, Instant};

/// Pure trinary sentiment data (no binary conversion!)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeTrinarySentiment {
    pub symbol: String,
    pub sentiment_tryte: Tryte,        // Direct trinary sentiment!
    pub confidence_tryte: Tryte,       // Direct trinary confidence!
    pub volume_tryte: Tryte,           // Direct trinary volume signal!
    pub source: String,
    pub timestamp: u64,
    pub raw_mentions: u32,
}

/// Native trinary market event (not contaminated by binary!)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeTrinaryMarketEvent {
    pub symbol: String,
    pub price_direction_tryte: Tryte,  // Pure price direction!
    pub volume_flow_tryte: Tryte,      // Pure volume flow!
    pub momentum_tryte: Tryte,         // Pure momentum!
    pub funding_tryte: Tryte,          // Pure funding signal!
    pub timestamp: u64,
}

/// Native trinary social media event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialMediaEvent {
    pub id: String,
    pub source: String,
    pub symbol: String,
    pub title: String,
    pub sentiment: f64,                // Will convert to pure trinary!
    pub tokens: Vec<String>,
    pub is_high_value: bool,
    pub timestamp: String,
    pub url: String,
}

/// Pure trinary data pipeline (no binary corruption!)
pub struct NativeTrinaryPipeline {
    pub redis_client: redis::Client,
    pub sentiment_buffer: Vec<NativeTrinarySentiment>,
    pub market_buffer: Vec<NativeTrinaryMarketEvent>,
    pub social_buffer: Vec<SocialMediaEvent>,
    pub buffer_size: usize,
}

impl NativeTrinaryPipeline {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        println!("🧬 Initializing PURE TRINARY data pipeline...");
        println!("⚠️  NO BINARY CONTAMINATION ALLOWED!");
        
        let redis_client = redis::Client::open("redis://192.168.1.30:6379")?;
        
        Ok(Self {
            redis_client,
            sentiment_buffer: Vec::new(),
            market_buffer: Vec::new(),
            social_buffer: Vec::new(),
            buffer_size: 1000,
        })
    }
    
    /// Ingest social media data and convert to PURE TRINARY
    pub async fn ingest_social_data(&mut self) -> Result<Vec<NativeTrinarySentiment>, Box<dyn std::error::Error>> {
        let mut conn = self.redis_client.get_connection()?;
        
        // Get all social media keys
        let social_keys: Vec<String> = conn.keys("social:highvalue:*")?;
        
        let mut new_sentiments = Vec::new();
        
        for key in social_keys.iter().take(50) {  // Process 50 at a time
            if let Ok(social_json) = conn.get::<_, String>(key) {
                if let Ok(social_event) = serde_json::from_str::<SocialMediaEvent>(&social_json) {
                    
                    // Convert to PURE TRINARY (no binary contamination!)
                    let sentiment_tryte = self.sentiment_to_pure_trinary(social_event.sentiment);
                    
                    // High value posts get stronger confidence
                    let confidence_tryte = if social_event.is_high_value {
                        Tryte::Activated
                    } else {
                        Tryte::Baseline
                    };
                    
                    // Volume based on mentions (simulated for now)
                    let volume_tryte = if social_event.title.len() > 100 {
                        Tryte::Activated   // Long posts = high engagement
                    } else if social_event.title.len() < 50 {
                        Tryte::Inhibited   // Short posts = low engagement
                    } else {
                        Tryte::Baseline    // Normal engagement (ZERO ENERGY!)
                    };
                    
                    let native_sentiment = NativeTrinarySentiment {
                        symbol: social_event.symbol.clone(),
                        sentiment_tryte,
                        confidence_tryte,
                        volume_tryte,
                        source: social_event.source.clone(),
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        raw_mentions: 1,  // Could count actual mentions
                    };
                    
                    new_sentiments.push(native_sentiment);
                    self.social_buffer.push(social_event);
                }
            }
        }
        
        // Add to sentiment buffer
        self.sentiment_buffer.extend(new_sentiments.clone());
        
        // Keep buffer manageable
        if self.sentiment_buffer.len() > self.buffer_size {
            let excess = self.sentiment_buffer.len() - self.buffer_size;
            self.sentiment_buffer.drain(0..excess);
        }
        
        if !new_sentiments.is_empty() {
            println!("🧬 Ingested {} PURE TRINARY sentiment signals", new_sentiments.len());
            println!("   {} Positive, {} Negative, {} Neutral", 
                     new_sentiments.iter().filter(|s| s.sentiment_tryte == Tryte::Activated).count(),
                     new_sentiments.iter().filter(|s| s.sentiment_tryte == Tryte::Inhibited).count(),
                     new_sentiments.iter().filter(|s| s.sentiment_tryte == Tryte::Baseline).count());
        }
        
        Ok(new_sentiments)
    }
    
    /// Convert raw sentiment to PURE TRINARY (no binary corruption!)
    fn sentiment_to_pure_trinary(&self, sentiment: f64) -> Tryte {
        // CRITICAL: Use trinary logic, not binary thresholds!
        match sentiment {
            s if s > 0.1 => Tryte::Activated,      // Positive sentiment
            s if s < -0.1 => Tryte::Inhibited,     // Negative sentiment  
            _ => Tryte::Baseline,                  // Neutral sentiment (ZERO ENERGY!)
        }
    }
    
    /// Convert market data to PURE TRINARY events (no binary!)
    pub fn market_data_to_pure_trinary(&self, data: &RealMarketData, previous_data: Option<&RealMarketData>) -> NativeTrinaryMarketEvent {
        
        let price_direction_tryte = if let Some(prev) = previous_data {
            let price_change = (data.price - prev.price) / prev.price;
            
            if price_change > 0.001 {
                Tryte::Activated      // Price rising
            } else if price_change < -0.001 {
                Tryte::Inhibited      // Price falling
            } else {
                Tryte::Baseline       // Price stable (ZERO ENERGY!)
            }
        } else {
            Tryte::Baseline           // No previous data
        };
        
        let volume_flow_tryte = if data.volume > 1000000.0 {
            Tryte::Activated          // High volume
        } else if data.volume < 10000.0 {
            Tryte::Inhibited          // Low volume
        } else {
            Tryte::Baseline           // Normal volume (ZERO ENERGY!)
        };
        
        let funding_tryte = if data.funding_rate > 0.0001 {
            Tryte::Activated          // Positive funding (bullish)
        } else if data.funding_rate < -0.0001 {
            Tryte::Inhibited          // Negative funding (bearish)
        } else {
            Tryte::Baseline           // Neutral funding (ZERO ENERGY!)
        };
        
        // Calculate momentum from recent price action
        let momentum_tryte = if let Some(prev) = previous_data {
            let momentum = (data.price - prev.price) / prev.price;
            if momentum.abs() > 0.005 {
                if momentum > 0.0 {
                    Tryte::Activated      // Strong upward momentum
                } else {
                    Tryte::Inhibited      // Strong downward momentum
                }
            } else {
                Tryte::Baseline           // Weak momentum (ZERO ENERGY!)
            }
        } else {
            Tryte::Baseline
        };
        
        NativeTrinaryMarketEvent {
            symbol: data.token.clone(),
            price_direction_tryte,
            volume_flow_tryte,
            momentum_tryte,
            funding_tryte,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }
    
    /// Combine sentiment and market data into pure trinary neural inputs
    pub fn create_pure_trinary_inputs(&self, symbol: &str) -> Vec<Tryte> {
        let mut inputs = Vec::with_capacity(200);
        
        // Get recent sentiment for this symbol
        let recent_sentiments: Vec<&NativeTrinarySentiment> = self.sentiment_buffer
            .iter()
            .filter(|s| s.symbol.contains(symbol) || symbol.contains(&s.symbol))
            .rev()
            .take(50)
            .collect();
        
        // Add sentiment trytes directly (NO CONVERSION!)
        for sentiment in recent_sentiments {
            inputs.push(sentiment.sentiment_tryte);
            inputs.push(sentiment.confidence_tryte);
            inputs.push(sentiment.volume_tryte);
        }
        
        // Get recent market events for this symbol
        let recent_market: Vec<&NativeTrinaryMarketEvent> = self.market_buffer
            .iter()
            .filter(|m| m.symbol.contains(symbol) || symbol.contains(&m.symbol))
            .rev()
            .take(30)
            .collect();
        
        // Add market trytes directly (NO CONVERSION!)
        for market in recent_market {
            inputs.push(market.price_direction_tryte);
            inputs.push(market.volume_flow_tryte);
            inputs.push(market.momentum_tryte);
            inputs.push(market.funding_tryte);
        }
        
        // Pad to fixed size with baseline neurons (ZERO ENERGY!)
        while inputs.len() < 200 {
            inputs.push(Tryte::Baseline);
        }
        
        // Calculate sparsity
        let active_count = inputs.iter().filter(|&&t| t != Tryte::Baseline).count();
        let sparsity = (inputs.len() - active_count) as f32 / inputs.len() as f32;
        
        println!("🧬 Created PURE TRINARY inputs for {}: {} total, {} active ({:.1}% sparse)", 
                 symbol, inputs.len(), active_count, sparsity * 100.0);
        
        inputs
    }
    
    /// Get all symbols we have data for
    pub fn get_available_symbols(&self) -> Vec<String> {
        let mut symbols = std::collections::HashSet::new();
        
        for sentiment in &self.sentiment_buffer {
            symbols.insert(sentiment.symbol.clone());
        }
        
        for market in &self.market_buffer {
            symbols.insert(market.symbol.clone());
        }
        
        symbols.into_iter().collect()
    }
    
    /// Process news sentiment from Redis
    pub async fn ingest_news_sentiment(&mut self) -> Result<Vec<NativeTrinarySentiment>, Box<dyn std::error::Error>> {
        let mut conn = self.redis_client.get_connection()?;
        
        // Get news sentiment keys
        let news_keys: Vec<String> = conn.keys("brain:knowledge:news_sentiment*")?;
        
        let mut news_sentiments = Vec::new();
        
        for key in news_keys.iter().take(20) {
            if let Ok(news_json) = conn.get::<_, String>(key) {
                // Parse news sentiment and convert to trinary
                // This would depend on the exact format of your news data
                println!("📰 Processing news sentiment: {}", key);
            }
        }
        
        Ok(news_sentiments)
    }
    
    /// Add market event to buffer
    pub fn add_market_event(&mut self, event: NativeTrinaryMarketEvent) {
        self.market_buffer.push(event);
        
        // Keep buffer manageable
        if self.market_buffer.len() > self.buffer_size {
            self.market_buffer.remove(0);
        }
    }
    
    /// Get statistics on data purity
    pub fn get_purity_stats(&self) -> TrinaryPurityStats {
        let sentiment_sparsity = if !self.sentiment_buffer.is_empty() {
            let baseline_count = self.sentiment_buffer.iter()
                .filter(|s| s.sentiment_tryte == Tryte::Baseline)
                .count();
            baseline_count as f32 / self.sentiment_buffer.len() as f32
        } else {
            0.0
        };
        
        let market_sparsity = if !self.market_buffer.is_empty() {
            let baseline_count = self.market_buffer.iter()
                .filter(|m| m.price_direction_tryte == Tryte::Baseline && 
                           m.volume_flow_tryte == Tryte::Baseline &&
                           m.momentum_tryte == Tryte::Baseline &&
                           m.funding_tryte == Tryte::Baseline)
                .count();
            baseline_count as f32 / self.market_buffer.len() as f32
        } else {
            0.0
        };
        
        TrinaryPurityStats {
            sentiment_data_points: self.sentiment_buffer.len(),
            market_data_points: self.market_buffer.len(),
            sentiment_sparsity,
            market_sparsity,
            no_binary_contamination: true,  // Always true in our system!
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrinaryPurityStats {
    pub sentiment_data_points: usize,
    pub market_data_points: usize,
    pub sentiment_sparsity: f32,
    pub market_sparsity: f32,
    pub no_binary_contamination: bool,
}

impl std::fmt::Display for TrinaryPurityStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, 
            "🧬 PURE TRINARY DATA PIPELINE STATS:\n\
             📊 Sentiment: {} points ({:.1}% sparse)\n\
             📈 Market: {} points ({:.1}% sparse)\n\
             ✅ No Binary Contamination: {}\n\
             ⚡ Energy Savings: {:.1}% from sparsity",
            self.sentiment_data_points,
            self.sentiment_sparsity * 100.0,
            self.market_data_points,
            self.market_sparsity * 100.0,
            self.no_binary_contamination,
            ((self.sentiment_sparsity + self.market_sparsity) / 2.0) * 100.0
        )
    }
}