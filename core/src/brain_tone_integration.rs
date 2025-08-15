//! Brain Tone Integration for Trinary Neural Networks
//! 
//! Integrates emotional/psychological market state with pure trinary processing.
//! Uses actual brain tone data from Redis to modulate neural network behavior.

use crate::tryte::Tryte;
use crate::protein_synthesis::{ProteinSynthesisNeuron, ProteinType, MemoryFormation};
use crate::native_trinary_ingestion::NativeTrinaryPipeline;

use std::collections::HashMap;
use redis::Commands;
use serde::{Deserialize, Serialize};

/// Processed brain tone signal for neural network input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainToneSignal {
    pub greed_index: f32,
    pub fear_index: f32,
    pub momentum_strength: f32,
    pub volatility_clustering_strength: f32,
    pub social_sentiment_velocity: f32,
    pub market_regime_confidence: f32,
}

/// Emotional state enum for market sentiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmotionalState {
    ExtremeGreed,
    Greed,
    Neutral,
    Fear,
    ExtremeFear,
}

/// Brain tone/emotional state data from Redis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainToneData {
    pub emotional_trading_correlation: f64,
    pub volatility_clustering_strength: f64,
    pub hawkes_intensity: f64,
    pub trend_strength: f64,
    pub signal_confidence: f64,
    pub market_fear_greed: f64,
    pub timestamp: u64,
}

/// NeuroTrader knowledge from brain system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuroTraderKnowledge {
    pub technical_analysis_methods: HashMap<String, String>,
    pub pattern_recognition_techniques: Vec<String>,
    pub quantitative_methods: Vec<String>,
    pub trendline_analysis: TrendlineAnalysis,
    pub rsi_pca_analysis: RSIPCAAnalysis,
    pub hawkes_volatility: HawkesVolatility,
    pub integration_priority: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendlineAnalysis {
    pub trend_slope: f64,
    pub trend_intercept: f64,
    pub trend_direction: String,
    pub support_level: f64,
    pub resistance_level: f64,
    pub trend_strength: f64,
    pub r_squared: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RSIPCAAnalysis {
    pub rsi_periods: Vec<u32>,
    pub principal_component_weights: Vec<f64>,
    pub pc_scores: Vec<f64>,
    pub latest_pc_score: f64,
    pub signal: String,
    pub signal_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HawkesVolatility {
    pub current_volatility: f64,
    pub volatility_threshold: f64,
    pub hawkes_intensity: f64,
    pub clustering_strength: f64,
    pub volatility_signal: String,
    pub high_vol_events: u32,
    pub event_rate: f64,
}

/// Brain tone processor for trinary networks
pub struct BrainToneProcessor {
    pub redis_client: redis::Client,
    pub current_brain_tone: Option<BrainToneData>,
    pub neurotrader_knowledge: Option<NeuroTraderKnowledge>,
    pub tone_history: Vec<BrainToneData>,
    pub max_history: usize,
    
    /// Protein synthesis for brain tone memory
    pub tone_memory_neuron: ProteinSynthesisNeuron,
}

impl BrainToneProcessor {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        println!("🧠 Initializing Brain Tone Processor...");
        println!("🎭 Integrating emotional/psychological market state");
        
        let redis_client = redis::Client::open("redis://192.168.1.30:6379")?;
        
        Ok(Self {
            redis_client,
            current_brain_tone: None,
            neurotrader_knowledge: None,
            tone_history: Vec::new(),
            max_history: 1000,
            tone_memory_neuron: ProteinSynthesisNeuron::new(),
        })
    }
    
    /// Fetch latest brain tone data from Redis
    pub async fn fetch_brain_tone(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut conn = self.redis_client.get_connection()?;
        
        // Get emotional trading correlation
        let emotional_data: Option<String> = conn.get("brain:knowledge:emotional_trading_correlation:1755075493").ok();
        
        // Get neurotrader knowledge
        let neurotrader_data: Option<String> = conn.get("brain:neurotrader888_knowledge").ok();
        
        if let Some(neuro_json) = neurotrader_data {
            if let Ok(neuro_knowledge) = serde_json::from_str::<NeuroTraderKnowledge>(&neuro_json) {
                println!("🧠 Loaded NeuroTrader Knowledge:");
                println!("   Trend: {} (strength: {:.2})", 
                         neuro_knowledge.trendline_analysis.trend_direction,
                         neuro_knowledge.trendline_analysis.trend_strength);
                println!("   RSI Signal: {} (strength: {:.2})", 
                         neuro_knowledge.rsi_pca_analysis.signal,
                         neuro_knowledge.rsi_pca_analysis.signal_strength);
                println!("   Hawkes Intensity: {:.2} (clustering: {:.2})", 
                         neuro_knowledge.hawkes_volatility.hawkes_intensity,
                         neuro_knowledge.hawkes_volatility.clustering_strength);
                
                // Convert to brain tone data
                let brain_tone = BrainToneData {
                    emotional_trading_correlation: if neuro_knowledge.rsi_pca_analysis.signal == "buy" { 0.7 } else { -0.3 },
                    volatility_clustering_strength: neuro_knowledge.hawkes_volatility.clustering_strength,
                    hawkes_intensity: neuro_knowledge.hawkes_volatility.hawkes_intensity,
                    trend_strength: neuro_knowledge.trendline_analysis.trend_strength.abs(),
                    signal_confidence: neuro_knowledge.rsi_pca_analysis.signal_strength.abs(),
                    market_fear_greed: self.calculate_fear_greed_from_volatility(neuro_knowledge.hawkes_volatility.current_volatility),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                };
                
                self.current_brain_tone = Some(brain_tone.clone());
                self.neurotrader_knowledge = Some(neuro_knowledge);
                
                // Add to history
                self.tone_history.push(brain_tone);
                if self.tone_history.len() > self.max_history {
                    self.tone_history.remove(0);
                }
                
                println!("✅ Brain tone updated with real neural network data");
            }
        }
        
        Ok(())
    }
    
    /// Convert brain tone to trinary signals for neural network
    pub fn brain_tone_to_trinary(&self) -> Vec<Tryte> {
        let mut signals = Vec::new();
        
        if let Some(ref tone) = self.current_brain_tone {
            // Emotional correlation → trinary
            signals.push(if tone.emotional_trading_correlation > 0.3 {
                Tryte::Activated      // Strong positive emotion
            } else if tone.emotional_trading_correlation < -0.3 {
                Tryte::Inhibited      // Strong negative emotion  
            } else {
                Tryte::Baseline       // Neutral emotion (ZERO ENERGY!)
            });
            
            // Volatility clustering → trinary
            signals.push(if tone.volatility_clustering_strength > 0.7 {
                Tryte::Activated      // High clustering (volatile)
            } else if tone.volatility_clustering_strength < 0.3 {
                Tryte::Inhibited      // Low clustering (calm)
            } else {
                Tryte::Baseline       // Normal clustering (ZERO ENERGY!)
            });
            
            // Hawkes intensity → trinary
            signals.push(if tone.hawkes_intensity > 1.5 {
                Tryte::Activated      // High market activity
            } else if tone.hawkes_intensity < 0.5 {
                Tryte::Inhibited      // Low market activity
            } else {
                Tryte::Baseline       // Normal activity (ZERO ENERGY!)
            });
            
            // Trend strength → trinary
            signals.push(if tone.trend_strength > 20.0 {
                Tryte::Activated      // Strong trend
            } else if tone.trend_strength < 5.0 {
                Tryte::Inhibited      // Weak trend
            } else {
                Tryte::Baseline       // Normal trend (ZERO ENERGY!)
            });
            
            // Signal confidence → trinary
            signals.push(if tone.signal_confidence > 1.0 {
                Tryte::Activated      // High confidence
            } else if tone.signal_confidence < 0.3 {
                Tryte::Inhibited      // Low confidence
            } else {
                Tryte::Baseline       // Normal confidence (ZERO ENERGY!)
            });
            
            // Fear/Greed → trinary
            signals.push(if tone.market_fear_greed > 0.6 {
                Tryte::Activated      // Extreme greed
            } else if tone.market_fear_greed < -0.6 {
                Tryte::Inhibited      // Extreme fear
            } else {
                Tryte::Baseline       // Neutral sentiment (ZERO ENERGY!)
            });
        }
        
        // Calculate sparsity
        let active_count = signals.iter().filter(|&&s| s != Tryte::Baseline).count();
        let sparsity = if !signals.is_empty() {
            (signals.len() - active_count) as f32 / signals.len() as f32
        } else {
            0.0
        };
        
        println!("🧠 Brain tone → {} trinary signals ({:.1}% sparse)", 
                 signals.len(), sparsity * 100.0);
        
        signals
    }
    
    /// Modulate neural network behavior based on brain tone
    pub fn modulate_neural_network(&mut self, base_signals: &[Tryte]) -> Vec<Tryte> {
        let mut modulated_signals = base_signals.to_vec();
        
        if let Some(ref tone) = self.current_brain_tone {
            // High volatility clustering = more cautious (more inhibited signals)
            if tone.volatility_clustering_strength > 0.8 {
                for signal in &mut modulated_signals {
                    if *signal == Tryte::Activated {
                        // 30% chance to downgrade to baseline (more cautious)
                        if fastrand::f32() < 0.3 {
                            *signal = Tryte::Baseline;
                        }
                    }
                }
                
                // Trigger protein synthesis for stress memory
                let stress_result = self.tone_memory_neuron.process_with_proteins(
                    tone.volatility_clustering_strength as f32, 
                    1
                );
                
                if matches!(stress_result, MemoryFormation::LatePhaseLTP) {
                    println!("🧠 STRESS MEMORY formed: high volatility clustering ({})", 
                             tone.volatility_clustering_strength);
                }
            }
            
            // Strong trend = amplify signals in trend direction
            if tone.trend_strength > 15.0 {
                let trend_direction = if let Some(ref neuro) = self.neurotrader_knowledge {
                    &neuro.trendline_analysis.trend_direction
                } else {
                    "neutral"
                };
                
                for signal in &mut modulated_signals {
                    match (*signal, trend_direction.as_ref()) {
                        (Tryte::Activated, "bullish") => {
                            // Amplify bullish signals in bullish trend
                            // (Already activated, keep as is)
                        },
                        (Tryte::Inhibited, "bearish") => {
                            // Amplify bearish signals in bearish trend
                            // (Already inhibited, keep as is)
                        },
                        (Tryte::Baseline, "bullish") => {
                            // 20% chance to activate baseline in strong bullish trend
                            if fastrand::f32() < 0.2 {
                                *signal = Tryte::Activated;
                            }
                        },
                        (Tryte::Baseline, "bearish") => {
                            // 20% chance to inhibit baseline in strong bearish trend
                            if fastrand::f32() < 0.2 {
                                *signal = Tryte::Inhibited;
                            }
                        },
                        _ => {
                            // No modulation for other cases
                        }
                    }
                }
                
                // Consolidate trend memory
                let trend_result = self.tone_memory_neuron.process_with_proteins(
                    (tone.trend_strength / 50.0) as f32,  // Normalize to 0-1
                    2
                );
                
                if matches!(trend_result, MemoryFormation::LongTermMemory) {
                    println!("🧠 TREND MEMORY consolidated: {} trend (strength: {})", 
                             trend_direction, tone.trend_strength);
                }
            }
        }
        
        modulated_signals
    }
    
    /// Process tone data from JSON to create BrainToneSignal
    pub fn process_tone_data(&self, data: &serde_json::Value) -> BrainToneSignal {
        let greed_index = data.get("market_fear_greed")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
            
        let fear_index = 1.0 - greed_index.abs(); // Inverse of greed
        
        let momentum_strength = data.get("trend_strength")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
            
        let volatility_clustering_strength = data.get("volatility_clustering_strength")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
            
        let social_sentiment_velocity = data.get("hawkes_intensity")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
            
        let market_regime_confidence = data.get("signal_confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        
        BrainToneSignal {
            greed_index,
            fear_index,
            momentum_strength,
            volatility_clustering_strength,
            social_sentiment_velocity,
            market_regime_confidence,
        }
    }
    
    /// Calculate fear/greed index from volatility
    fn calculate_fear_greed_from_volatility(&self, volatility: f64) -> f64 {
        // Simple mapping: high volatility = fear, low volatility = greed
        // Normalize around 0.01 (1% volatility as neutral)
        let normalized_vol = (volatility - 0.01) / 0.02;  // Scale around ±0.02
        
        // Invert: high volatility = negative (fear)
        -normalized_vol.clamp(-1.0, 1.0)
    }
    
    /// Get current brain tone statistics
    pub fn get_brain_tone_stats(&self) -> BrainToneStats {
        let recent_tones = if self.tone_history.len() >= 10 {
            &self.tone_history[self.tone_history.len()-10..]
        } else {
            &self.tone_history
        };
        
        let avg_emotional_correlation = if !recent_tones.is_empty() {
            recent_tones.iter().map(|t| t.emotional_trading_correlation).sum::<f64>() / recent_tones.len() as f64
        } else {
            0.0
        };
        
        let avg_volatility_clustering = if !recent_tones.is_empty() {
            recent_tones.iter().map(|t| t.volatility_clustering_strength).sum::<f64>() / recent_tones.len() as f64
        } else {
            0.0
        };
        
        let avg_hawkes_intensity = if !recent_tones.is_empty() {
            recent_tones.iter().map(|t| t.hawkes_intensity).sum::<f64>() / recent_tones.len() as f64
        } else {
            0.0
        };
        
        BrainToneStats {
            data_points: self.tone_history.len(),
            avg_emotional_correlation,
            avg_volatility_clustering,
            avg_hawkes_intensity,
            current_fear_greed: self.current_brain_tone.as_ref().map(|t| t.market_fear_greed).unwrap_or(0.0),
            memory_consolidations: self.tone_memory_neuron.consolidation_strength,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BrainToneStats {
    pub data_points: usize,
    pub avg_emotional_correlation: f64,
    pub avg_volatility_clustering: f64,
    pub avg_hawkes_intensity: f64,
    pub current_fear_greed: f64,
    pub memory_consolidations: f32,
}

impl std::fmt::Display for BrainToneStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, 
            "🧠 BRAIN TONE ANALYSIS:\n\
             📊 Data points: {}\n\
             😊 Avg emotional correlation: {:.2}\n\
             📈 Avg volatility clustering: {:.2}\n\
             ⚡ Avg Hawkes intensity: {:.2}\n\
             😱 Current fear/greed: {:.2}\n\
             🧬 Memory consolidations: {:.2}\n\
             💡 Neural state modulating trinary signals",
            self.data_points,
            self.avg_emotional_correlation,
            self.avg_volatility_clustering,
            self.avg_hawkes_intensity,
            self.current_fear_greed,
            self.memory_consolidations
        )
    }
}