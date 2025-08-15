//! Word Association Learning - Tabula Rasa Style
//! 
//! Learns word meanings through temporal correlation with market outcomes.
//! NO pre-programmed word meanings - pure biological association formation!

use crate::tryte::Tryte;
use crate::protein_synthesis::{ProteinSynthesisNeuron, ProteinType, MemoryFormation};
use crate::native_trinary_ingestion::{SocialMediaEvent, NativeTrinaryMarketEvent};

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};

/// A word-outcome association learned through experience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordAssociation {
    pub word: String,
    pub positive_outcomes: u32,      // Times word preceded price increase
    pub negative_outcomes: u32,      // Times word preceded price decrease
    pub neutral_outcomes: u32,       // Times word had no effect
    pub confidence: f32,             // Statistical confidence in association
    pub last_seen: u64,              // When we last saw this word
    pub protein_strength: f32,       // CREB level when association formed
}

/// Learns associations between words and market outcomes
pub struct TabulaRasaWordLearner {
    /// All word associations learned from experience
    pub associations: Arc<RwLock<HashMap<String, WordAssociation>>>,
    
    /// Recent social events waiting for outcome validation
    pub pending_validations: Arc<RwLock<Vec<PendingAssociation>>>,
    
    /// Protein synthesis neuron for memory consolidation
    pub memory_neuron: ProteinSynthesisNeuron,
    
    /// Learning parameters
    pub validation_window: u64,     // How long to wait for market outcome (seconds)
    pub min_confidence_threshold: f32,
    pub learning_rate: f32,
}

/// Social event waiting for market outcome to validate association
#[derive(Debug, Clone)]
pub struct PendingAssociation {
    pub words: Vec<String>,
    pub social_event: SocialMediaEvent,
    pub timestamp: u64,
    pub symbol: String,
    pub baseline_price: f64,         // Price when social event occurred
}

/// Market outcome used to validate word associations
#[derive(Debug, Clone)]
pub struct MarketOutcome {
    pub price_change_percent: f64,
    pub time_to_outcome: u64,        // Seconds from social event to outcome
    pub outcome_type: OutcomeType,
}

#[derive(Debug, Clone)]
pub enum OutcomeType {
    StrongPositive,    // > 2% increase
    WeakPositive,      // 0.5-2% increase  
    Neutral,           // -0.5% to +0.5%
    WeakNegative,      // -2% to -0.5% decrease
    StrongNegative,    // > 2% decrease
}

impl TabulaRasaWordLearner {
    pub fn new() -> Self {
        println!("🧠 Initializing Tabula Rasa Word Learning System...");
        println!("📚 NO pre-programmed word meanings - learning from pure experience!");
        
        Self {
            associations: Arc::new(RwLock::new(HashMap::new())),
            pending_validations: Arc::new(RwLock::new(Vec::new())),
            memory_neuron: ProteinSynthesisNeuron::new(),
            validation_window: 3600,  // 1 hour window
            min_confidence_threshold: 0.6,
            learning_rate: 0.1,
        }
    }
    
    /// Process social media event and extract words for learning
    pub fn process_social_event(&mut self, event: SocialMediaEvent, current_price: f64) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        // Extract meaningful words from title (simple tokenization)
        let words = self.extract_meaningful_words(&event.title);
        
        // Create pending association for outcome validation
        let pending = PendingAssociation {
            words: words.clone(),
            social_event: event.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            symbol: event.symbol.clone(),
            baseline_price: current_price,
        };
        
        // Add to pending validations
        {
            let mut pending_validations = self.pending_validations.write().unwrap();
            pending_validations.push(pending);
            
            // Clean up old pending validations (beyond window)
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            pending_validations.retain(|p| now - p.timestamp < self.validation_window);
        }
        
        println!("📝 Extracted {} words from social event: {:?}", words.len(), words);
        
        Ok(words)
    }
    
    /// Validate pending associations with market outcome
    pub fn validate_with_market_outcome(&mut self, symbol: &str, new_price: f64) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let mut validated_words = Vec::new();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        // Collect pending validations to process (to avoid borrow issues)
        let pending_to_process: Vec<(usize, PendingAssociation)> = {
            let pending_validations = self.pending_validations.read().unwrap();
            pending_validations
                .iter()
                .enumerate()
                .filter(|(_, p)| p.symbol == symbol)
                .map(|(i, p)| (i, p.clone()))
                .collect()
        };
        
        // Process each pending validation
        for (index, pending) in pending_to_process.iter().rev() {  // Reverse to avoid index issues
            // Calculate market outcome
            let price_change_percent = (new_price - pending.baseline_price) / pending.baseline_price;
            let time_to_outcome = now - pending.timestamp;
            
            let outcome_type = self.classify_outcome(price_change_percent);
            
            let market_outcome = MarketOutcome {
                price_change_percent,
                time_to_outcome,
                outcome_type: outcome_type.clone(),
            };
            
            // Update word associations based on outcome
            self.update_word_associations(&pending.words, &market_outcome)?;
            validated_words.extend(pending.words.clone());
            
            println!("✅ Validated association: {} words → {:?} ({:.2}% price change)", 
                     pending.words.len(), outcome_type, price_change_percent * 100.0);
        }
        
        // Remove processed validations
        {
            let mut pending_validations = self.pending_validations.write().unwrap();
            let indices_to_remove: Vec<usize> = pending_to_process.iter().map(|(i, _)| *i).collect();
            for &index in indices_to_remove.iter().rev() {
                pending_validations.remove(index);
            }
        }
        
        Ok(validated_words)
    }
    
    /// Extract meaningful words from social media text
    fn extract_meaningful_words(&self, text: &str) -> Vec<String> {
        // Simple but effective word extraction
        let words: Vec<String> = text
            .to_lowercase()
            .split_whitespace()
            .filter(|word| word.len() > 3)  // Skip short words
            .filter(|word| !self.is_stop_word(word))  // Skip common words
            .map(|word| {
                // Clean punctuation
                word.chars()
                    .filter(|c| c.is_alphabetic())
                    .collect::<String>()
            })
            .filter(|word| word.len() > 3)
            .take(10)  // Limit to top 10 words per post
            .collect();
        
        words
    }
    
    /// Check if word is a stop word (common words with no meaning)
    fn is_stop_word(&self, word: &str) -> bool {
        matches!(word, 
            "the" | "and" | "for" | "are" | "but" | "not" | "you" | "all" | 
            "can" | "had" | "her" | "was" | "one" | "our" | "out" | "day" |
            "get" | "has" | "him" | "his" | "how" | "its" | "may" | "new" |
            "now" | "old" | "see" | "two" | "who" | "boy" | "did" | "why" |
            "this" | "that" | "with" | "have" | "from" | "they" | "know" |
            "want" | "been" | "good" | "much" | "some" | "time" | "very" |
            "when" | "come" | "here" | "just" | "like" | "long" | "make" |
            "many" | "over" | "such" | "take" | "than" | "them" | "well" |
            "were" | "will"
        )
    }
    
    /// Classify market outcome based on price change
    fn classify_outcome(&self, price_change_percent: f64) -> OutcomeType {
        match price_change_percent {
            p if p > 0.02 => OutcomeType::StrongPositive,
            p if p > 0.005 => OutcomeType::WeakPositive,
            p if p < -0.02 => OutcomeType::StrongNegative,
            p if p < -0.005 => OutcomeType::WeakNegative,
            _ => OutcomeType::Neutral,
        }
    }
    
    /// Update word associations based on market outcome (CORE LEARNING!)
    fn update_word_associations(&mut self, words: &[String], outcome: &MarketOutcome) -> Result<(), Box<dyn std::error::Error>> {
        let mut associations = self.associations.write().unwrap();
        
        for word in words {
            let association = associations.entry(word.clone())
                .or_insert_with(|| WordAssociation {
                    word: word.clone(),
                    positive_outcomes: 0,
                    negative_outcomes: 0,
                    neutral_outcomes: 0,
                    confidence: 0.0,
                    last_seen: 0,
                    protein_strength: 0.0,
                });
            
            // Update outcome counts based on market result
            match outcome.outcome_type {
                OutcomeType::StrongPositive | OutcomeType::WeakPositive => {
                    association.positive_outcomes += 1;
                },
                OutcomeType::StrongNegative | OutcomeType::WeakNegative => {
                    association.negative_outcomes += 1;
                },
                OutcomeType::Neutral => {
                    association.neutral_outcomes += 1;
                },
            }
            
            association.last_seen = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            // Calculate statistical confidence
            let total_outcomes = association.positive_outcomes + association.negative_outcomes + association.neutral_outcomes;
            if total_outcomes >= 5 {  // Need at least 5 examples
                let positive_ratio = association.positive_outcomes as f32 / total_outcomes as f32;
                let negative_ratio = association.negative_outcomes as f32 / total_outcomes as f32;
                
                // Confidence based on how skewed the outcomes are
                association.confidence = (positive_ratio - 0.5).abs() + (negative_ratio - 0.5).abs();
            }
            
            // Protein synthesis for strong associations (biological memory!)
            if association.confidence > 0.3 && total_outcomes >= 10 {
                let protein_result = self.memory_neuron.process_with_proteins(
                    association.confidence, 
                    total_outcomes
                );
                
                association.protein_strength = self.memory_neuron.proteins[&ProteinType::CREB];
                
                // Consolidate memory for strong associations
                if matches!(protein_result, MemoryFormation::LatePhaseLTP | MemoryFormation::LongTermMemory) {
                    self.memory_neuron.consolidate_memory();
                    
                    println!("🧠 MEMORY CONSOLIDATED for word '{}': {} pos, {} neg, {} neut (confidence: {:.2})", 
                             word, association.positive_outcomes, association.negative_outcomes, 
                             association.neutral_outcomes, association.confidence);
                }
            }
        }
        
        Ok(())
    }
    
    /// Convert learned word associations to trinary signals
    pub fn words_to_trinary_signal(&self, words: &[String]) -> Vec<Tryte> {
        let associations = self.associations.read().unwrap();
        let mut signals = Vec::new();
        
        for word in words {
            if let Some(association) = associations.get(word) {
                // Only use confident associations
                if association.confidence > self.min_confidence_threshold {
                    let total = association.positive_outcomes + association.negative_outcomes + association.neutral_outcomes;
                    
                    if total > 0 {
                        let positive_ratio = association.positive_outcomes as f32 / total as f32;
                        let negative_ratio = association.negative_outcomes as f32 / total as f32;
                        
                        let signal = if positive_ratio > 0.6 {
                            Tryte::Activated      // Learned: word tends to predict positive outcomes
                        } else if negative_ratio > 0.6 {
                            Tryte::Inhibited      // Learned: word tends to predict negative outcomes
                        } else {
                            Tryte::Baseline       // Learned: word is neutral (ZERO ENERGY!)
                        };
                        
                        signals.push(signal);
                    }
                } else {
                    // Low confidence associations contribute zero energy
                    signals.push(Tryte::Baseline);
                }
            } else {
                // Unknown word contributes zero energy (sparse!)
                signals.push(Tryte::Baseline);
            }
        }
        
        signals
    }
    
    /// Get statistics on learned associations
    pub fn get_learning_stats(&self) -> WordLearningStats {
        let associations = self.associations.read().unwrap();
        
        let total_words = associations.len();
        let confident_words = associations.values()
            .filter(|a| a.confidence > self.min_confidence_threshold)
            .count();
        
        let avg_confidence = if total_words > 0 {
            associations.values().map(|a| a.confidence).sum::<f32>() / total_words as f32
        } else {
            0.0
        };
        
        let pending_count = self.pending_validations.read().unwrap().len();
        
        WordLearningStats {
            total_words_learned: total_words,
            confident_associations: confident_words,
            average_confidence: avg_confidence,
            pending_validations: pending_count,
            memory_consolidation_strength: self.memory_neuron.consolidation_strength,
        }
    }
    
    /// Get the most confident word associations
    pub fn get_strongest_associations(&self, limit: usize) -> Vec<WordAssociation> {
        let associations = self.associations.read().unwrap();
        
        let mut sorted_associations: Vec<WordAssociation> = associations.values()
            .filter(|a| a.confidence > self.min_confidence_threshold)
            .cloned()
            .collect();
        
        sorted_associations.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        sorted_associations.truncate(limit);
        
        sorted_associations
    }
}

#[derive(Debug, Clone)]
pub struct WordLearningStats {
    pub total_words_learned: usize,
    pub confident_associations: usize,
    pub average_confidence: f32,
    pub pending_validations: usize,
    pub memory_consolidation_strength: f32,
}

impl std::fmt::Display for WordLearningStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, 
            "🧠 TABULA RASA WORD LEARNING STATS:\n\
             📚 Total words encountered: {}\n\
             ✅ Confident associations: {} ({:.1}%)\n\
             📊 Average confidence: {:.2}\n\
             ⏳ Pending validations: {}\n\
             🧬 Memory strength: {:.2}\n\
             💡 Learning purely from experience - NO pre-programming!",
            self.total_words_learned,
            self.confident_associations,
            if self.total_words_learned > 0 { 
                self.confident_associations as f32 / self.total_words_learned as f32 * 100.0 
            } else { 0.0 },
            self.average_confidence,
            self.pending_validations,
            self.memory_consolidation_strength
        )
    }
}