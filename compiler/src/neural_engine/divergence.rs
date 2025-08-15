// 📊 DIVERGENCE TRACKING
// Learns which biological features matter by comparing implementations

use std::collections::VecDeque;

/// Tracks divergence between biological and optimized implementations
pub struct DivergenceTracker {
    // History of divergence measurements
    divergence_history: VecDeque<f32>,
    history_size: usize,
    
    // Feature importance scores
    feature_importance: FeatureImportance,
    
    // Divergence statistics
    total_measurements: u64,
    cumulative_divergence: f64,
    max_divergence: f32,
    
    // Learning parameters
    adaptation_rate: f32,
}

#[derive(Clone, Debug)]
pub struct FeatureImportance {
    pub refractory_period: f32,
    pub adaptation_current: f32,
    pub synaptic_delay: f32,
    pub homeostatic_plasticity: f32,
    pub spike_timing_precision: f32,
    pub metabolic_constraints: f32,
}

impl DivergenceTracker {
    pub fn new() -> Self {
        DivergenceTracker {
            divergence_history: VecDeque::with_capacity(1000),
            history_size: 1000,
            feature_importance: FeatureImportance {
                refractory_period: 1.0,      // Start assuming all features matter
                adaptation_current: 1.0,
                synaptic_delay: 1.0,
                homeostatic_plasticity: 1.0,
                spike_timing_precision: 1.0,
                metabolic_constraints: 0.5,   // Start assuming this matters less
            },
            total_measurements: 0,
            cumulative_divergence: 0.0,
            max_divergence: 0.0,
            adaptation_rate: 0.01,
        }
    }
    
    /// Measure divergence between spike patterns
    pub fn measure(&mut self, biological: &[bool], optimized: &[bool]) -> f32 {
        assert_eq!(biological.len(), optimized.len(), "Spike arrays must be same length");
        
        let n = biological.len() as f32;
        
        // 1. Spike count difference
        let bio_count = biological.iter().filter(|&&s| s).count() as f32;
        let opt_count = optimized.iter().filter(|&&s| s).count() as f32;
        let count_divergence = (bio_count - opt_count).abs() / n.max(1.0);
        
        // 2. Spike timing difference (Hamming distance)
        let timing_errors = biological.iter()
            .zip(optimized.iter())
            .filter(|(b, o)| b != o)
            .count() as f32;
        let timing_divergence = timing_errors / n;
        
        // 3. Pattern correlation
        let correlation = self.spike_correlation(biological, optimized);
        let correlation_divergence = 1.0 - correlation;
        
        // 4. Burst pattern difference
        let bio_bursts = self.count_bursts(biological);
        let opt_bursts = self.count_bursts(optimized);
        let burst_divergence = (bio_bursts - opt_bursts).abs() / n.max(1.0);
        
        // Weighted combination
        let total_divergence = 
            count_divergence * 0.25 +
            timing_divergence * 0.35 +
            correlation_divergence * 0.30 +
            burst_divergence * 0.10;
        
        // Update statistics
        self.total_measurements += 1;
        self.cumulative_divergence += total_divergence as f64;
        self.max_divergence = self.max_divergence.max(total_divergence);
        
        // Update history
        self.divergence_history.push_back(total_divergence);
        if self.divergence_history.len() > self.history_size {
            self.divergence_history.pop_front();
        }
        
        // Learn from divergence patterns
        self.update_feature_importance(total_divergence);
        
        total_divergence
    }
    
    /// Calculate spike correlation between patterns
    fn spike_correlation(&self, biological: &[bool], optimized: &[bool]) -> f32 {
        let n = biological.len() as f32;
        
        // Calculate means
        let bio_mean = biological.iter().filter(|&&s| s).count() as f32 / n;
        let opt_mean = optimized.iter().filter(|&&s| s).count() as f32 / n;
        
        // Calculate correlation
        let mut covariance = 0.0;
        let mut bio_variance = 0.0;
        let mut opt_variance = 0.0;
        
        for (b, o) in biological.iter().zip(optimized.iter()) {
            let b_val = if *b { 1.0 } else { 0.0 };
            let o_val = if *o { 1.0 } else { 0.0 };
            
            let b_diff = b_val - bio_mean;
            let o_diff = o_val - opt_mean;
            
            covariance += b_diff * o_diff;
            bio_variance += b_diff * b_diff;
            opt_variance += o_diff * o_diff;
        }
        
        if bio_variance * opt_variance > 0.0 {
            (covariance / (bio_variance * opt_variance).sqrt()).abs()
        } else {
            0.0
        }
    }
    
    /// Count burst patterns in spike train
    fn count_bursts(&self, spikes: &[bool]) -> f32 {
        let mut burst_count = 0.0;
        let mut consecutive_spikes = 0;
        
        for &spike in spikes {
            if spike {
                consecutive_spikes += 1;
                if consecutive_spikes >= 3 {
                    burst_count += 1.0;
                }
            } else {
                consecutive_spikes = 0;
            }
        }
        
        burst_count
    }
    
    /// Update feature importance based on divergence patterns
    fn update_feature_importance(&mut self, current_divergence: f32) {
        // If divergence is stable and low, reduce importance of complex features
        if self.is_divergence_stable() && current_divergence < 0.1 {
            // Gradually reduce importance of metabolically expensive features
            self.feature_importance.metabolic_constraints *= 1.0 - self.adaptation_rate;
            self.feature_importance.synaptic_delay *= 1.0 - self.adaptation_rate * 0.5;
        }
        
        // If divergence is high, increase importance of timing features
        if current_divergence > 0.3 {
            self.feature_importance.spike_timing_precision *= 1.0 + self.adaptation_rate;
            self.feature_importance.refractory_period *= 1.0 + self.adaptation_rate * 0.5;
        }
        
        // Normalize importance scores
        self.normalize_importance();
    }
    
    /// Check if divergence is stable over recent history
    fn is_divergence_stable(&self) -> bool {
        if self.divergence_history.len() < 100 {
            return false;
        }
        
        // Calculate variance of recent divergence
        let recent: Vec<f32> = self.divergence_history.iter()
            .rev()
            .take(100)
            .copied()
            .collect();
        
        let mean = recent.iter().sum::<f32>() / recent.len() as f32;
        let variance = recent.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / recent.len() as f32;
        
        variance < 0.01  // Low variance means stable
    }
    
    /// Normalize importance scores to sum to constant
    fn normalize_importance(&mut self) {
        let sum = self.feature_importance.refractory_period +
                  self.feature_importance.adaptation_current +
                  self.feature_importance.synaptic_delay +
                  self.feature_importance.homeostatic_plasticity +
                  self.feature_importance.spike_timing_precision +
                  self.feature_importance.metabolic_constraints;
        
        if sum > 0.0 {
            let scale = 6.0 / sum;  // Normalize to sum of 6
            self.feature_importance.refractory_period *= scale;
            self.feature_importance.adaptation_current *= scale;
            self.feature_importance.synaptic_delay *= scale;
            self.feature_importance.homeostatic_plasticity *= scale;
            self.feature_importance.spike_timing_precision *= scale;
            self.feature_importance.metabolic_constraints *= scale;
        }
    }
    
    /// Get current feature importance scores
    pub fn get_feature_importance(&self) -> &FeatureImportance {
        &self.feature_importance
    }
    
    /// Get average divergence
    pub fn average_divergence(&self) -> f32 {
        if self.total_measurements > 0 {
            (self.cumulative_divergence / self.total_measurements as f64) as f32
        } else {
            0.0
        }
    }
    
    /// Get recent divergence trend
    pub fn divergence_trend(&self) -> f32 {
        if self.divergence_history.len() < 20 {
            return 0.0;
        }
        
        let recent_avg = self.divergence_history.iter()
            .rev()
            .take(10)
            .sum::<f32>() / 10.0;
            
        let older_avg = self.divergence_history.iter()
            .rev()
            .skip(10)
            .take(10)
            .sum::<f32>() / 10.0;
        
        recent_avg - older_avg  // Positive means increasing divergence
    }
    
    /// Generate insights about what biological features matter
    pub fn generate_insights(&self) -> String {
        let mut insights = Vec::new();
        
        // Identify most important features
        let fi = &self.feature_importance;
        
        if fi.spike_timing_precision > 1.2 {
            insights.push("Spike timing precision is critical");
        }
        
        if fi.metabolic_constraints < 0.3 {
            insights.push("Metabolic constraints can be simplified");
        }
        
        if fi.refractory_period > 1.5 {
            insights.push("Refractory period dynamics are essential");
        }
        
        if fi.synaptic_delay < 0.5 {
            insights.push("Synaptic delays can be approximated");
        }
        
        if self.average_divergence() < 0.05 {
            insights.push("Optimized model captures biological behavior well");
        } else if self.average_divergence() > 0.3 {
            insights.push("Significant divergence - missing important biological features");
        }
        
        if insights.is_empty() {
            "Models are converging, continue monitoring".to_string()
        } else {
            insights.join("; ")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_identical_patterns() {
        let mut tracker = DivergenceTracker::new();
        
        let pattern = vec![true, false, true, false, false, true];
        let divergence = tracker.measure(&pattern, &pattern);
        
        assert_eq!(divergence, 0.0);
    }
    
    #[test]
    fn test_opposite_patterns() {
        let mut tracker = DivergenceTracker::new();
        
        let bio = vec![true, false, true, false];
        let opt = vec![false, true, false, true];
        
        let divergence = tracker.measure(&bio, &opt);
        
        assert!(divergence > 0.5);
    }
    
    #[test]
    fn test_feature_learning() {
        let mut tracker = DivergenceTracker::new();
        
        // Simulate low stable divergence
        for _ in 0..200 {
            let bio = vec![true, false, false, true, false];
            let opt = vec![true, false, false, true, false];
            tracker.measure(&bio, &opt);
        }
        
        // Metabolic constraints should have reduced importance
        assert!(tracker.feature_importance.metabolic_constraints < 0.5);
    }
}