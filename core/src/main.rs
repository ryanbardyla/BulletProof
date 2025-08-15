//! NeuronLang Core - Trinary Computing Neural Network
//! 
//! Main entry point for testing the complete tabula rasa learning system.
//! This demonstrates:
//! - Pure trinary neural networks (no binary contamination!)
//! - Biological protein synthesis memory
//! - Real market data integration
//! - Word association learning from scratch
//! - Brain tone integration
//! 
//! NO SIMULATIONS - ALL REAL DATA!

mod tryte;
mod protein_synthesis;
mod sparse_network;
mod sparse_trith_net;
mod dna_compression;
mod memory_substrate;
mod real_brain;
mod native_trinary_ingestion;
mod word_association_learning;
mod complete_learning_system;
mod brain_tone_integration;
mod integrated_test_system;
mod real_data_compression_test;
mod real_time_training_system;
mod ewc_trinary_implementation;
mod spiking_trinary_dynamics;

use integrated_test_system::IntegratedTestSystem;
use real_data_compression_test::RealDataCompressionTester;
use real_time_training_system::{RealTimeTrainingSystem, RealTimeTrainingConfig};
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 === NEURONLANG: TRINARY NEURAL NETWORK SYSTEM ===");
    println!("🧬 Pure biological learning with no pre-training!");
    println!("⚡ Zero energy baseline neurons for maximum efficiency");
    println!("🎯 Testing with REAL market data - no simulations!\n");
    
    // Show menu
    println!("Choose test mode:");
    println!("1. Quick Redis Data Test (30 seconds)");
    println!("2. Trinary Processing Test (1 minute)");
    println!("3. Real Data Compression Test (195MB data!)");
    println!("4. Complete Integrated Test (30 minutes)");
    println!("5. Real-Time Training System (LIVE!)");
    println!("6. Custom Test Duration");
    print!("Enter choice (1-6): ");
    io::stdout().flush()?;
    
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    
    match input.trim() {
        "1" => run_redis_test().await?,
        "2" => run_trinary_test().await?,
        "3" => run_compression_test().await?,
        "4" => run_complete_test().await?,
        "5" => run_live_training().await?,
        "6" => run_custom_test().await?,
        _ => {
            println!("Invalid choice, running compression test...");
            run_compression_test().await?;
        }
    }
    
    Ok(())
}

/// DNA compression test on real 195MB trading data
async fn run_compression_test() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧬 === REAL DATA COMPRESSION TEST ===");
    println!("📁 Testing on 195MB of real trading data");
    println!("🔬 Proving trinary > binary compression");
    
    let mut tester = RealDataCompressionTester::new()?;
    let results = tester.test_all_real_data().await?;
    
    println!("\n{}", results);
    
    // Save results to file
    let results_json = serde_json::to_string_pretty(&serde_json::json!({
        "test_type": "real_data_compression",
        "files_processed": results.total_files_processed,
        "data_size_mb": results.total_data_size_mb,
        "trinary_compression_ratio": results.compression_ratio_trinary,
        "gzip_compression_ratio": results.compression_ratio_gzip,
        "trinary_advantage": results.trinary_advantage,
        "sparsity_percentage": results.trinary_sparsity_percentage * 100.0,
        "energy_savings": results.energy_savings_estimate * 100.0,
        "processing_time_ms": results.processing_time_ms
    }))?;
    
    std::fs::write("dna_compression_results.json", results_json)?;
    println!("📝 Results saved to dna_compression_results.json");
    
    if results.trinary_advantage > 1.5 {
        println!("\n🎉 TRINARY COMPRESSION SUPERIORITY PROVEN!");
        println!("🚀 Trinary DNA compression is {:.1}x better than traditional methods!", results.trinary_advantage);
    } else {
        println!("\n📊 Test completed - results show compression capabilities");
    }
    
    Ok(())
}

/// Quick Redis data source test
async fn run_redis_test() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 === REDIS DATA SOURCE TEST ===");
    
    let test_system = IntegratedTestSystem::new().await?;
    let redis_results = test_system.test_redis_data_sources().await?;
    
    println!("\n📊 REDIS TEST RESULTS:");
    println!("📱 Social data points: {}", redis_results.social_data_points);
    println!("🧠 Brain knowledge points: {}", redis_results.brain_knowledge_points);
    println!("🧬 NeuroTrader accessible: {}", redis_results.neurotrader_accessible);
    println!("😊 Emotion correlation points: {}", redis_results.emotion_correlation_points);
    println!("📋 Total data points found: {}", redis_results.total_keys_found);
    
    if redis_results.total_keys_found > 0 {
        println!("✅ Redis data sources are accessible and populated!");
    } else {
        println!("⚠️  No data found in Redis. Check data collection services.");
    }
    
    Ok(())
}

/// Trinary processing test
async fn run_trinary_test() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧬 === TRINARY PROCESSING TEST ===");
    
    let test_system = IntegratedTestSystem::new().await?;
    let trinary_results = test_system.test_trinary_processing().await?;
    
    println!("\n📊 TRINARY PROCESSING RESULTS:");
    println!("📚 Words extracted: {}", trinary_results.words_extracted);
    println!("⚡ Trinary conversions: {}", trinary_results.trinary_conversions);
    println!("🔋 Baseline neurons: {}", trinary_results.baseline_count);
    println!("💡 Sparsity achieved: {:.1}%", trinary_results.sparsity_percentage);
    
    if trinary_results.word_extraction_successful {
        println!("✅ Trinary processing working perfectly!");
        println!("🔋 Energy savings from sparsity: {:.1}%", trinary_results.sparsity_percentage);
    } else {
        println!("❌ Trinary processing has issues");
    }
    
    Ok(())
}

/// Complete integrated test (30 minutes)
async fn run_complete_test() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 === COMPLETE INTEGRATED TEST ===");
    println!("⏱️  Duration: 30 minutes");
    println!("🧠 Testing all systems with real data streams");
    
    let mut test_system = IntegratedTestSystem::new().await?;
    let results = test_system.run_complete_test().await?;
    
    println!("\n🎉 TEST COMPLETED SUCCESSFULLY!");
    println!("{}", results);
    
    // Save results to file
    let results_json = serde_json::to_string_pretty(&serde_json::json!({
        "test_duration_seconds": results.test_duration_seconds,
        "total_operations": results.total_brain_tone_updates + results.total_social_events + results.total_market_validations,
        "energy_savings": results.energy_savings_achieved,
        "redis_success_rate": results.redis_success_rate,
        "prediction_accuracy": results.final_prediction_accuracy,
        "word_associations": results.total_word_associations,
        "memory_consolidations": results.total_memory_consolidations
    }))?;
    
    std::fs::write("neuronlang_test_results.json", results_json)?;
    println!("📝 Results saved to neuronlang_test_results.json");
    
    Ok(())
}

/// Real-time training system with live market data
async fn run_live_training() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 === REAL-TIME TRAINING SYSTEM ===");
    println!("📡 Training with LIVE market data from Redis & HyperLiquid");
    println!("🧠 Continuous tabula rasa learning with no pre-training");
    println!("⚡ Energy-efficient sparse trinary processing\n");
    
    println!("Configuration options:");
    println!("1. Fast Training (high learning rate, 50ms polling)");
    println!("2. Balanced Training (medium settings)");
    println!("3. Conservative Training (slow but stable)");
    print!("Choose config (1-3, or Enter for balanced): ");
    io::stdout().flush()?;
    
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    
    let config = match input.trim() {
        "1" => RealTimeTrainingConfig {
            learning_rate: 0.01,
            redis_poll_ms: 50,
            batch_size: 16,
            energy_threshold: 85.0,
            ..Default::default()
        },
        "3" => RealTimeTrainingConfig {
            learning_rate: 0.0001,
            redis_poll_ms: 500,
            batch_size: 64,
            energy_threshold: 75.0,
            ..Default::default()
        },
        _ => RealTimeTrainingConfig::default(), // Balanced
    };
    
    println!("🎯 Starting live training with:");
    println!("  📊 Learning rate: {}", config.learning_rate);
    println!("  ⏱️  Polling rate: {}ms", config.redis_poll_ms);
    println!("  🔋 Energy target: {:.1}% sparsity", config.energy_threshold);
    println!("  📦 Batch size: {}", config.batch_size);
    println!("\n⚠️  Press Ctrl+C to stop training and save checkpoint\n");
    
    let mut training_system = RealTimeTrainingSystem::new(config)?;
    
    // Start training (will run indefinitely until interrupted)
    match training_system.start_training().await {
        Ok(_) => {
            println!("Training completed successfully!");
        },
        Err(e) => {
            println!("Training interrupted: {}", e);
            println!("💾 Saving emergency checkpoint...");
            training_system.save_checkpoint("emergency_checkpoint.json").await?;
        }
    }
    
    // Save final checkpoint
    training_system.save_checkpoint("final_training_checkpoint.json").await?;
    
    // Display final metrics
    let metrics = training_system.get_metrics();
    println!("\n🎉 === TRAINING SESSION COMPLETE ===");
    println!("📊 Total samples processed: {}", metrics.total_samples);
    println!("📈 Training steps: {}", metrics.training_steps);
    println!("🎯 Final validation accuracy: {:.1}%", metrics.validation_accuracy * 100.0);
    println!("⚡ Energy efficiency achieved: {:.1}%", metrics.energy_efficiency);
    println!("🧬 Memory consolidations: {}", metrics.memory_consolidations);
    
    if metrics.catastrophic_forgetting_prevented {
        println!("🛡️  Successfully prevented catastrophic forgetting!");
    }
    
    if metrics.energy_efficiency >= 80.0 {
        println!("🎉 Energy efficiency target exceeded!");
    }
    
    Ok(())
}

/// Custom duration test
async fn run_custom_test() -> Result<(), Box<dyn std::error::Error>> {
    print!("Enter test duration in minutes: ");
    io::stdout().flush()?;
    
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    
    let duration: u64 = input.trim().parse().unwrap_or(5);
    
    println!("\n🎯 === CUSTOM TEST ({} minutes) ===", duration);
    
    let mut test_system = IntegratedTestSystem::new().await?;
    test_system.test_duration_minutes = duration;
    
    let results = test_system.run_complete_test().await?;
    
    println!("\n🎉 CUSTOM TEST COMPLETED!");
    println!("{}", results);
    
    Ok(())
}