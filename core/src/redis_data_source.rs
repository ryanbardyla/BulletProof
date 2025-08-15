use redis::aio::MultiplexedConnection;
use redis::{AsyncCommands, Client};
use serde_json::Value;
use anyhow::{Result, anyhow};
use tracing::{info, warn};

pub struct RedisDataSource {
    conn: MultiplexedConnection,
    client: Client,
}

impl RedisDataSource {
    pub async fn new(redis_url: &str) -> Result<Self> {
        info!("📡 Connecting to Redis at {}", redis_url);
        let client = Client::open(redis_url)?;
        let conn = client.get_multiplexed_async_connection().await?;
        
        info!("✅ Redis connection established!");
        Ok(Self { conn, client })
    }
    
    pub async fn get_latest_market_data(&mut self) -> Result<Value> {
        // Get from HyperLiquid market snapshot
        let data: Option<String> = self.conn.get("datalake:market:latest").await?;
        
        if let Some(json_str) = data {
            Ok(serde_json::from_str(&json_str)?)
        } else {
            // Return default if no data yet
            Ok(serde_json::json!({
                "price_change": 0.0,
                "volume": 1000000.0,
                "bid": 0.0,
                "ask": 0.0,
                "timestamp": chrono::Utc::now().timestamp()
            }))
        }
    }
    
    pub async fn get_latest_sentiment(&mut self) -> Result<Value> {
        // Get latest Reddit sentiment
        let data: Option<String> = self.conn.get("fenrisa:sentiment:latest").await?;
        
        if let Some(json_str) = data {
            Ok(serde_json::from_str(&json_str)?)
        } else {
            Ok(serde_json::json!({
                "value": 0.0,
                "confidence": 0.5,
                "source": "reddit",
                "timestamp": chrono::Utc::now().timestamp()
            }))
        }
    }
    
    pub async fn publish_decision(&mut self, decision: &[i8]) -> Result<()> {
        // Publish trading decision to Redis
        let action = match (decision.get(0), decision.get(1), decision.get(2)) {
            (Some(1), _, _) => "BUY",
            (_, Some(1), _) => "HOLD",
            (_, _, Some(1)) => "SELL",
            _ => "WAIT",
        };
        
        let decision_json = serde_json::json!({
            "action": action,
            "neurons": decision,
            "timestamp": chrono::Utc::now().timestamp_millis(),
            "source": "neuronlang_gpu"
        });
        
        self.conn.publish::<_, _, ()>(
            "neuronlang:trading:decision",
            decision_json.to_string()
        ).await?;
        
        Ok(())
    }
}