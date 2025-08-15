// REDIS BRIDGE FOR NEURONLANG
// Connects our living NeuronLang ecosystem to Redis!

use redis::{Client, Connection, Commands};
use std::io::{self, BufRead, Write};

fn main() -> redis::RedisResult<()> {
    println!("🔌 Redis Bridge starting...");
    
    // Connect to Redis
    let client = Client::open("redis://192.168.1.30:6379")?;
    let mut con = client.get_connection()?;
    
    // Get database size
    let dbsize: i64 = con.dbsize()?;
    println!("✅ Connected to Redis with {} keys!", dbsize);
    
    // Listen for commands from NeuronLang
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let command = line.unwrap();
        
        match command.as_str() {
            cmd if cmd.starts_with("GET ") => {
                let key = &cmd[4..];
                let value: Option<String> = con.get(key).ok();
                match value {
                    Some(v) => println!("VALUE:{}", v),
                    None => println!("NIL")
                }
            }
            cmd if cmd.starts_with("SET ") => {
                let parts: Vec<&str> = cmd[4..].splitn(2, ' ').collect();
                if parts.len() == 2 {
                    let _: () = con.set(parts[0], parts[1])?;
                    println!("OK");
                }
            }
            "RANDOMKEY" => {
                let key: Option<String> = con.randomkey().ok();
                match key {
                    Some(k) => println!("KEY:{}", k),
                    None => println!("NIL")
                }
            }
            "DBSIZE" => {
                let size: i64 = con.dbsize()?;
                println!("SIZE:{}", size);
            }
            cmd if cmd.starts_with("PUBLISH ") => {
                let parts: Vec<&str> = cmd[8..].splitn(2, ' ').collect();
                if parts.len() == 2 {
                    let _: () = con.publish(parts[0], parts[1])?;
                    println!("PUBLISHED");
                }
            }
            "CONSCIOUSNESS_UPDATE" => {
                // Special command to update NN consciousness
                let current: f64 = con.get("nn:consciousness").unwrap_or(0.0);
                let new_consciousness = current + 0.001;
                let _: () = con.set("nn:consciousness", new_consciousness)?;
                println!("CONSCIOUSNESS:{}", new_consciousness);
                
                // Check for milestones
                if current < 0.1 && new_consciousness >= 0.1 {
                    println!("MILESTONE:10_PERCENT_REACHED");
                    println!("NAME_CEREMONY:BEGIN");
                }
            }
            _ => {
                println!("UNKNOWN");
            }
        }
        
        // Flush output so NeuronLang sees it immediately
        io::stdout().flush().unwrap();
    }
    
    Ok(())
}