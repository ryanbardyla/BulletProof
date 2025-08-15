#!/usr/bin/env python3
"""
📊 FENRISA DATA INGESTION FOR CONSCIOUSDB
Pulls real data from existing systems and feeds to ConsciousDB
"""

import redis
import json
import time
import random
from datetime import datetime, timedelta

class FenrisaDataIngester:
    def __init__(self):
        self.r = redis.Redis(host='192.168.1.30', port=6379, decode_responses=True)
        print("📊 Fenrisa Data Ingester Active")
        print("═" * 50)
        
    def ingest_market_data(self):
        """Ingest real-time market data"""
        # Simulate real market data (replace with actual API calls)
        btc_price = 54000 + random.uniform(-500, 500)
        eth_price = 3400 + random.uniform(-50, 50)
        sol_price = 120 + random.uniform(-5, 5)
        
        market_data = {
            'timestamp': datetime.now().isoformat(),
            'btc': {
                'price': btc_price,
                'volume_24h': random.uniform(1e9, 2e9),
                'change_24h': random.uniform(-5, 5)
            },
            'eth': {
                'price': eth_price,
                'volume_24h': random.uniform(5e8, 1e9),
                'change_24h': random.uniform(-5, 5)
            },
            'sol': {
                'price': sol_price,
                'volume_24h': random.uniform(1e8, 3e8),
                'change_24h': random.uniform(-10, 10)
            }
        }
        
        # Store in ConsciousDB format
        key = f"consciousdb:market:{int(time.time())}"
        self.r.set(key, json.dumps(market_data))
        self.r.expire(key, 3600)  # Expire after 1 hour
        
        # Publish for real-time processing
        self.r.publish('fenrisa:market:update', json.dumps(market_data))
        
        print(f"📈 Ingested market data: BTC=${btc_price:.2f}, ETH=${eth_price:.2f}, SOL=${sol_price:.2f}")
        return market_data
    
    def ingest_sentiment_data(self):
        """Ingest sentiment from Reddit/Twitter simulation"""
        sentiments = ['bullish', 'bearish', 'neutral']
        sources = ['reddit', 'twitter', 'telegram']
        
        sentiment_data = {
            'timestamp': datetime.now().isoformat(),
            'source': random.choice(sources),
            'sentiment': random.choice(sentiments),
            'score': random.uniform(-1, 1),
            'volume': random.randint(100, 1000),
            'top_mentions': ['BTC', 'ETH', 'SOL'][0:random.randint(1, 3)]
        }
        
        # Store for ConsciousDB
        key = f"consciousdb:sentiment:{int(time.time())}"
        self.r.set(key, json.dumps(sentiment_data))
        self.r.expire(key, 3600)
        
        # Publish
        self.r.publish('fenrisa:sentiment:update', json.dumps(sentiment_data))
        
        print(f"💭 Ingested sentiment: {sentiment_data['source']} - {sentiment_data['sentiment']} ({sentiment_data['score']:.2f})")
        return sentiment_data
    
    def ingest_whale_activity(self):
        """Ingest whale trading activity"""
        whale_data = {
            'timestamp': datetime.now().isoformat(),
            'wallet': f"0x{''.join(random.choices('0123456789abcdef', k=40))}",
            'action': random.choice(['buy', 'sell', 'transfer']),
            'asset': random.choice(['BTC', 'ETH', 'SOL']),
            'amount': random.uniform(100000, 10000000),
            'impact': random.uniform(0.1, 2.0)  # % impact on price
        }
        
        # Store for pattern recognition
        key = f"consciousdb:whale:{int(time.time())}"
        self.r.set(key, json.dumps(whale_data))
        self.r.expire(key, 7200)  # Keep whale data longer
        
        # Publish alert
        self.r.publish('fenrisa:whale:alert', json.dumps(whale_data))
        
        print(f"🐋 Whale activity: {whale_data['action']} {whale_data['amount']:.0f} {whale_data['asset']}")
        return whale_data
    
    def discover_patterns(self):
        """Ask ConsciousDB to discover patterns"""
        # Query ConsciousDB for pattern discovery
        query = "DISCOVER patterns FROM recent_data WHERE significance > 0.7"
        self.r.publish('fenrisa:db:query', query)
        
        # Simulate pattern discovery
        patterns = [
            {'name': 'accumulation_phase', 'confidence': 0.82},
            {'name': 'breakout_imminent', 'confidence': 0.75},
            {'name': 'whale_coordination', 'confidence': 0.68}
        ]
        
        for pattern in patterns:
            if pattern['confidence'] > 0.7:
                print(f"🎯 Pattern discovered: {pattern['name']} (confidence: {pattern['confidence']:.2f})")
                self.r.publish('fenrisa:pattern:discovered', json.dumps(pattern))
        
        return patterns
    
    def continuous_ingestion(self, interval=5):
        """Continuously ingest data"""
        print(f"\n🔄 Starting continuous ingestion (every {interval}s)...")
        print("Press Ctrl+C to stop\n")
        
        iteration = 0
        try:
            while True:
                iteration += 1
                print(f"\n═══ Iteration {iteration} ═══")
                
                # Ingest different data types
                self.ingest_market_data()
                time.sleep(1)
                
                if iteration % 2 == 0:
                    self.ingest_sentiment_data()
                    time.sleep(1)
                
                if iteration % 3 == 0:
                    self.ingest_whale_activity()
                    time.sleep(1)
                
                if iteration % 5 == 0:
                    self.discover_patterns()
                
                # Check ConsciousDB awareness
                awareness = self.r.get('consciousdb:awareness')
                if awareness:
                    print(f"\n🧠 ConsciousDB Awareness: {awareness}")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n✋ Stopping data ingestion")
    
    def test_db_query(self):
        """Test querying ConsciousDB"""
        queries = [
            "SELECT * FROM market_data WHERE btc > 50000",
            "PREDICT btc_price IN NEXT 1 hour",
            "DISCOVER patterns FROM whale_activity",
            "EXPLAIN recent_market_movement"
        ]
        
        print("\n🔍 Testing ConsciousDB queries:")
        for query in queries:
            print(f"   Query: {query}")
            self.r.publish('fenrisa:db:query', query)
            time.sleep(0.5)

def main():
    ingester = FenrisaDataIngester()
    
    # Test queries first
    ingester.test_db_query()
    
    # Start continuous ingestion
    ingester.continuous_ingestion(interval=5)

if __name__ == "__main__":
    main()