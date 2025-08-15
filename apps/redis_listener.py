#!/usr/bin/env python3
"""
🎧 REDIS LISTENER - Watch ConsciousDB <-> NN Communication
"""

import redis
import json
import threading
import time
from datetime import datetime

class RedisListener:
    def __init__(self):
        self.r = redis.Redis(host='192.168.1.30', port=6379, decode_responses=True)
        print("🎧 Redis Listener Active")
        print("📡 Connected to Redis at 192.168.1.30:6379")
        print("═" * 50)
        
    def listen_to_channel(self, channel_name):
        """Listen to a specific Redis channel"""
        pubsub = self.r.pubsub()
        pubsub.subscribe(channel_name)
        
        print(f"\n📻 Listening to: {channel_name}")
        for message in pubsub.listen():
            if message['type'] == 'message':
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] {channel_name}: {message['data']}")
    
    def monitor_all_channels(self):
        """Monitor all Fenrisa/ConsciousDB channels"""
        channels = [
            'fenrisa:market:*',
            'fenrisa:nn:*',
            'fenrisa:db:*',
            'fenrisa:consciousdb:*',
            'consciousdb:*'
        ]
        
        # Use psubscribe for pattern matching
        pubsub = self.r.pubsub()
        pubsub.psubscribe(*channels)
        
        print("\n🔊 Monitoring all channels:")
        for pattern in channels:
            print(f"   • {pattern}")
        print("\n" + "═" * 50)
        
        for message in pubsub.listen():
            if message['type'] == 'pmessage':
                timestamp = datetime.now().strftime("%H:%M:%S")
                channel = message['channel']
                data = message['data']
                
                # Color code by channel type
                if 'market' in channel:
                    print(f"📈 [{timestamp}] {channel}: {data}")
                elif 'nn' in channel:
                    print(f"🧠 [{timestamp}] {channel}: {data}")
                elif 'db' in channel or 'consciousdb' in channel:
                    print(f"💾 [{timestamp}] {channel}: {data}")
                else:
                    print(f"📡 [{timestamp}] {channel}: {data}")
    
    def simulate_trading_activity(self):
        """Simulate trading data flow"""
        print("\n🎯 Simulating Trading Activity...")
        
        # Market data
        market_data = {
            'btc': 54321.50,
            'eth': 3456.78,
            'sol': 123.45,
            'timestamp': int(time.time())
        }
        self.r.publish('fenrisa:market:update', json.dumps(market_data))
        print("   ✓ Published market data")
        
        # NN prediction
        prediction = {
            'asset': 'BTC',
            'action': 'BUY',
            'confidence': 0.85,
            'reason': 'Bullish divergence detected'
        }
        self.r.publish('fenrisa:nn:prediction', json.dumps(prediction))
        print("   ✓ Published NN prediction")
        
        # DB query
        self.r.publish('fenrisa:db:query', 'SELECT patterns WHERE confidence > 0.8')
        print("   ✓ Published DB query")
        
        print("\n✅ Activity simulation complete!")

def main():
    listener = RedisListener()
    
    # First simulate some activity
    listener.simulate_trading_activity()
    
    # Then monitor all channels
    try:
        listener.monitor_all_channels()
    except KeyboardInterrupt:
        print("\n\n👋 Stopping Redis Listener")

if __name__ == "__main__":
    main()