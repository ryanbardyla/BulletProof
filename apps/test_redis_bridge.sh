#!/bin/bash
# Test Redis pub/sub communication between ConsciousDB and Neural Networks

echo "🔗 Testing Redis Pub/Sub Bridge"
echo "════════════════════════════════"

# Test 1: Publish market data to Redis
echo ""
echo "📊 Publishing market data to Redis..."
redis-cli -h 192.168.1.30 PUBLISH "fenrisa:market:btc" '{"price": 54321, "volume": 1234567, "timestamp": "'$(date +%s)'"}'
redis-cli -h 192.168.1.30 PUBLISH "fenrisa:market:eth" '{"price": 3456, "volume": 987654, "timestamp": "'$(date +%s)'"}'

# Test 2: Store patterns in Redis for DB
echo ""
echo "🧠 Storing patterns for ConsciousDB..."
redis-cli -h 192.168.1.30 SET "consciousdb:pattern:whale_accumulation" '{"confidence": 0.85, "action": "BUY"}'
redis-cli -h 192.168.1.30 SET "consciousdb:pattern:support_bounce" '{"confidence": 0.72, "action": "HOLD"}'

# Test 3: Publish NN insights
echo ""
echo "💡 Publishing Neural Network insights..."
redis-cli -h 192.168.1.30 PUBLISH "fenrisa:nn:insight" "Detected bullish divergence on BTC 4H chart"
redis-cli -h 192.168.1.30 PUBLISH "fenrisa:nn:prediction" '{"asset": "BTC", "direction": "UP", "target": 55000, "timeframe": "1h"}'

# Test 4: DB queries for NN
echo ""
echo "🔍 ConsciousDB querying Neural Network..."
redis-cli -h 192.168.1.30 PUBLISH "fenrisa:db:query" "PREDICT btc_price NEXT 1 hour"

# Test 5: Check stored data
echo ""
echo "📦 Checking stored data..."
redis-cli -h 192.168.1.30 GET "consciousdb:pattern:whale_accumulation"

echo ""
echo "✅ Redis bridge test complete!"