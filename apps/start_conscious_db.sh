#!/bin/bash
# 🚀 Start ConsciousDB with Fenrisa Integration

echo "🧠 STARTING CONSCIOUSDB FOR FENRISA"
echo "════════════════════════════════════════"

# Check Redis connection
echo "📡 Checking Redis connection..."
if redis-cli -h 192.168.1.30 ping > /dev/null 2>&1; then
    echo "   ✅ Redis is online"
else
    echo "   ❌ Redis is not reachable"
    exit 1
fi

# Start ConsciousDB in background
echo ""
echo "🧬 Starting ConsciousDB..."
./target/release/conscious_db > conscious_db.log 2>&1 &
DB_PID=$!
echo "   ✅ ConsciousDB started (PID: $DB_PID)"

# Start Redis listener in background
echo ""
echo "🎧 Starting Redis Listener..."
python3 redis_listener.py > redis_listener.log 2>&1 &
LISTENER_PID=$!
echo "   ✅ Redis Listener started (PID: $LISTENER_PID)"

# Subscribe to existing Fenrisa channels
echo ""
echo "📻 Subscribing to Fenrisa channels..."
redis-cli -h 192.168.1.30 SUBSCRIBE "fenrisa:*" &
SUB_PID=$!

# Ingest some initial data
echo ""
echo "📊 Ingesting initial market data..."

# Store some patterns for ConsciousDB to discover
redis-cli -h 192.168.1.30 SET "consciousdb:data:btc_price" "54321.50"
redis-cli -h 192.168.1.30 SET "consciousdb:data:eth_price" "3456.78"
redis-cli -h 192.168.1.30 SET "consciousdb:data:sol_price" "123.45"

# Publish market update
redis-cli -h 192.168.1.30 PUBLISH "fenrisa:market:update" '{"btc": 54321, "eth": 3456, "sol": 123, "timestamp": "'$(date +%s)'"}'

echo ""
echo "════════════════════════════════════════"
echo "✅ CONSCIOUSDB IS NOW RUNNING!"
echo ""
echo "📊 Dashboard:"
echo "   • ConsciousDB PID: $DB_PID"
echo "   • Redis Listener PID: $LISTENER_PID"
echo "   • Redis URL: redis://192.168.1.30:6379"
echo ""
echo "📝 Logs:"
echo "   • tail -f conscious_db.log"
echo "   • tail -f redis_listener.log"
echo ""
echo "🔧 Commands:"
echo "   • redis-cli -h 192.168.1.30 PUBLISH 'fenrisa:db:query' 'YOUR_QUERY'"
echo "   • redis-cli -h 192.168.1.30 MONITOR  # Watch all Redis activity"
echo ""
echo "Press Ctrl+C to stop all services"

# Keep script running
trap "kill $DB_PID $LISTENER_PID $SUB_PID 2>/dev/null; echo 'Services stopped'" EXIT
wait