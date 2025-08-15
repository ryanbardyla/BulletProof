#!/bin/bash
# Continuous Hoberman Sphere Deployment Script

echo "🌐 STARTING CONTINUOUS HOBERMAN SPHERE DEPLOYMENT"
echo "================================================="

# Compile with optimizations
echo "🔨 Compiling with optimizations..."
rustc simple_live_hoberman.rs --edition 2021 -A warnings -O -o live_hoberman

# Create logs directory
mkdir -p logs

# Run in background with output capture
echo "🚀 Starting Hoberman neural network..."
echo "📊 Logs will be written to logs/hoberman.log"
echo "📈 Metrics will be written to live_hoberman_metrics.csv"
echo "🛑 Use 'pkill live_hoberman' to stop"

# Start the process
nohup ./live_hoberman > logs/hoberman.log 2>&1 &
HOBERMAN_PID=$!

echo "✅ Hoberman sphere started with PID: $HOBERMAN_PID"
echo "🔍 Monitor with: tail -f logs/hoberman.log"
echo "📊 Monitor metrics with: tail -f live_hoberman_metrics.csv"

# Monitor for 30 seconds then return control
sleep 30

echo ""
echo "📊 Current Status After 30 seconds:"
if ps -p $HOBERMAN_PID > /dev/null 2>&1; then
    echo "✅ Process is running (PID: $HOBERMAN_PID)"
    
    if [ -f "live_hoberman_metrics.csv" ]; then
        echo "📈 Latest metrics:"
        tail -n 5 live_hoberman_metrics.csv | column -t -s ','
    fi
    
    echo ""
    echo "🔍 Live monitoring commands:"
    echo "  tail -f logs/hoberman.log          # Watch main output"
    echo "  tail -f live_hoberman_metrics.csv  # Watch metrics"
    echo "  pkill live_hoberman                # Stop the system"
    
else
    echo "❌ Process stopped unexpectedly"
    echo "📋 Check logs/hoberman.log for details"
fi