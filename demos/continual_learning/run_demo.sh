#!/bin/bash

# NeuronLang Continual Learning Demo Runner
# The killer demo that proves we solved catastrophic forgetting

set -e

echo "🧬 NEURONLANG CONTINUAL LEARNING DEMO LAUNCHER 🧬"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo -e "${RED}Error: Not in continual_learning directory${NC}"
    echo "Please run from: demos/continual_learning/"
    exit 1
fi

# Build the demo
echo -e "${BLUE}📦 Building demo in release mode...${NC}"
cargo build --release

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed! Check error messages above.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Build successful!${NC}"
echo ""

# Run the demo
echo -e "${YELLOW}🚀 Launching the killer demo...${NC}"
echo -e "${YELLOW}Watch as the AI learns without forgetting!${NC}"
echo ""

# Set environment for optimal performance
export RUST_LOG=info
export RUST_BACKTRACE=1

# Run with timing
time cargo run --release

# Check if results were generated
if [ -f "continual_learning_results.json" ]; then
    echo ""
    echo -e "${GREEN}✅ Demo completed successfully!${NC}"
    echo -e "${GREEN}📊 Results saved to: continual_learning_results.json${NC}"
    
    # Pretty print results
    echo ""
    echo "📈 Quick Results Summary:"
    cat continual_learning_results.json | python3 -m json.tool | head -20
    
    echo ""
    echo -e "${YELLOW}🎬 Ready to record demo video?${NC}"
    echo "1. Install asciinema: sudo apt install asciinema"
    echo "2. Start recording: asciinema rec demo.cast"
    echo "3. Run demo: ./run_demo.sh"
    echo "4. Stop recording: Ctrl+D"
    echo "5. Upload: asciinema upload demo.cast"
else
    echo -e "${RED}Warning: Results file not generated${NC}"
fi

echo ""
echo "=================================================="
echo -e "${BLUE}Share your results: #NeuronLang #NoForgetting${NC}"
echo "=================================================="