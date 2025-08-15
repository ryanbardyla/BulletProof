organism TestNeuralForward {
    fn main() {
        express "🧠 Testing Neural Network Forward Pass";
        express "======================================";
        
        // Create weights and inputs
        let weights = [2, 3, 1];
        let inputs = [10, 20, 30];
        
        express "Weights: [2, 3, 1]";
        express "Inputs: [10, 20, 30]";
        
        // The forward expression expects (layer, input)
        // For now, we'll use weights as a simple layer
        let result = forward(weights, inputs);
        
        express "";
        express "Forward pass result:";
        express "Expected: 2*10 + 3*20 + 1*30 = 20 + 60 + 30 = 110";
        express "Actual result:";
        synthesize result;
        
        express "";
        express "✅ Neural forward pass complete!";
    }
}