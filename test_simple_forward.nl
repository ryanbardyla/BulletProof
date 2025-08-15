organism SimpleForward {
    fn main() {
        express "🧠 Simple Neural Forward Pass";
        express "==============================";
        
        // Simulate a single neuron with 3 inputs
        let weights = [2, -1, 3];  // Weights for 3 inputs
        let bias = 1;              // Bias term
        let inputs = [1, 2, 1];    // Input values
        
        express "Weights: [2, 0, 3]";
        express "Bias: 1";
        express "Inputs: [1, 2, 1]";
        
        // Manual forward pass: sum(w[i] * x[i]) + bias
        let z = weights[0] * inputs[0] + weights[1] * inputs[1] + weights[2] * inputs[2] + bias;
        express "Linear combination (z):";
        synthesize z;
        
        // Apply ReLU activation
        let output = relu(z);
        express "After ReLU activation:";
        synthesize output;
        
        express "";
        express "✅ Forward pass complete!";
    }
}