organism NeuralNetworkDemo {
    // Define a simple 2-layer neural network
    layer hidden(3, 4, relu)       // 3 inputs, 4 outputs, ReLU activation
    layer output(4, 2, sigmoid)     // 4 inputs, 2 outputs, Sigmoid activation
    
    network SimpleNet {
        hidden -> output
    }
    
    fn main() {
        express "🧠 Neural Network Test";
        express "======================";
        
        // Create input data  
        let input = [1, 0, 1];  // Using integers for now
        
        // Forward pass through hidden layer
        let hidden_out = forward(hidden, input);
        express "Hidden layer output computed";
        
        // Forward pass through output layer
        let final_out = forward(output, hidden_out);
        express "Output layer result computed";
        
        // Simulate backward pass with gradient
        let gradient = [1, -1];  // Using integers for now
        let learning_rate = 1;    // Simplified for testing
        
        let result = backward(output, gradient, 1);
        express "Backpropagation completed";
        
        express "✅ Neural network test complete!";
    }
}