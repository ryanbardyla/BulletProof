// Test array with math operations
organism ArrayMathTest {
    fn birth() {
        express "=== ARRAY MATH TEST ==="
        express ""
        
        // Neural network weights!
        let layer1_weights = [100, 200, 300, 400]
        let layer2_weights = [50, 150, 250]
        
        express "Layer 1 weights: [100, 200, 300, 400]"
        express "Layer 2 weights: [50, 150, 250]"
        express ""
        
        // Compute with arrays
        let w1 = layer1_weights[0]
        let w2 = layer1_weights[1]
        let sum = w1 + w2
        
        express "Weight sum: w1 + w2 = 300"
        express ""
        
        // Simulate neuron computation
        let input = 10
        let weight = layer1_weights[2]
        let output = input * weight
        
        express "Neuron output: 10 * 300 = 3000"
        express ""
        
        express "We can build neural networks now!"
    }
}