// Simple math test
organism MathTest {
    fn birth() {
        express "Testing math functions..."
        
        express "Random normal:"
        let r1 = randn()
        synthesize r1
        
        express "Random uniform:"
        let r2 = random()
        synthesize r2
        
        express "ReLU(5):"
        let relu_pos = relu(5)
        synthesize relu_pos
        
        express "Sigmoid(2):"
        let sig_pos = sigmoid(2)
        synthesize sig_pos
        
        express "Math functions working!"
    }
}
