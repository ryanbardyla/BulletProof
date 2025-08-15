// Test negative numbers
organism NegativeTest {
    fn birth() {
        express "Testing negative numbers..."
        
        let a = -5
        synthesize a
        
        let b = -10
        synthesize b
        
        express "ReLU(-3):"
        let relu_neg = relu(-3)
        synthesize relu_neg
        
        express "Sigmoid(-1):"
        let sig_neg = sigmoid(-1)
        synthesize sig_neg
        
        express "Negative numbers working!"
    }
}
