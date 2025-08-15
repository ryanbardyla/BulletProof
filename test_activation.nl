organism TestActivation {
    fn main() {
        express "🧠 Testing Activation Functions";
        express "================================";
        
        // Test ReLU
        express "ReLU tests:";
        let x1 = 5;
        let r1 = relu(x1);
        express "relu(5) = ";
        synthesize r1;
        
        let x2 = -3;
        let r2 = relu(x2);
        express "relu(-3) = ";
        synthesize r2;
        
        // Test Sigmoid (approximation)
        express "";
        express "Sigmoid tests:";
        let s1 = sigmoid(0);
        express "sigmoid(0) ≈ ";
        synthesize s1;
        
        let s2 = sigmoid(100);
        express "sigmoid(100) ≈ ";
        synthesize s2;
        
        let x3 = -100;
        let s3 = sigmoid(x3);
        express "sigmoid(-100) ≈ ";
        synthesize s3;
        
        express "";
        express "✅ Activation functions tested!";
    }
}