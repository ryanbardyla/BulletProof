organism ForwardWithNegative {
    fn main() {
        express "🧠 Forward Pass with Negative Weights";
        express "======================================";
        
        // Weights including negative
        let w = [2, -1, 3];
        let x = [1, 2, 1];
        let b = 5;
        
        express "Weights: [2, -1, 3]";
        express "Inputs: [1, 2, 1]";
        express "Bias: 5";
        
        // Calculate weighted sum
        let w0 = w[0];
        let x0 = x[0];
        let p0 = w0 * x0;  // 2 * 1 = 2
        
        let w1 = w[1];
        let x1 = x[1];
        let p1 = w1 * x1;  // -1 * 2 = -2
        
        let w2 = w[2];
        let x2 = x[2];
        let p2 = w2 * x2;  // 3 * 1 = 3
        
        express "";
        express "Products:";
        synthesize p0;
        synthesize p1;
        synthesize p2;
        
        let sum = p0 + p1 + p2 + b;  // 2 + (-2) + 3 + 5 = 8
        express "";
        express "Weighted sum:";
        synthesize sum;
        
        let output = relu(sum);
        express "After ReLU:";
        synthesize output;
        
        express "";
        express "✅ Forward pass complete!";
    }
}