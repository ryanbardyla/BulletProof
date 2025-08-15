organism SimpleForward2 {
    fn main() {
        express "Testing forward pass...";
        
        // Create arrays
        let w = [2, 1, 3];
        let x = [1, 2, 1];
        let b = 5;
        
        // Calculate weighted sum manually
        let w0 = w[0];
        let x0 = x[0];
        let p0 = w0 * x0;
        
        let w1 = w[1];
        let x1 = x[1];
        let p1 = w1 * x1;
        
        let w2 = w[2];
        let x2 = x[2];
        let p2 = w2 * x2;
        
        let sum = p0 + p1 + p2 + b;
        express "Weighted sum:";
        synthesize sum;
        
        let output = relu(sum);
        express "After ReLU:";
        synthesize output;
        
        express "Done!";
    }
}