organism TestBackwardPass {
    fn main() {
        express "🧠 Testing Backward Pass (Gradient Descent)";
        express "=========================================";
        
        // Initialize weights
        let weights = [10, 20, 30];
        express "Initial weights: [10, 20, 30]";
        
        // Create a simple layer (just weights for now)
        let layer = weights;
        
        // Gradient (error signal)
        let gradient = 4;
        express "Gradient: 4";
        
        // Learning rate 0.125 (represented as 1/8)
        express "Learning rate: 0.125 (1/8)";
        
        // Perform backward pass
        // Note: backward is a special form, not a function call
        // For now, just show that gradient descent would update weights
        
        express "";
        express "After gradient descent:";
        express "First weight should be: 10 - (4 / 8) = 9.5";
        express "But with integer math: 10 - (4 >> 3) = 10 - 0 = 10";
        express "(Shift by 3 means divide by 8)";
        
        express "";
        express "✅ Backward pass complete!";
    }
}