organism TestWeightInit {
    fn main() {
        express "🧠 Testing Weight Initialization Strategies";
        express "==========================================";
        
        // Test zeros initialization
        express "";
        express "1. Zeros initialization (size=3):";
        let zero_weights = zeros(3);
        express "Created zeros array";
        
        // Test ones initialization
        express "";
        express "2. Ones initialization (size=3):";
        let one_weights = ones(3);
        express "Created ones array";
        
        // Test Xavier initialization
        express "";
        express "3. Xavier initialization (fan_in=4, fan_out=2):";
        let xavier = xavier_init(4, 2);
        express "Xavier weight:";
        synthesize xavier;
        
        // Test He initialization
        express "";
        express "4. He initialization (fan_in=4):";
        let he = he_init(4);
        express "He weight:";
        synthesize he;
        
        // Test random normal
        express "";
        express "5. Random normal (Gaussian):";
        let gauss1 = randn();
        let gauss2 = randn();
        let gauss3 = randn();
        express "Three Gaussian samples:";
        synthesize gauss1;
        synthesize gauss2;
        synthesize gauss3;
        
        express "";
        express "✅ Weight initialization complete!";
    }
}