organism TestFunctions {
    // Define a simple function that adds two numbers
    fn add(a, b) {
        return a + b;
    }
    
    // Define a function that squares a number
    fn square(x) {
        return x * x;
    }
    
    // Define a function that calculates factorial
    fn factorial(n) {
        if n <= 1 {
            return 1;
        }
        return n * factorial(n - 1);
    }
    
    // Main function
    fn main() {
        express "🧠 Testing User-Defined Functions";
        express "==================================";
        
        // Test add function
        express "";
        express "1. Testing add(3, 5):";
        let sum = add(3, 5);
        synthesize sum;
        
        // Test square function
        express "";
        express "2. Testing square(7):";
        let sq = square(7);
        synthesize sq;
        
        // Test nested function calls
        express "";
        express "3. Testing add(square(3), square(4)):";
        let result = add(square(3), square(4));
        express "Should be 9 + 16 = 25:";
        synthesize result;
        
        // Test factorial (recursive)
        express "";
        express "4. Testing factorial(5):";
        let fact = factorial(5);
        express "Should be 120:";
        synthesize fact;
        
        express "";
        express "✅ Function tests complete!";
    }
}