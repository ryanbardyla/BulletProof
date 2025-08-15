// 🧬 MODULE SYSTEM TEST
// Test import statements and module definitions

// Define a module
module MathUtils {
    export fn add_numbers(a, b) {
        return a + b;
    }
    
    export fn multiply_numbers(x, y) {
        return x * y; 
    }
    
    // This function is not exported
    fn private_helper() {
        return 42;
    }
}

// Import specific functions from the module
import "MathUtils" { add_numbers, multiply_numbers };

organism ModuleTest {
    fn main() {
        express "=== MODULE SYSTEM TEST ===";
        
        // Test imported functions
        express "Testing imported add_numbers:";
        let result1 = add_numbers(5, 3);
        synthesize result1; // Should output 8
        
        express "Testing imported multiply_numbers:";
        let result2 = multiply_numbers(4, 7);
        synthesize result2; // Should output 28
        
        // Test using the results in computations
        express "Using imported functions in computation:";
        let combined = add_numbers(result1, result2);
        synthesize combined; // Should output 36 (8 + 28)
        
        express "=== MODULE SYSTEM TEST COMPLETE ===";
    }
}