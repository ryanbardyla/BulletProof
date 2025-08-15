organism TestParams {
    // Test function with parameters
    fn add(a, b) {
        return a + b;
    }
    
    fn multiply(x, y) {
        return x * y;
    }
    
    fn calculate(a, b, c) {
        let sum = add(a, b);
        let product = multiply(sum, c);
        return product;
    }
    
    fn main() {
        express "Testing function parameters:";
        
        // Test add with parameters
        let result1 = add(10, 20);
        express "add(10, 20) = ";
        synthesize result1;
        
        // Test multiply
        let result2 = multiply(5, 6);
        express "multiply(5, 6) = ";
        synthesize result2;
        
        // Test nested calls with multiple parameters
        let result3 = calculate(2, 3, 4);
        express "calculate(2, 3, 4) = (2+3)*4 = ";
        synthesize result3;
        
        express "Parameters test complete!";
    }
}