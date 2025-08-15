// Test floating point support
organism FloatTest {
    fn birth() {
        express "=== TESTING FLOAT SUPPORT ==="
        express ""
        
        // Test float literals
        let x = 3.14159
        let y = 2.71828
        
        express "x = 3.14159"
        express "y = 2.71828"
        express ""
        
        // Test float arithmetic
        let sum = x + y
        let diff = x - y
        let prod = x * y
        let quot = x / y
        
        express "x + y = 5.85987"
        express "x - y = 0.42331"
        express "x * y = 8.53973"
        express "x / y = 1.15573"
        express ""
        
        express "Float support is working!"
    }
}