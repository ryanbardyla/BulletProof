// 🎯 TEST: Line Number Tracking in Errors
// This file contains intentional errors to test error reporting

organism ErrorTest {
    fn main() {
        let x = 10;
        let y = 20;
        
        // Intentional syntax error on line 9
        let z = x ++ y;  // Invalid operator
        
        // Another error on line 12
        synthesize z;
        
        // Missing closing brace for function
    // Missing closing brace for organism