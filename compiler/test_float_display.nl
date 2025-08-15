// 🔢 FLOAT DISPLAY TEST
// Test how floats are currently displayed in synthesize

organism FloatDisplayTest {
    fn main() {
        express "Testing float display in synthesize:";
        
        // Test various float values
        let pi = 3.14159;
        let small_float = 0.5;
        let negative_float = -2.75;
        let large_float = 123.456;
        
        express "Pi value:";
        synthesize pi;
        
        express "Small float:";
        synthesize small_float;
        
        express "Negative float:";
        synthesize negative_float;
        
        express "Large float:";
        synthesize large_float;
        
        // Also test integer for comparison
        let integer = 42;
        express "Integer for comparison:";
        synthesize integer;
    }
}