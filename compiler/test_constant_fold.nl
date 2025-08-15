organism TestConstantFold {
    fn main() {
        express "Testing constant folding optimization:";
        
        // These should be folded at compile time
        let x = 2 + 3 * 4;  // Should be folded to 14
        express "2 + 3 * 4 =";
        synthesize x;
        
        let y = 100 - 50 / 2;  // Should be folded to 75
        express "100 - 50 / 2 =";
        synthesize y;
        
        let z = (10 + 5) * 2;  // Should be folded to 30
        express "(10 + 5) * 2 =";
        synthesize z;
        
        // String concatenation folding
        let message = "Hello, " + "World!";
        express message;
        
        // Mixed with variables (partial folding)
        let a = 5;
        let b = a + 10 * 2;  // 10 * 2 should be folded to 20
        express "a + 10 * 2 where a=5:";
        synthesize b;
        
        express "Constant folding test complete!";
    }
}