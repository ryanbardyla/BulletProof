organism TestParams {
    fn subtract(a, b) {
        return a - b;
    }
    
    fn main() {
        express "Testing parameter order:";
        
        // If params are in correct order: 10 - 3 = 7
        // If params are reversed: 3 - 10 = -7
        let result = subtract(10, 3);
        express "10 - 3 should equal 7:";
        synthesize result;
        
        // Another test
        let result2 = subtract(100, 25);
        express "100 - 25 should equal 75:";
        synthesize result2;
        
        express "Parameter test complete!";
    }
}