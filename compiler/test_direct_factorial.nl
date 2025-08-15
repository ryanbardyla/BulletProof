organism DirectFactorial {
    fn main() {
        express "Testing direct optimized factorial:";
        
        // Test small values first
        express "factorial(0) should be 1:";
        let f0 = factorial(0);
        synthesize f0;
        
        express "factorial(1) should be 1:";
        let f1 = factorial(1);
        synthesize f1;
        
        express "factorial(3) should be 6:";
        let f3 = factorial(3);
        synthesize f3;
        
        express "factorial(4) should be 24:";
        let f4 = factorial(4);
        synthesize f4;
    }
}