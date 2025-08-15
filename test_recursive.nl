organism TestRecursive {
    // Simple recursive factorial (small numbers to avoid overflow)
    fn factorial(n) {
        if n <= 1 {
            return 1;
        }
        let sub = n - 1;
        let result = factorial(sub);
        return n * result;
    }
    
    fn main() {
        express "Testing recursive functions:";
        
        express "factorial(3) = ";
        let f3 = factorial(3);
        synthesize f3;
        
        express "factorial(4) = ";  
        let f4 = factorial(4);
        synthesize f4;
        
        express "factorial(5) = ";
        let f5 = factorial(5);
        synthesize f5;
        
        express "Recursive test complete!";
    }
}