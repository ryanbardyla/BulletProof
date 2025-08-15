organism TestRecursive {
    fn factorial(n) {
        if n <= 1 {
            return 1;
        }
        let n_minus_1 = n - 1;
        let result = factorial(n_minus_1);
        return n * result;
    }
    
    fn main() {
        express "Testing recursive factorial:";
        
        let result = factorial(5);
        express "5! should be 120:";
        synthesize result;
        
        express "Recursion test complete!";
    }
}