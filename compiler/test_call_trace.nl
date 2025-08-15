// 📞 TEST: Function Call Trace
// Simple test to verify function tracking

organism CallTrace {
    fn add(a, b) {
        return a + b;
    }
    
    fn multiply(x, y) {
        let result = 0;
        for i in 0 to y {
            result = add(result, x);
        }
        return result;
    }
    
    fn factorial(n) {
        if n <= 1 {
            return 1;
        }
        return multiply(n, factorial(n - 1));
    }
    
    fn main() {
        let result = factorial(5);
        synthesize result;  // Should print 120
    }
}