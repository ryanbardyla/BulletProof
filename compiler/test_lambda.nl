organism LambdaTest {
    fn main() {
        express "Testing lambda expressions:";
        
        // Simple lambda with no parameters
        let simple = lambda() => {
            return 42;
        };
        express "Simple lambda (no params):";
        synthesize simple;
        
        // Lambda with one parameter  
        let double = lambda(x) => {
            return x * 2;
        };
        express "Lambda with parameter (should be ID):";
        synthesize double;
        
        // Lambda with multiple parameters
        let add = lambda(a, b) => {
            return a + b;
        };
        express "Lambda with multiple params (should be ID):";
        synthesize add;
        
        express "Lambda test complete!";
    }
}