organism TestLambda {
    fn main() {
        express "Testing lambda expressions:";
        
        // Simple lambda that doubles a number
        let double = |x| x * 2;
        
        let result = double(5);
        express "double(5) = ";
        synthesize result;
        
        // Lambda with multiple parameters
        let add = |a, b| a + b;
        
        let sum = add(3, 7);
        express "add(3, 7) = ";
        synthesize sum;
        
        express "Lambda test complete!";
    }
}