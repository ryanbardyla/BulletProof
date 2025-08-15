organism TestSimpleFunc {
    fn add(a, b) {
        return a + b;
    }
    
    fn main() {
        express "Testing simple function call:";
        
        let result = add(5, 3);
        express "5 + 3 =";
        synthesize result;
        
        express "Function call test complete!";
    }
}