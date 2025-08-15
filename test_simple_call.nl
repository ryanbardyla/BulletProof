organism TestSimpleCall {
    fn add_numbers(a, b) {
        return a + b;
    }
    
    fn main() {
        express "Testing simple function call:";
        let result = add_numbers(5, 3);
        express "5 + 3 = ";
        synthesize result;
    }
}