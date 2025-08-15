organism TestNoParams {
    fn simple() {
        express "In simple function!";
        return 42;
    }
    
    fn main() {
        express "Testing no-param function:";
        
        let result = simple();
        express "Result:";
        synthesize result;
        
        express "Done!";
    }
}