organism TestParamsDebug {
    fn identity(x) {
        return x;
    }
    
    fn main() {
        express "Testing single parameter:";
        
        let result = identity(42);
        express "Should be 42:";
        synthesize result;
        
        express "Done!";
    }
}