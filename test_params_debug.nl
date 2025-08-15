organism TestParamsDebug {
    fn test_params(a, b) {
        express "a = ";
        synthesize a;
        express "b = ";
        synthesize b;
        return a + b;
    }
    
    fn main() {
        express "Testing parameters with debug:";
        let result = test_params(10, 20);
        express "Result = ";
        synthesize result;
    }
}