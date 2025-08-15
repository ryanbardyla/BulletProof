// 🔍 TEST: Stack Trace Generation
// This program demonstrates stack trace functionality

organism StackTraceTest {
    fn deep_function_3() {
        // This will cause an error to show the stack trace
        let arr = [1, 2, 3];
        let bad_index = 10;
        
        // This should trigger a bounds check error
        synthesize arr[bad_index];
    }
    
    fn deep_function_2() {
        deep_function_3();
    }
    
    fn deep_function_1() {
        deep_function_2();
    }
    
    fn main() {
        synthesize "Starting stack trace test...";
        deep_function_1();
        synthesize "This shouldn't print";
    }
}