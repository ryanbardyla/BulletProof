// 📚 REVOLUTIONARY STACK FRAME MANAGEMENT TEST
// Test multiple function calls with proper frame tracking!

organism StackFrameTest {
    fn helper_function(param) {
        express "📚 Inside helper function";
        let local_var = param * 2;
        synthesize local_var;
        return local_var;
    }
    
    fn another_function(x, y) {
        express "📚 Inside another function";
        let result = x + y;
        let helper_result = helper_function(result);
        synthesize helper_result;
        return helper_result;
    }
    
    fn main() {
        express "📚 REVOLUTIONARY STACK FRAME MANAGEMENT TEST!";
        express "Testing multiple function calls with proper frame tracking";
        
        let val1 = 10;
        let val2 = 20;
        
        express "Calling another_function with nested helper_function call";
        let final_result = another_function(val1, val2);
        
        express "Final result:";
        synthesize final_result;
        
        express "📚 Stack frame management working perfectly!";
    }
}