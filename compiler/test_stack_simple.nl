// 📚 TEST: Simple Stack Trace
// Tests basic function call tracking

organism StackTest {
    fn level_3() {
        synthesize "Level 3";
    }
    
    fn level_2() {
        synthesize "Level 2";
        level_3();
    }
    
    fn level_1() {
        synthesize "Level 1";
        level_2();
    }
    
    fn main() {
        synthesize "Starting stack test";
        level_1();
        synthesize "Stack test complete";
    }
}