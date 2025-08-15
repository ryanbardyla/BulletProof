// 🔍 TEST: Peephole Optimizations
// This test contains patterns that can be optimized

organism PeepholeTest {
    fn redundant_ops() {
        // These operations create push/pop patterns
        let x = 10;
        let y = x;  // Often creates push x; pop y
        let z = y;  // Another push/pop
        
        // Arithmetic with zero (should be eliminated)
        let a = x + 0;
        let b = y - 0;
        
        synthesize z + a + b;
    }
    
    fn simple_math() {
        // Simple operations that might benefit from optimization
        let val = 5;
        let result = val + val + val;  // Multiple additions
        
        synthesize result;
    }
    
    fn main() {
        redundant_ops();
        simple_math();
    }
}