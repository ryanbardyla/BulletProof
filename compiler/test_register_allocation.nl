// 🎯 TEST: Register Allocation Optimization
// This test creates multiple variables to test register allocation

organism RegisterTest {
    fn calculate(a: int, b: int) {
        // Create several local variables
        let x = a + 10;
        let y = b * 2;
        let z = x + y;
        
        // Use variables multiple times (to test live ranges)
        let temp1 = x * x;
        let temp2 = y * y;
        let result = temp1 + temp2 + z;
        
        // Final computation
        let final = result * 2;
        synthesize final;
    }
    
    fn main() {
        // Test with some values
        calculate(5, 7);
        calculate(10, 20);
        calculate(3, 4);
    }
}