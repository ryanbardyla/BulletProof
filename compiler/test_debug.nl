// 🐛 TEST: Debug Symbol Generation
// Test program for verifying debug information

organism DebugTest {
    fn calculate(x, y) {
        let sum = x + y;
        let diff = x - y;
        let product = sum * diff;
        return product;
    }
    
    fn main() {
        let a = 10;
        let b = 5;
        let result = calculate(a, b);
        synthesize result;
    }
}