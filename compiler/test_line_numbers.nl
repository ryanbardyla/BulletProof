// 📍 TEST: Line Number Tracking Success
// This file should compile successfully

organism LineNumberTest {
    fn calculate(a, b) {
        let sum = a + b;
        let diff = a - b;
        return sum + diff;
    }
    
    fn main() {
        let x = 10;
        let y = 5;
        
        let result = calculate(x, y);
        
        synthesize result;
    }
}