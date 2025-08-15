// 🔧 TEST: Module system segfault reproduction

module MathUtils {
    export fn add(a, b) {
        return a + b;
    }
    
    export fn multiply(x, y) {
        return x * y;
    }
}

import "MathUtils" { add, multiply }

organism ModuleTest {
    fn main() {
        let result1 = add(5, 3);
        synthesize result1;
        
        let result2 = multiply(4, 6);
        synthesize result2;
    }
}