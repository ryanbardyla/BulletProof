// 📚 TEST: Standard Library Usage

import "std.math" { abs, max, min, square }
import "std.array" { sum, average, range }
import "std.io" { println }

organism StdLibTest {
    fn main() {
        // Test math functions
        let a = -42;
        let b = abs(a);
        synthesize b;  // Should print 42
        
        let x = max(10, 20);
        synthesize x;  // Should print 20
        
        let y = min(10, 20);
        synthesize y;  // Should print 10
        
        let sq = square(7);
        synthesize sq; // Should print 49
        
        // Test array functions
        let nums = range(1, 5);  // [1, 2, 3, 4]
        let total = sum(nums);
        synthesize total; // Should print 10
        
        let avg = average(nums);
        synthesize avg;   // Should print 2.5
        
        // Test IO function
        println(123);     // Should print 123 with newline
    }
}