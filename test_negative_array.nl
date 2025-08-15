organism TestNegativeArray {
    fn main() {
        express "Testing negative numbers in arrays...";
        
        // Test 1: Simple negative number
        let x = -5;
        express "x = -5:";
        synthesize x;
        
        // Test 2: Array with negative
        let arr = [1, -2, 3];
        express "Created array [1, -2, 3]";
        
        express "Done!";
    }
}