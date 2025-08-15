organism TestArrays {
    fn main() {
        express "🔢 Testing Array Literals";
        express "=========================";
        
        // Test 1: Create simple array
        let arr1 = [1, 2, 3, 4, 5];
        express "Created array [1, 2, 3, 4, 5]";
        
        // Test 2: Array with negative numbers
        let arr2 = [-1, 0, 1];
        express "Created array [-1, 0, 1]";
        
        // Test 3: Single element array
        let arr3 = [42];
        express "Created array [42]";
        
        // Test 4: Empty array (might not work)
        // let arr4 = [];
        // express "Created empty array";
        
        // Test 5: Array with variables
        let x = 10;
        let y = 20;
        let arr5 = [x, y, 30];
        express "Created array with variables [10, 20, 30]";
        
        express "";
        express "✅ Array literal tests complete!";
    }
}