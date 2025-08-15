organism TestArrayMethods {
    fn main() {
        express "🧠 Testing Array Methods";
        express "========================";
        
        // Test array length
        express "";
        express "1. Testing len() on array:";
        let arr = [10, 20, 30];
        let arr_len = len(arr);
        express "Array [10, 20, 30] length:";
        synthesize arr_len;
        
        // Test string length
        express "";
        express "2. Testing len() on string:";
        let str = "Hello";
        let str_len = len(str);
        express "String 'Hello' length:";
        synthesize str_len;
        
        // Test array push
        express "";
        express "3. Testing push():";
        let nums = [1, 2, 3];
        express "Original array: [1, 2, 3]";
        let nums2 = push(nums, 4);
        express "After push(nums, 4):";
        let new_len = len(nums2);
        express "New length:";
        synthesize new_len;
        
        // Test array pop
        express "";
        express "4. Testing pop():";
        let stack = [100, 200, 300];
        express "Original array: [100, 200, 300]";
        let popped = pop(stack);
        express "Popped value:";
        synthesize popped;
        
        // Test push and pop together
        express "";
        express "5. Testing push/pop sequence:";
        let data = [5];
        let d1 = push(data, 10);
        let d2 = push(d1, 15);
        express "After pushing 10 and 15 to [5]:";
        let final_len = len(d2);
        express "Length:";
        synthesize final_len;
        
        let val1 = pop(d2);
        express "First pop:";
        synthesize val1;
        
        express "";
        express "✅ Array methods complete!";
    }
}