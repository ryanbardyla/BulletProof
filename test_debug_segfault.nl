organism TestDebugSegfault {
    fn main() {
        express "🔍 Debugging segfault...";
        
        // Test 1: Direct synthesize works
        express "Test 1: Direct synthesize...";
        synthesize 42;
        express "✓ Direct synthesize works";
        
        // Test 2: Variable assignment and synthesize
        express "Test 2: Variable synthesize...";
        let x = 99;
        synthesize x;
        express "✓ Variable synthesize works";
        
        // Test 3: HashMap creation
        express "Test 3: HashMap creation...";
        let h = {1: 100};
        express "✓ HashMap creation works";
        
        // Test 4: HashMap get with fixed return (999)
        express "Test 4: HashMap get...";
        let v = get(h, 1);
        express "✓ HashMap get works";
        
        // Test 5: The problem - synthesize the result
        express "Test 5: Synthesize HashMap value...";
        synthesize v;
        express "✓ All tests passed!";
    }
}