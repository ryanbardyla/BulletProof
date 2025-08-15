organism StringVariableConcatTest {
    fn main() {
        express "=== STRING VARIABLE CONCATENATION TEST ===";
        
        // Test 1: Simple string variable concatenation
        express "Test 1: Basic variable concatenation";
        let str1 = "Hello";
        let str2 = "World";
        let result = str1 + str2;
        express "Result should be HelloWorld:";
        synthesize result;
        
        // Test 2: Multiple concatenations
        express "Test 2: Multiple concatenations";
        let a = "A";
        let b = "B"; 
        let c = "C";
        let abc = a + b + c;
        express "Result should be ABC:";
        synthesize abc;
        
        // Test 3: Mixed literals and variables
        express "Test 3: Mixed literals and variables";
        let name = "Alice";
        let greeting = "Hello " + name + "!";
        express "Result should be Hello Alice!:";
        synthesize greeting;
        
        // Test 4: Reuse variables after concatenation
        express "Test 4: Reuse variables after concatenation";
        express "str1 after concatenation:";
        synthesize str1;
        express "str2 after concatenation:";
        synthesize str2;
        
        express "=== END TEST ===";
    }
}