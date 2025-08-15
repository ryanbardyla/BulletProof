organism TestTypeCheck {
    fn main() {
        express "Testing type checking:";
        
        // This should work - correct types
        let x: int = 42;
        let msg: string = "Hello";
        
        express "Correct types compiled successfully";
        
        // This will generate a warning but still compile
        // (since our type system is currently permissive)
        let wrong: string = 123;  // Type mismatch warning
        
        express "Type checking test complete!";
    }
}