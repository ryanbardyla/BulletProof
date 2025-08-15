// Simple number test
organism SimpleTest {
    fn birth() {
        express "Testing simple numbers..."
        
        // Test simple integer
        let a = 5
        synthesize a
        
        // Test another number
        let b = 42
        synthesize b
        
        express "Done!"
    }
}