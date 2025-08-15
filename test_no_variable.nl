// Test save_weights without variable assignment
organism NoVariable {
    fn birth() {
        express "Testing save_weights without variable assignment"
        
        // This should be invalid syntax, but let's see what happens
        save_weights("no_var.weights", [10, 20, 30, 40, 50])
        
        express "Test completed"
    }
}