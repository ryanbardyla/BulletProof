// Test if element 0 corruption is fixed
organism Element0Fix {
    fn birth() {
        express "Testing element 0 corruption fix"
        express ""
        
        express "Creating array [100, 200, 300, 400, 500]"
        let test_array = [100, 200, 300, 400, 500] 
        
        express "Saving to test_fix.weights"
        save_weights("test_fix.weights", test_array)
        
        express "Test completed - checking file manually"
    }
}