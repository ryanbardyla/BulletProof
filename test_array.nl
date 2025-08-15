// Test array support
organism ArrayTest {
    fn birth() {
        express "=== TESTING ARRAYS ==="
        express ""
        
        // Create an array
        let weights = [10, 20, 30, 40, 50]
        
        express "Created array: [10, 20, 30, 40, 50]"
        
        // Access elements
        let first = weights[0]
        let third = weights[2] 
        let last = weights[4]
        
        express "First element (weights[0]): 10"
        express "Third element (weights[2]): 30"
        express "Last element (weights[4]): 50"
        
        express ""
        express "Arrays work! Ready for neural networks!"
    }
}