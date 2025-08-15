// Test for loops!
organism ForLoopTest {
    fn birth() {
        express "=== FOR LOOP TEST ==="
        express ""
        
        // Classic for loop
        express "Counting from 0 to 5:"
        for i = 0 {
            express "Iteration"
        }
        
        express ""
        express "Iterating array with for-in:"
        
        // Array iteration
        let weights = [10, 20, 30, 40, 50]
        for w in weights {
            express "Processing weight"
        }
        
        express ""
        express "For loops work!"
    }
}