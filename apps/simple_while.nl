// Simple while loop test
organism SimpleWhile {
    fn birth() {
        express "Starting"
        let x = 1
        
        while x <= 3 {
            express "Loop iteration"
            x = x + 1
        }
        
        express "Finished"
    }
}