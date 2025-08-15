// Debug while loop
organism DebugWhile {
    fn birth() {
        express "Starting while loop"
        let i = 1
        
        while i <= 3 {
            express "Loop iteration"
            i = i + 1
        }
        
        express "While loop done"
    }
}