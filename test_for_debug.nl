// Debug for loop 
organism ForDebug {
    fn birth() {
        express "Before loop"
        
        // Loop should go from 0 to 10 (hardcoded)
        for i = 1 {
            express "In loop!"
        }
        
        express "After loop"
    }
}