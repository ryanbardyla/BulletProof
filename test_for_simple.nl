// Simple for loop test
organism SimpleForTest {
    fn birth() {
        express "Testing for loop:"
        
        // Loop from 0 to 5 (hardcoded to 10 in parser for now)
        for i = 0 {
            express "Count"
        }
        
        express "Done!"
    }
}