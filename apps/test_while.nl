// Test while loop functionality
organism TestWhile {
    fn birth() {
        express "Testing while loops"
        
        let count = 1
        express "Counting to 5:"
        
        while count <= 5 {
            express count
            count = count + 1
        }
        
        express "Done counting!"
    }
}