// Test number variables
organism TestNumberVars {
    fn birth() {
        express "Testing number variables:"
        let a = 5
        let b = 10
        let sum = a + b
        
        express "Numbers:"
        express a
        express b
        express sum
        express "Math works!"
    }
}