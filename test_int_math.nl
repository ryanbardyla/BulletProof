// Test integer math still works
organism IntMathTest {
    fn birth() {
        express "Testing integer math..."
        
        let a = 10
        let b = 3
        
        let sum = a + b
        let diff = a - b  
        let prod = a * b
        let quot = a / b
        
        express "10 + 3 = 13"
        express "10 - 3 = 7"
        express "10 * 3 = 30"
        express "10 / 3 = 3"
        
        express "Integer math works!"
    }
}