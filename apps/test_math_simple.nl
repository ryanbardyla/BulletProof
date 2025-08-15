// Simple math test
organism MathTest {
    fn birth() {
        express "Testing simple math"
        
        let a = 3
        let b = 4
        let sum = a + b
        
        express "3 + 4 ="
        express sum
        
        let product = a * b
        express "3 * 4 ="
        express product
        
        express "Math works!"
    }
}