// Test comparison operators including <= and >=
organism TestComparisons {
    fn birth() {
        express "Testing <= and >= operators"
        
        let a = 5
        let b = 10
        
        if a <= b {
            express "5 <= 10 works!"
        }
        
        if b >= a {
            express "10 >= 5 works!"
        }
        
        if a <= a {
            express "5 <= 5 works!"
        }
        
        if a >= a {
            express "5 >= 5 works!"
        }
        
        express "All comparison tests passed!"
    }
}