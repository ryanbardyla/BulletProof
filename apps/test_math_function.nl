// Test math function
organism TestMath {
    fn birth() {
        express "Testing math function"
        let result = add(3, 4)
        express "3 + 4 ="
        express result
        express "Done"
    }
}

fn add(a, b) {
    return a + b
}