// Test array access immediately after creation
organism ArrayImmediate {
    fn birth() {
        express "Creating and immediately accessing array [100, 200, 300, 400, 500]"
        
        let arr = [100, 200, 300, 400, 500]
        
        express "Immediate access:"
        express "Element 0:"
        synthesize arr[0]
        
        express "Element 1:"
        synthesize arr[1]
        
        express "Element 2:"
        synthesize arr[2]
        
        express "Element 3:"
        synthesize arr[3]
        
        express "Element 4:"
        synthesize arr[4]
    }
}