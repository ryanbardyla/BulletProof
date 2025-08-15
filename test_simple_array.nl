// Test simple array values
organism SimpleArray {
    fn birth() {
        express "Creating array [1, 2, 3, 4, 5]"
        let arr = [1, 2, 3, 4, 5]
        
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