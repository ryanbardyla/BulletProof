// Test array content without save_weights
organism ArrayContent {
    fn birth() {
        express "Creating array [10, 20, 30, 40, 50]"
        let arr = [10, 20, 30, 40, 50]
        
        express "Array element 0:"
        synthesize arr[0]
        
        express "Array element 1:"
        synthesize arr[1]
        
        express "Array element 2:" 
        synthesize arr[2]
        
        express "Array element 3:"
        synthesize arr[3]
        
        express "Array element 4:"
        synthesize arr[4]
    }
}