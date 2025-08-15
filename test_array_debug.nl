// Debug how arrays work
organism ArrayDebug {
    fn birth() {
        express "Creating array..."
        let arr = [10, 20, 30]
        express "Array created"
        
        express "Accessing array elements:"
        let first = arr[0]  
        synthesize first
        
        let second = arr[1]
        synthesize second
        
        let third = arr[2]
        synthesize third
        
        express "Array debug complete"
    }
}