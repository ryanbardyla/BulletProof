// Test to understand memory layout  
organism MemoryDebug {
    fn birth() {
        express "Creating array [1, 2, 3, 4, 5]..."
        let arr = [1, 2, 3, 4, 5]
        express "Array created"
        
        express "Testing save_weights function..."
        save_weights("debug.weights", arr)
        express "Save completed"
    }
}