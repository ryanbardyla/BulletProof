organism TestMalloc {
    fn main() {
        express "Testing dynamic memory allocation:";
        
        // Allocate 100 bytes
        let size = 100;
        express "Allocating 100 bytes...";
        let ptr = malloc(size);
        
        // Check if allocation succeeded (non-zero pointer)
        if ptr {
            express "Memory allocated successfully!";
            
            // In a real program, we'd use the memory here
            // For now, just free it
            
            express "Freeing memory...";
            let result = free(ptr, size);
            
            if result == 0 {
                express "Memory freed successfully!";
            } else {
                express "Failed to free memory";
            }
        } else {
            express "Memory allocation failed!";
        }
        
        express "Dynamic allocation test complete!";
    }
}