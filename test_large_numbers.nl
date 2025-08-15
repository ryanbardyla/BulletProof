// Test save_weights with larger numbers that synthesize can handle
organism LargeNumbers {
    fn birth() {
        express "Creating array [10, 20, 30, 40, 50]"
        let arr = [10, 20, 30, 40, 50]
        
        express "Saving to large.weights..."
        save_weights("large.weights", arr)
        express "Save completed"
    }
}