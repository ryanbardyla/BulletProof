// Test save_weights and load_weights functions with integers
organism FileIOSimple {
    fn birth() {
        express "🔧 Testing File I/O Functions"
        express ""
        
        express "📊 Creating test array..."
        let weights = [1, 2, 3, 4, 5]
        express "Array created with 5 elements"
        
        express ""
        express "💾 Saving weights to file..."
        save_weights("test.weights", weights)
        express "Weights saved to test.weights"
        
        express ""
        express "📂 Loading weights from file..."
        let loaded_weights = load_weights("test.weights")
        express "Weights loaded successfully"
        
        express ""
        express "✅ File I/O test completed!"
    }
}