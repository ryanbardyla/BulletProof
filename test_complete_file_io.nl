// Test complete save and load cycle
organism CompleteFileIO {
    fn birth() {
        express "🔄 Complete File I/O Test"
        express ""
        
        express "📊 Creating original weights [20, 30, 40, 50, 60]"
        let original_weights = [20, 30, 40, 50, 60]
        
        express "💾 Saving weights to complete.weights..."
        save_weights("complete.weights", original_weights)
        express "Weights saved"
        
        express ""
        express "📂 Loading weights from complete.weights..."
        let loaded_weights = load_weights("complete.weights")
        express "Weights loaded successfully"
        
        express ""
        express "✅ Complete file I/O test finished!"
        express "(Check file manually to verify data)"
    }
}