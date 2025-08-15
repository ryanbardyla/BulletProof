organism TestSaveWeights {
    fn main() {
        express "Testing save_weights to verify file operations...";
        
        let weights = [10, 20, 30, 40, 50];
        let result = save_weights("test_weights.bin", weights);
        
        express "Save weights complete!";
    }
}