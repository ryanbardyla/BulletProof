// Test two synthesize calls
organism TwoSynthesize {
    fn birth() {
        let a = 5
        synthesize a
        let b = 3
        synthesize b
    }
}