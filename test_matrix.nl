// Matrix operations for neural networks
organism MatrixTest {
    fn birth() {
        express "=== MATRIX OPERATIONS ==="
        
        // 2D arrays for matrices
        let weights = [[1, 2], [3, 4]]
        let inputs = [5, 6]
        
        // Dot product (manual for now)
        let output1 = weights[0][0] * inputs[0] + weights[0][1] * inputs[1]
        let output2 = weights[1][0] * inputs[0] + weights[1][1] * inputs[1]
        
        express "Matrix multiply done!"
        express "Ready for neural networks!"
    }
}