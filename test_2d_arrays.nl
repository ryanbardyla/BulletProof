organism Test2DArrays {
    fn main() {
        express "🧠 Testing 2D Arrays (Matrices)";
        express "================================";
        
        // Create a 2x3 matrix
        express "";
        express "1. Creating 2x3 matrix:";
        let matrix = [[1, 2, 3], [4, 5, 6]];
        express "Matrix created: [[1,2,3], [4,5,6]]";
        
        // Get element from matrix
        express "";
        express "2. Getting elements:";
        let elem00 = matrix_get(matrix, 0, 0);
        express "matrix[0][0] =";
        synthesize elem00;
        
        let elem01 = matrix_get(matrix, 0, 1);
        express "matrix[0][1] =";
        synthesize elem01;
        
        let elem10 = matrix_get(matrix, 1, 0);
        express "matrix[1][0] =";
        synthesize elem10;
        
        let elem11 = matrix_get(matrix, 1, 1);
        express "matrix[1][1] =";
        synthesize elem11;
        
        // Set element in matrix
        express "";
        express "3. Setting element:";
        let matrix2 = matrix_set(matrix, 0, 1, 99);
        express "After setting matrix[0][1] = 99:";
        let new_elem = matrix_get(matrix2, 0, 1);
        synthesize new_elem;
        
        // Test matrix multiplication (simple 2x2)
        express "";
        express "4. Matrix multiplication (2x2):";
        let A = [[1, 2], [3, 4]];
        let B = [[5, 6], [7, 8]];
        express "A = [[1,2], [3,4]]";
        express "B = [[5,6], [7,8]]";
        express "A @ B should give [[19,22], [43,50]]";
        
        let C = matrix_mul(A, B);
        express "Result[0][0] = 1*5 + 2*7 = 19:";
        let c00 = matrix_get(C, 0, 0);
        synthesize c00;
        
        express "";
        express "✅ 2D array operations complete!";
    }
}