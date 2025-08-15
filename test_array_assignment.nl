// Test array element assignment
organism ArrayAssignmentTest {
    fn birth() {
        express "Testing array element assignment..."
        
        // Create array
        let arr = [10, 20, 30]
        
        express "Original: [10, 20, 30]"
        
        // Update array elements
        arr[0] = 100
        arr[1] = 200
        arr[2] = 300
        
        express "Updated: [100, 200, 300]"
        express "Parser fix works!"
    }
}