// Test float operations in NeuronLang
organism FloatTest {
    fn birth() {
        express "Testing float operations..."
        
        // Integer operations
        let a = 10
        let b = 3
        let int_result = a + b
        express "10 + 3 ="
        synthesize int_result
        
        // Float operations
        let x = 3.14159
        let y = 2.71828
        let float_sum = x + y
        express "3.14159 + 2.71828 ="
        synthesize float_sum
        
        // Mixed operations
        let z = 100.5
        let w = 50
        let mixed = z * 2.0
        express "100.5 * 2.0 ="
        synthesize mixed
        
        // Division with floats
        let div_result = 22.0 / 7.0
        express "22.0 / 7.0 ="
        synthesize div_result
        
        express "Float support working!"
    }
}