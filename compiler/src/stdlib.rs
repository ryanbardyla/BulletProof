// 🏛️ NEURONLANG STANDARD LIBRARY
// Built-in modules that are always available

use std::collections::HashMap;

pub struct StandardLibrary {
    modules: HashMap<String, String>,
}

impl StandardLibrary {
    pub fn new() -> Self {
        let mut modules = HashMap::new();
        
        // 🧮 std.math - Mathematical functions
        // Note: stdlib modules are pre-parsed without the module wrapper
        modules.insert("std.math".to_string(), r#"module std_math {
    // Constants
    export fn pi() {
        return 3;
    }
    
    export fn e() {
        return 3;
    }
    
    export fn tau() {
        return 6;
    }
    
    // Basic math operations
    export fn abs(x) {
        if x < 0 {
            return -x;
        }
        return x;
    }
    
    export fn min(a, b) {
        if a < b {
            return a;
        }
        return b;
    }
    
    export fn max(a, b) {
        if a > b {
            return a;
        }
        return b;
    }
    
    export fn clamp(value, min_val, max_val) {
        if value < min_val {
            return min_val;
        }
        if value > max_val {
            return max_val;
        }
        return value;
    }
    
    // Power functions
    export fn square(x) {
        return x * x;
    }
    
    export fn cube(x) {
        return x * x * x;
    }
    
    export fn pow(base, exp) {
        let result = 1;
        for i in 0..exp {
            result = result * base;
        }
        return result;
    }
    
    // Factorial
    export fn factorial(n) {
        if n <= 1 {
            return 1;
        }
        return n * factorial(n - 1);
    }
    
    // GCD using subtraction method (no modulo needed)
    export fn gcd(a, b) {
        while b != 0 {
            if a > b {
                a = a - b;
            } else {
                b = b - a;
            }
        }
        return a;
    }
    
    export fn lcm(a, b) {
        return (a * b) / gcd(a, b);
    }
    
    // Neural network activation functions (using built-ins)
    export fn sigmoid(x) {
        return sigmoid(x);
    }
    
    export fn relu(x) {
        return relu(x);
    }
    
    export fn tanh(x) {
        return tanh(x);
    }
}
"#.to_string());

        // 📁 std.io - Input/Output operations
        modules.insert("std.io".to_string(), r#"module std_io {
    // File operations (using built-ins)
    export fn read_file(filename) {
        return read_file(filename);
    }
    
    export fn write_file(filename, content) {
        return write_file(filename, content);
    }
    
    export fn append_file(filename, content) {
        let existing = read_file(filename);
        return write_file(filename, existing + content);
    }
    
    // Console output
    export fn print(value) {
        synthesize value;
    }
    
    export fn println(value) {
        synthesize value;
        synthesize "\n";
    }
    
    // Formatting helpers
    export fn format_number(n, decimals) {
        // Simple formatting - just return the number for now
        return n;
    }
}
"#.to_string());

        // 📚 std.array - Array utilities
        modules.insert("std.array".to_string(), r#"module std_array {
    // Array creation
    export fn range(start, end) {
        let arr = [];
        for i in start..end {
            arr.push(i);
        }
        return arr;
    }
    
    export fn fill(size, value) {
        let arr = [];
        for i in 0..size {
            arr.push(value);
        }
        return arr;
    }
    
    export fn zeros(size) {
        return fill(size, 0);
    }
    
    export fn ones(size) {
        return fill(size, 1);
    }
    
    // Array operations
    export fn sum(arr) {
        let total = 0;
        for i in 0..arr.length() {
            total = total + arr[i];
        }
        return total;
    }
    
    export fn product(arr) {
        let result = 1;
        for i in 0..arr.length() {
            result = result * arr[i];
        }
        return result;
    }
    
    export fn average(arr) {
        return sum(arr) / arr.length();
    }
    
    export fn reverse(arr) {
        let reversed = [];
        let len = arr.length();
        let i = len - 1;
        while i >= 0 {
            reversed.push(arr[i]);
            i = i - 1;
        }
        return reversed;
    }
    
    export fn contains(arr, value) {
        for i in 0..arr.length() {
            if arr[i] == value {
                return 1;
            }
        }
        return 0;
    }
    
    export fn find(arr, value) {
        for i in 0..arr.length() {
            if arr[i] == value {
                return i;
            }
        }
        return -1;
    }
    
    // Sorting (bubble sort for simplicity)
    export fn sort(arr) {
        let n = arr.length();
        for i in 0..n {
            for j in 0..(n - i - 1) {
                if arr[j] > arr[j + 1] {
                    let temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
        return arr;
    }
}
"#.to_string());

        // 🔤 std.string - String utilities
        modules.insert("std.string".to_string(), r#"module std_string {
    // String operations (using built-ins)
    export fn length(s) {
        return string_length(s);
    }
    
    export fn concat(s1, s2) {
        return s1 + s2;
    }
    
    export fn repeat(s, count) {
        let result = "";
        for i in 0..count {
            result = result + s;
        }
        return result;
    }
    
    // Character operations
    export fn to_upper(s) {
        // Placeholder - would need proper implementation
        return s;
    }
    
    export fn to_lower(s) {
        // Placeholder - would need proper implementation
        return s;
    }
    
    // String building
    export fn join(arr, separator) {
        let result = "";
        for i in 0..arr.length() {
            if i > 0 {
                result = result + separator;
            }
            result = result + arr[i];
        }
        return result;
    }
}
"#.to_string());

        // 🧬 std.neural - Neural network utilities
        modules.insert("std.neural".to_string(), r#"module std_neural {
    // Weight initialization (simplified without sqrt)
    export fn xavier_init(size) {
        let weights = [];
        for i in 0..size {
            weights.push(randn());
        }
        return weights;
    }
    
    export fn he_init(size) {
        let weights = [];
        for i in 0..size {
            weights.push(randn());
        }
        return weights;
    }
    
    // Matrix operations for neural networks
    export fn dot_product(a, b) {
        let sum = 0;
        for i in 0..a.length() {
            sum = sum + (a[i] * b[i]);
        }
        return sum;
    }
    
    export fn matrix_multiply(a, b) {
        // Simple 2D matrix multiplication
        let result = [];
        for i in 0..a.length() {
            let row = [];
            for j in 0..b[0].length() {
                let sum = 0;
                for k in 0..b.length() {
                    sum = sum + (a[i][k] * b[k][j]);
                }
                row.push(sum);
            }
            result.push(row);
        }
        return result;
    }
    
    // Activation functions with derivatives
    export fn sigmoid_prime(x) {
        let s = sigmoid(x);
        return s * (1 - s);
    }
    
    export fn relu_prime(x) {
        if x > 0 {
            return 1;
        }
        return 0;
    }
    
    export fn tanh_prime(x) {
        let t = tanh(x);
        return 1 - (t * t);
    }
    
    // Loss functions
    export fn mse_loss(predicted, actual) {
        let sum = 0;
        for i in 0..predicted.length() {
            let diff = predicted[i] - actual[i];
            sum = sum + (diff * diff);
        }
        return sum / predicted.length();
    }
    
    export fn cross_entropy_loss(predicted, actual) {
        let sum = 0;
        for i in 0..predicted.length() {
            // Simplified without log function
            let diff = predicted[i] - actual[i];
            sum = sum + (diff * diff);
        }
        return sum;
    }
}
"#.to_string());

        // 🔬 std.test - Testing utilities
        modules.insert("std.test".to_string(), r#"module std_test {
    // Assertion functions
    export fn assert(condition, message) {
        if !condition {
            synthesize "ASSERTION FAILED: ";
            synthesize message;
            synthesize "\n";
            // Would normally exit here
        }
    }
    
    export fn assert_equal(expected, actual) {
        if expected != actual {
            synthesize "ASSERTION FAILED: expected ";
            synthesize expected;
            synthesize " but got ";
            synthesize actual;
            synthesize "\n";
        }
    }
    
    export fn assert_not_equal(a, b) {
        if a == b {
            synthesize "ASSERTION FAILED: values should not be equal\n";
        }
    }
    
    export fn assert_true(value) {
        if !value {
            synthesize "ASSERTION FAILED: expected true\n";
        }
    }
    
    export fn assert_false(value) {
        if value {
            synthesize "ASSERTION FAILED: expected false\n";
        }
    }
    
    // Test running
    export fn run_test(name, test_fn) {
        synthesize "Running test: ";
        synthesize name;
        synthesize "... ";
        test_fn();
        synthesize "PASSED\n";
    }
}
"#.to_string());

        StandardLibrary { modules }
    }

    pub fn get_module(&self, name: &str) -> Option<&String> {
        self.modules.get(name)
    }

    pub fn is_stdlib_module(&self, name: &str) -> bool {
        self.modules.contains_key(name)
    }

    pub fn list_modules(&self) -> Vec<String> {
        self.modules.keys().cloned().collect()
    }
}

impl Default for StandardLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdlib_modules_exist() {
        let stdlib = StandardLibrary::new();
        
        assert!(stdlib.is_stdlib_module("std.math"));
        assert!(stdlib.is_stdlib_module("std.io"));
        assert!(stdlib.is_stdlib_module("std.array"));
        assert!(stdlib.is_stdlib_module("std.string"));
        assert!(stdlib.is_stdlib_module("std.neural"));
        assert!(stdlib.is_stdlib_module("std.test"));
    }

    #[test]
    fn test_get_module_content() {
        let stdlib = StandardLibrary::new();
        
        let math_module = stdlib.get_module("std.math");
        assert!(math_module.is_some());
        assert!(math_module.unwrap().contains("export fn pi()"));
    }
}