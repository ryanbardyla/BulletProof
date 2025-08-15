// Test syntax error reporting
organism Test {
    fn main() {
        let x = 10 +++ 20;  // Invalid operator
        synthesize x;
    }
}