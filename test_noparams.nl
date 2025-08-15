organism TestNoParams {
    fn get_ten() {
        return 10;
    }
    
    fn main() {
        express "Testing function with no params:";
        let x = get_ten();
        synthesize x;
    }
}