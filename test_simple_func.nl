organism TestSimpleFunc {
    // Define a very simple function
    fn get_five() {
        return 5;
    }
    
    fn main() {
        express "Testing simple function";
        
        let x = get_five();
        express "Result:";
        synthesize x;
        
        express "Done!";
    }
}