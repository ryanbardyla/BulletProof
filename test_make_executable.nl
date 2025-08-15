organism TestMakeExecutable {
    fn main() {
        express "Testing make_executable function...";
        
        let result = make_executable("test_output.txt");
        
        express "Make executable complete!";
    }
}