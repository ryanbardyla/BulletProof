organism TestWriteFile {
    fn main() {
        express "Testing write_file function...";
        
        let content = "Hello from NeuronLang!";
        let result = write_file("test_output.txt", content);
        
        express "Write file complete!";
    }
}