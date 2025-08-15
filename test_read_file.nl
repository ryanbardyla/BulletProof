organism TestReadFile {
    fn main() {
        // Create a simple file first
        express "Testing read_file function...";
        
        // Try to read a simple text file!
        let content = read_file("/home/ryan/repo/Fenrisa/BULLETPROOF_PROJECT/simple.txt");
        express "File content:";
        express content;
        
        express "read_file test complete!";
    }
}