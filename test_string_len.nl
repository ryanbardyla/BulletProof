organism TestStringLen {
    fn main() {
        express "Testing string length...";
        
        let s = "Hello";
        let length = len(s);
        express "Length of 'Hello':";
        synthesize length;
        
        let s2 = "NeuronLang";
        let length2 = len(s2);
        express "Length of 'NeuronLang':";
        synthesize length2;
        
        express "Done!";
    }
}