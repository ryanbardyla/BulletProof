organism TestTypes {
    fn main() {
        // Type annotations
        let x: int = 42;
        let y: float = 3.14;
        let name: string = "NeuronLang";
        let nums: array = [1, 2, 3];
        
        express "Testing type annotations:";
        express "x = ";
        synthesize x;
        express "name = ";
        express name;
        
        express "Type test complete!";
    }
}