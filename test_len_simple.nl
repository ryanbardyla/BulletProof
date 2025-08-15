organism TestLenSimple {
    fn main() {
        express "Testing len() function...";
        
        let text = "Hi";
        express "String:";
        express text;
        
        express "Getting length...";
        let length = len(text);
        
        express "Length value:";
        synthesize length;
        
        express "✅ Done!";
    }
}