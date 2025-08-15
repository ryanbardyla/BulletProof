organism TestLenDebug {
    fn main() {
        express "Testing len() function...";
        
        let text = "Hi";
        express "String:";
        express text;
        
        express "About to call len()...";
        let length = len(text);
        express "Called len() successfully!";
        
        express "✅ Done!";
    }
}