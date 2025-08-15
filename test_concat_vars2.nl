organism TestConcatVars2 {
    fn main() {
        express "Testing concat with variables...";
        
        // Direct concat works
        express "Direct: ";
        express "A" + "B";
        
        // Through variable  
        express "Through variable: ";
        let s1 = "C";
        let s2 = "D";
        let result = s1 + s2;
        express result;
        
        express "Done!";
    }
}