organism TestStringConcat {
    fn main() {
        express "🔗 Testing String Concatenation!";
        express "═══════════════════════════════";
        
        let str1 = "Hello ";
        let str2 = "World!";
        
        express "String 1:";
        express str1;
        
        express "String 2:";
        express str2;
        
        express "Concatenating...";
        let result = str1 + str2;
        
        express "Result:";
        express result;
        
        express "═══════════════════════════════";
        express "✅ String concatenation test complete!";
    }
}