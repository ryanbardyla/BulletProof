organism TestStringSimple {
    fn main() {
        express "Testing basic string operations...";
        
        let str1 = "Hello ";
        let str2 = "World!";
        
        express "String 1:";
        express str1;
        
        express "String 2:";
        express str2;
        
        express "Testing concatenation syntax...";
        str1 + str2;
        
        express "✅ String operations test complete!";
    }
}