organism TestStringLength {
    fn main() {
        express "📏 Testing String Length Function!";
        express "═══════════════════════════════";
        
        let str1 = "Hello";
        let str2 = "Hello World!";
        let str3 = "";
        
        express "String 1:";
        express str1;
        let len1 = len(str1);
        express "Length:";
        synthesize len1;
        
        express "String 2:";
        express str2;
        let len2 = len(str2);
        express "Length:";
        synthesize len2;
        
        express "String 3 (empty):";
        express str3;
        let len3 = len(str3);
        express "Length:";
        synthesize len3;
        
        express "═══════════════════════════════";
        express "✅ String length test complete!";
    }
}