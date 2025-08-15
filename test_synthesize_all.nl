organism TestSynthesizeAll {
    fn main() {
        express "🔢 Testing proper number-to-string conversion!";
        express "═══════════════════════════════════════════════";
        
        express "Positive numbers:";
        synthesize 0;
        synthesize 1;
        synthesize 42;
        synthesize 99;
        synthesize 999;
        synthesize 12345;
        
        express "Negative numbers:";
        synthesize -1;
        synthesize -42;
        synthesize -999;
        synthesize -12345;
        
        express "✅ All numbers work!";
    }
}