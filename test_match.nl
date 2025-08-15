organism TestMatch {
    fn main() {
        express "🔀 Testing Match/Switch Statements!";
        express "═══════════════════════════════════";
        
        express "Testing basic match with numbers...";
        let value = 2;
        
        let result = match value {
            1 => "one",
            2 => "two", 
            3 => "three",
            _ => "unknown"
        };
        
        express "Match result:";
        express result;
        
        express "Testing match with trinary values...";
        let trit = +1;
        
        match trit {
            +1 => express "Positive!",
            0 => express "Baseline!",
            -1 => express "Negative!",
            _ => express "Unknown trit!"
        };
        
        express "✅ Match statements work!";
        express "═══════════════════════════════════";
    }
}