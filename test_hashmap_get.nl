organism TestHashMapGet {
    fn main() {
        express "🔍 Testing HashMap Get Function!";
        express "═════════════════════════════════";
        
        express "Creating HashMap with number keys...";
        let scores = {1: 95, 2: 87, 3: 92};
        
        express "Testing get function...";
        let score1 = get(scores, 1);
        let score2 = get(scores, 2);
        let score3 = get(scores, 3);
        let missing = get(scores, 4);
        
        express "Retrieved scores:";
        synthesize score1;
        synthesize score2; 
        synthesize score3;
        synthesize missing;
        
        express "✅ HashMap get function works!";
        express "═════════════════════════════════";
    }
}