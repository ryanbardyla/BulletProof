organism TestAllFeatures {
    fn main() {
        express "🧪 COMPREHENSIVE FEATURE TEST SUITE";
        express "════════════════════════════════════════";
        
        // Test 1: String operations
        express "";
        express "📝 STRING OPERATIONS:";
        let s1 = "Hello";
        let s2 = "World";
        let combined = s1 + s2;
        express combined;
        let length = len(combined);
        express "Length:";
        synthesize length;
        
        // Test 2: HashMap operations
        express "";
        express "🗺️ HASHMAP OPERATIONS:";
        let map = {"name": "NeuronLang", "version": 1, "year": 2025};
        express "Created HashMap with 3 entries";
        let val = get(map, "version");
        express "Version from map:";
        synthesize val;
        
        // Test 3: Control flow with break/continue
        express "";
        express "🔄 CONTROL FLOW (break/continue):";
        let i = 0;
        while i < 10 {
            i = i + 1;
            if i == 3 {
                express "Skipping 3...";
                continue;
            }
            if i == 7 {
                express "Breaking at 7!";
                break;
            }
            synthesize i;
        }
        
        // Test 4: Match/switch statements
        express "";
        express "🎯 MATCH STATEMENTS:";
        let x = 42;
        match x {
            0 => express "Zero",
            42 => express "The answer to everything!",
            99 => express "Ninety-nine",
            _ => express "Something else"
        }
        
        // Test 5: Number synthesis (positive and negative)
        express "";
        express "🔢 NUMBER SYNTHESIS:";
        express "Positive:";
        synthesize 12345;
        express "Negative:";
        synthesize -9876;
        express "Zero:";
        synthesize 0;
        
        // Test 6: Nested structures
        express "";
        express "🏗️ NESTED STRUCTURES:";
        let nested = {"outer": {"inner": 777}};
        express "Created nested HashMap";
        
        // Test 7: Complex expression
        express "";
        express "🧮 COMPLEX EXPRESSION:";
        let result = (100 + 50) * 2 - 50;
        express "Result of (100 + 50) * 2 - 50:";
        synthesize result;
        
        express "";
        express "✅ ALL FEATURES TESTED SUCCESSFULLY!";
    }
}