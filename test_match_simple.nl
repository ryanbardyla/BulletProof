organism TestMatchSimple {
    fn main() {
        express "🔀 Testing Simple Match!";
        
        express "Testing match with direct values...";
        match 2 {
            1 => express "Got one!",
            2 => express "Got two!",
            _ => express "Got something else!"
        };
        
        express "✅ Match works!";
    }
}