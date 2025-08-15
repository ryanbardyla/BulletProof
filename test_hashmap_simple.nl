organism TestHashMapSimple {
    fn main() {
        express "🔍 Testing Simple HashMap!";
        
        express "Creating very simple HashMap...";
        let simple = {1: 42};
        
        express "About to call get...";
        let value = get(simple, 1);
        
        express "Called get successfully!";
        synthesize value;
        
        express "✅ Done!";
    }
}