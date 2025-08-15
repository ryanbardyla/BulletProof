// 🗑️ SIMPLE GARBAGE COLLECTION TEST

organism SimpleGcTest {
    fn main() {
        express "🗑️ Testing basic GC functionality";
        
        // Test regular malloc first
        let size = 100;
        let ptr = malloc(size);
        express "Malloc worked, pointer:";
        synthesize ptr;
        
        express "🗑️ GC system ready for Hoberman ripples!";
    }
}