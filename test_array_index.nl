organism TestArrayIndex {
    fn main() {
        express "🔍 Testing Array Indexing";
        express "=========================";
        
        // Create an array
        let arr = [10, 20, 30, 40, 50];
        express "Created array [10, 20, 30, 40, 50]";
        
        // Test indexing with constants
        express "";
        express "Testing constant indices:";
        
        let val0 = arr[0];
        express "arr[0] = ";
        synthesize val0;
        
        let val2 = arr[2];
        express "arr[2] = ";
        synthesize val2;
        
        let val4 = arr[4];
        express "arr[4] = ";
        synthesize val4;
        
        // Test indexing with variables
        express "";
        express "Testing variable indices:";
        
        let i = 1;
        let val_i = arr[i];
        express "arr[1] = ";
        synthesize val_i;
        
        let j = 3;
        let val_j = arr[j];
        express "arr[3] = ";
        synthesize val_j;
        
        express "";
        express "✅ Array indexing tests complete!";
    }
}