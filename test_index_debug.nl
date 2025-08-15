organism TestIndexDebug {
    fn main() {
        express "Debug array indexing...";
        
        let arr = [100, 200, 300];
        
        // Direct indexing
        let v = arr[1];
        express "arr[1] should be 200:";
        synthesize v;
        
        // Variable indexing  
        let idx = 1;
        let v2 = arr[idx];
        express "arr[idx] where idx=1 should be 200:";
        synthesize v2;
        
        express "Done!";
    }
}