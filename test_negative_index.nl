organism TestNegativeIndex {
    fn main() {
        express "Testing negative numbers in arrays...";
        
        let arr = [10, -20, 30, -40];
        express "Created array [10, -20, 30, -40]";
        
        let v0 = arr[0];
        express "arr[0] = ";
        synthesize v0;
        
        let v1 = arr[1];
        express "arr[1] = ";
        synthesize v1;
        
        let v2 = arr[2];
        express "arr[2] = ";
        synthesize v2;
        
        let v3 = arr[3];
        express "arr[3] = ";
        synthesize v3;
        
        express "Done!";
    }
}