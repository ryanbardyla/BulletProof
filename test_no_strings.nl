organism TestNoStrings {
    fn main() {
        express "Testing without string operations...";
        
        // Test HashMap
        express "Creating HashMap...";
        let map = {"key": 123};
        express "Getting value...";
        let val = get(map, "key");
        express "Value:";
        synthesize val;
        
        // Test match
        express "Testing match...";
        let x = 42;
        match x {
            42 => express "Found 42!",
            _ => express "Not 42"
        }
        
        // Test loop with break/continue
        express "Testing loop...";
        let i = 0;
        while i < 5 {
            i = i + 1;
            if i == 2 {
                continue;
            }
            if i == 4 {
                break;
            }
            synthesize i;
        }
        
        express "Done!";
    }
}