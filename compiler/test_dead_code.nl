organism TestDeadCode {
    // Case 1: Code after return (should be eliminated)
    fn test_return() {
        express "This will execute";
        return 42;
        express "This is dead code - should not execute";
        let dead_var = 100;
        synthesize dead_var;
    }
    
    fn main() {
        express "Testing dead code elimination:";
        
        let result = test_return();
        express "Function returned:";
        synthesize result;
        
        // Case 2: if(false) block (should be eliminated)
        if 0 {
            express "This if(false) block should be eliminated";
            let x = 999;
            synthesize x;
        } else {
            express "This else block should execute";
        }
        
        // Case 3: if(true) block (else should be eliminated)
        if 1 {
            express "This if(true) block should execute";
        } else {
            express "This else block should be eliminated";
            let y = 888;
            synthesize y;
        }
        
        // Case 4: while(false) loop (should be eliminated)
        while 0 {
            express "This while(false) should be eliminated";
            let z = 777;
            synthesize z;
        }
        
        // Case 5: Constant folding + dead code elimination
        if 2 - 2 {  // Evaluates to 0
            express "This should be eliminated (2-2 = 0)";
        } else {
            express "This should execute (2-2 = 0, so else runs)";
        }
        
        express "Dead code elimination test complete!";
    }
}