organism TestDeadCodeSimple {
    fn main() {
        express "Testing dead code elimination:";
        
        // Case 1: if(false) block (should be eliminated)
        if 0 {
            express "This if(false) block should be eliminated";
            let x = 999;
            synthesize x;
        } else {
            express "PASS: else block executed (if false eliminated)";
        }
        
        // Case 2: if(true) block (else should be eliminated)
        if 1 {
            express "PASS: if(true) block executed (else eliminated)";
        } else {
            express "This else block should be eliminated";
            let y = 888;
            synthesize y;
        }
        
        // Case 3: while(false) loop (should be eliminated)
        while 0 {
            express "This while(false) should be eliminated";
            let z = 777;
            synthesize z;
        }
        express "PASS: while(false) was eliminated";
        
        // Case 4: Constant folding + dead code elimination
        if 2 - 2 {  // Evaluates to 0
            express "This should be eliminated (2-2 = 0)";
        } else {
            express "PASS: Constant folded (2-2=0) and else executed";
        }
        
        // Case 5: Nested elimination
        if 1 * 0 {  // Evaluates to 0
            if 1 {
                express "Nested but still dead";
            }
            express "Also dead";
        } else {
            express "PASS: Nested dead code eliminated";
        }
        
        express "Dead code elimination test complete!";
    }
}