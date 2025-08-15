// 🔍 TEST: Heavy Peephole Optimization Opportunities
// This test is designed to create many optimization opportunities

organism HeavyOptTest {
    fn many_temporaries() {
        // Create many temporary variables (lots of push/pop)
        let a = 1;
        let b = a;
        let c = b;
        let d = c;
        let e = d;
        let f = e;
        
        // Operations with zero
        let x = a + 0;
        let y = b - 0;
        let z = c + 0 - 0 + 0;
        
        // Identity operations
        let same = x;
        let also_same = same;
        
        synthesize f + also_same;
    }
    
    fn redundant_computations() {
        let val = 42;
        
        // These create push/pop patterns
        let temp1 = val;
        let temp2 = temp1;
        let temp3 = temp2;
        
        // Multiple uses of same value
        let sum = temp3 + temp3 + temp3;
        
        synthesize sum;
    }
    
    fn main() {
        many_temporaries();
        redundant_computations();
    }
}