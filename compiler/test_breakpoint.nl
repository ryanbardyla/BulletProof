// Test breakpoint support in NeuronLang
function main() {
    let x = 10
    synthesize("Starting computation")
    
    // Add a breakpoint here
    breakpoint
    
    let y = x + 5
    synthesize("After breakpoint, y = ")
    synthesize(y)
    
    // Loop with breakpoint
    for i = 0; i < 3; i = i + 1 {
        synthesize("Loop iteration: ")
        synthesize(i)
        
        if i == 2 {
            breakpoint  // Hit on specific condition
            synthesize("Hit breakpoint at i=2")
        }
    }
    
    synthesize("Program complete")
}