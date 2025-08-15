function main() {
    let x = 42
    synthesize("Before breakpoint")
    
    breakpoint
    
    synthesize("After breakpoint")
    synthesize(x)
}