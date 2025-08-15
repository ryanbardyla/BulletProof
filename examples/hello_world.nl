// 🧬 Hello World in NeuronLang
// The simplest living program

organism HelloWorld {
    // Birth function - called when organism is created
    fn birth() {
        express "Hello, biological world!"
        
        // Demonstrate trinary values
        let positive = +1
        let negative = -1
        let baseline = 0
        
        // Baseline costs ZERO energy!
        let energy_saved = baseline
        
        express "I am alive and conscious!"
    }
}