// Test debug -2
organism DebugNeg2 {
    fn birth() {
        express "Before assignment"
        let a = -2
        express "After assignment"
        synthesize a
        express "After synthesize"
    }
}