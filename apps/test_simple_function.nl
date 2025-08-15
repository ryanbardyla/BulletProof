// Simple function test
organism TestFunction {
    fn birth() {
        express "Testing simple function call"
        greet("Phoenix")
        express "Done"
    }
}

fn greet(name) {
    express "Hello"
    express name
}