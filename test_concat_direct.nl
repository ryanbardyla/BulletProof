organism TestConcatDirect {
    fn main() {
        express "Testing direct string concat...";
        let result = "Hello" + "World";
        express result;
        express "Done!";
    }
}