organism TestSimpleString {
    fn main() {
        express "Testing string concatenation...";
        let s1 = "Hello";
        let s2 = "World";
        let combined = s1 + s2;
        express "Combined string:";
        express combined;
        express "Done!";
    }
}