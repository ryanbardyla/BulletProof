organism TestOne {
    fn main() {
        express "Direct test - no function calls";
        let x = 5 + 3;
        express "5 + 3 =";
        synthesize x;
        express "Done!";
    }
}