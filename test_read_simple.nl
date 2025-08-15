organism TestReadSimple {
    fn main() {
        express "Before read_file...";
        let content = read_file("dummy.txt");
        express "After read_file, content received!";
        express "read_file works!";
    }
}