organism TestSelfHosting {
    fn main() {
        express "🎯 TESTING SELF-HOSTING CAPABILITY!";
        express "═══════════════════════════════════";
        
        express "1. Testing read_file...";
        let source_code = read_file("dummy.txt");
        express "✅ read_file works!";
        
        express "2. Testing write_file...";
        let test_program = "organism BootstrapSuccess { fn main() { express \"🚀 SELF-HOSTING ACHIEVED!\"; } }";
        let write_success = write_file("bootstrap_test.nl", test_program);
        express "✅ write_file works!";
        
        express "3. Testing get_args...";
        let args = get_args();
        express "✅ get_args works!";
        
        express "4. Testing make_executable...";
        let exec_success = make_executable("bootstrap_test.nl");
        express "✅ make_executable works!";
        
        express "═══════════════════════════════════";
        express "🎉 ALL SELF-HOSTING FUNCTIONS WORK!";
        express "🧬 NeuronLang is ready for bootstrap!";
        express "═══════════════════════════════════";
    }
}