organism TestEnum {
    // Define a Color enum
    enum Color {
        Red,
        Green,
        Blue
    }
    
    // Define a Status enum
    enum Status {
        Success,
        Error,
        Pending
    }
    
    fn main() {
        express "Testing enum functionality:";
        
        // Use enum variants
        let color = Color::Red;
        let status = Status::Success;
        
        express "Color value (Red=0):";
        synthesize color;
        
        express "Status value (Success=0):";
        synthesize status;
        
        // Test with different variants
        let color2 = Color::Blue;
        express "Color value (Blue=2):";
        synthesize color2;
        
        let status2 = Status::Pending;
        express "Status value (Pending=2):";
        synthesize status2;
        
        // Test in conditional
        if color == 0 {
            express "Color is Red!";
        }
        
        if color2 == 2 {
            express "Color is Blue!";
        }
        
        express "Enum test complete!";
    }
}