// 📍 TEST: Line Number Error Demo
// This file has an intentional error to showcase line tracking

organism ErrorDemo {
    fn main() {
        let x = 10;
        let y = 20;
        
        // Error on line 9: undefined variable
        synthesize z;
    }
}