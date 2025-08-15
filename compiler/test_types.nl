organism TestTypes {
    // Custom struct type
    struct Point {
        x: int,
        y: int
    }
    
    // Custom enum type
    enum Status {
        OK,
        Error
    }
    
    fn main() {
        express "Testing type annotations:";
        
        // Integer type
        let x: int = 42;
        express "Integer value:";
        synthesize x;
        
        // Float type
        let pi: float = 3.14159;
        express "Float value:";
        synthesize pi;
        
        // String type
        let message: string = "Hello, typed world!";
        express message;
        
        // Boolean (using 0 and 1)
        let is_true: bool = 1;
        let is_false: bool = 0;
        express "Boolean values:";
        synthesize is_true;
        synthesize is_false;
        
        let p: Point = Point { x: 10, y: 20 };
        express "Point created with type annotation";
        
        let status: Status = Status::OK;
        express "Status value:";
        synthesize status;
        
        // Array type (implicit for now)
        let numbers = [1, 2, 3, 4, 5];
        express "Array created with 5 elements";
        
        express "Type annotation test complete!";
    }
}