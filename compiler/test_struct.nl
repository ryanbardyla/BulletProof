organism TestStruct {
    // Define a simple Point struct
    struct Point {
        x: int,
        y: int
    }
    
    // Define a Person struct
    struct Person {
        name: string,
        age: int
    }
    
    fn main() {
        express "Testing struct functionality:";
        
        // Create a Point instance
        let p = Point {
            x: 10,
            y: 20
        };
        
        express "Created Point struct";
        
        // Access fields
        let x_val = p.x;
        let y_val = p.y;
        
        express "Point x coordinate:";
        synthesize x_val;
        express "Point y coordinate:";
        synthesize y_val;
        
        // Create a Person instance
        let person = Person {
            name: "Alice",
            age: 30
        };
        
        express "Created Person struct";
        express "Person name field accessed";
        express "Struct test complete!";
    }
}