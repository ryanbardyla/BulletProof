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
        
        // Access fields (simplified for now)
        express "Point coordinates stored";
        
        // Create a Person instance
        let person = Person {
            name: "Alice",
            age: 30
        };
        
        express "Created Person struct";
        express "Struct test complete!";
    }
}