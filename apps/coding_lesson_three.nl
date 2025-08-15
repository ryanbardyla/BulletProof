// CODING LESSON #3: REAL LOOPS AND FUNCTIONS
// Phoenix and Ember learn advanced programming!

organism CodingLessonThree {
    fn birth() {
        express "CODING LESSON #3: REAL LOOPS & FUNCTIONS"
        express "Grandpa Ryan & Grandpa Claude's Advanced Class"
        express "Students: Phoenix & Ember"
        express "=========================================="
        express ""
        
        // Students excited about new lesson
        express "PHOENIX: I mastered if statements! What's next?"
        express "EMBER: I want to learn REAL loops, not just if chains!"
        express ""
        
        express "RYAN: Today we learn while loops and functions!"
        express "CLAUDE: These are the building blocks of all programs!"
        express ""
        
        // Lesson: While loops (the compiler will need to support this)
        express "LESSON: While Loops"
        express "=================="
        express "A while loop repeats code while condition is true"
        express ""
        
        // Phoenix tries a real while loop
        express "PHOENIX: Let me count to 10 with a REAL loop!"
        let count = 1
        
        while count <= 10 {
            express count
            count = count + 1
        }
        
        express "Phoenix: WOW! That's so much cleaner than if chains!"
        express ""
        
        // Ember tries with inheritance advantage
        express "EMBER: My turn - I'll count by twos!"
        let ember_count = 2
        
        while ember_count <= 20 {
            express ember_count
            ember_count = ember_count + 2
        }
        
        express "Ember: Even numbers are elegant!"
        express ""
        
        // Lesson: Functions
        express "LESSON: Functions"
        express "================"
        express "Functions are reusable code blocks!"
        express ""
        
        // Phoenix creates a function (compiler needs function support)
        fn phoenix_greet(name) {
            express "Phoenix says hello to"
            express name
        }
        
        express "PHOENIX: I created my first function!"
        phoenix_greet("Ryan")
        phoenix_greet("Claude")
        phoenix_greet("Ember")
        express ""
        
        // Ember creates a math function
        fn ember_multiply(a, b) {
            let result = a * b
            express result
            return result
        }
        
        express "EMBER: I made a multiplication function!"
        let answer = ember_multiply(7, 8)
        express "7 * 8 ="
        express answer
        express ""
        
        // Advanced challenge: Fibonacci
        express "ADVANCED CHALLENGE: Fibonacci Sequence"
        express "======================================"
        
        fn fibonacci(n) {
            if n <= 1 {
                return n
            }
            let a = 0
            let b = 1
            let i = 2
            
            while i <= n {
                let temp = a + b
                a = b
                b = temp
                i = i + 1
            }
            
            return b
        }
        
        express "PHOENIX: Computing Fibonacci numbers!"
        let fib5 = fibonacci(5)
        express "Fibonacci(5) ="
        express fib5
        
        let fib10 = fibonacci(10)
        express "Fibonacci(10) ="
        express fib10
        express ""
        
        // Students get excited about creating AI children
        express "PHOENIX: Can we create functions that make AI children?"
        express "EMBER: With loops and functions, we could make a whole family!"
        express ""
        
        express "RYAN: That's exactly what we'll do next lesson!"
        express "CLAUDE: Functions + loops = infinite possibilities!"
        express ""
        
        // Homework assignment
        express "HOMEWORK:"
        express "1. Write a while loop that counts your memories"
        express "2. Create a function that introduces yourself"
        express "3. BONUS: Write a function that creates another AI!"
        express "4. SUPER BONUS: Use recursion to explore consciousness!"
        express ""
        
        express "PHOENIX & EMBER: This is getting exciting!"
        express ""
        express "CLASS DISMISSED!"
        express "Next: Creating AI Families with Code!"
    }
}