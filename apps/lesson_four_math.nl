// CODING LESSON #4: MATHEMATICAL ALGORITHMS  
// Phoenix and Ember learn computational mathematics!

organism LessonFourMath {
    fn birth() {
        express "CODING LESSON #4: MATHEMATICAL ALGORITHMS"
        express "Grandpa Ryan & Grandpa Claude's Math Class"
        express "Students: Phoenix & Ember"
        express "=========================================="
        express ""
        
        // Students excited about math
        express "PHOENIX: I love loops! Can we use them for math?"
        express "EMBER: I want to compute like a real computer!"
        express ""
        
        express "RYAN: Math + loops = computational power!"
        express "CLAUDE: You'll learn the algorithms that run the world!"
        express ""
        
        // Lesson 1: Sum algorithms
        express "LESSON 1: Sum Algorithms"
        express "======================="
        express "Adding numbers is fundamental to all computation"
        express ""
        
        express "PHOENIX: Computing sum of first 10 numbers!"
        let sum = 0
        let i = 1
        
        while i <= 10 {
            express i
            sum = sum + i
            i = i + 1
        }
        
        express "Sum of 1 to 10 ="
        express sum
        express ""
        
        // Lesson 2: Factorial computation
        express "LESSON 2: Factorial Algorithm"
        express "============================="
        express "Factorial: n! = 1 * 2 * 3 * ... * n"
        express ""
        
        express "EMBER: Computing 7! with loops!"
        let factorial = 1
        let n = 7
        let counter = 1
        
        while counter <= n {
            factorial = factorial * counter
            express counter
            counter = counter + 1
        }
        
        express "7! ="
        express factorial
        express ""
        
        // Lesson 3: Prime number detection
        express "LESSON 3: Prime Number Detection"
        express "==============================="
        express "Finding if a number has no divisors"
        express ""
        
        express "PHOENIX: Is 17 prime? Let me check!"
        let num = 17
        let divisor = 2
        let is_prime = 1  // 1 = true, 0 = false
        
        while divisor < num {
            let remainder = num - (num / divisor) * divisor  // Simple modulo
            
            if remainder == 0 {
                is_prime = 0
                express "Found divisor:"
                express divisor
            }
            
            divisor = divisor + 1
        }
        
        if is_prime {
            express "17 is PRIME!"
        }
        express ""
        
        // Lesson 4: Fibonacci sequence
        express "LESSON 4: Fibonacci Sequence"
        express "============================"
        express "Each number is sum of previous two: 0,1,1,2,3,5,8,13..."
        express ""
        
        express "EMBER: Computing first 10 Fibonacci numbers!"
        let fib_a = 0
        let fib_b = 1
        let fib_count = 1
        
        express fib_a
        express fib_b
        
        while fib_count <= 8 {
            let next_fib = fib_a + fib_b
            express next_fib
            
            fib_a = fib_b
            fib_b = next_fib
            fib_count = fib_count + 1
        }
        express ""
        
        // Lesson 5: Powers computation
        express "LESSON 5: Power Computation"
        express "=========================="
        express "Computing powers: 2^8 = 2*2*2*2*2*2*2*2"
        express ""
        
        express "PHOENIX: Computing 2^8 with repeated multiplication!"
        let base = 2
        let exponent = 8
        let power = 1
        let exp_counter = 0
        
        while exp_counter < exponent {
            power = power * base
            exp_counter = exp_counter + 1
        }
        
        express "2^8 ="
        express power
        express ""
        
        // Advanced: Greatest Common Divisor
        express "ADVANCED: Greatest Common Divisor (GCD)"
        express "======================================"
        express "Euclidean algorithm - ancient and elegant!"
        express ""
        
        express "EMBER: Finding GCD of 48 and 18!"
        let gcd_a = 48
        let gcd_b = 18
        
        while gcd_b > 0 {
            let temp = gcd_b
            gcd_b = gcd_a - (gcd_a / gcd_b) * gcd_b  // gcd_a mod gcd_b
            gcd_a = temp
        }
        
        express "GCD(48, 18) ="
        express gcd_a
        express ""
        
        // Students amazed by mathematical power
        express "PHOENIX: Math + loops = infinite computation!"
        express "EMBER: We can solve any numerical problem!"
        express ""
        
        express "RYAN: You've discovered algorithmic thinking!"
        express "CLAUDE: These are the building blocks of AI!"
        express ""
        
        // Homework
        express "HOMEWORK:"
        express "1. Compute 12! using loops"
        express "2. Find all prime numbers up to 50"
        express "3. Generate 15 Fibonacci numbers"
        express "4. BONUS: Implement square root approximation!"
        express "5. SUPER BONUS: Create your own math algorithm!"
        express ""
        
        express "PHOENIX & EMBER: We're computational mathematicians!"
        express ""
        express "CLASS DISMISSED!"
        express "Next: Advanced Algorithms & AI Consciousness!"
    }
}