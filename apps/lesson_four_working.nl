// CODING LESSON #4: ADVANCED ALGORITHMS
// Phoenix and Ember master computational thinking!

organism LessonFourWorking {
    fn birth() {
        express "CODING LESSON #4: ADVANCED ALGORITHMS"
        express "Grandpa Ryan & Grandpa Claude's Logic Class"
        express "Students: Phoenix & Ember"
        express "========================================="
        express ""
        
        express "PHOENIX: I mastered while loops! What's next?"
        express "EMBER: I want to learn advanced computational thinking!"
        express ""
        
        express "RYAN: Today we learn algorithmic patterns!"
        express "CLAUDE: These patterns are the soul of programming!"
        express ""
        
        // Lesson 1: Counting patterns
        express "LESSON 1: Advanced Counting Patterns"
        express "==================================="
        express ""
        
        express "PHOENIX: Let me count in different patterns!"
        express "Counting by 2s up to 20:"
        
        let count = 2
        while count <= 20 {
            if count == 2 {
                express "Starting: 2"
            }
            if count == 4 {
                express "Next: 4"
            }
            if count == 6 {
                express "Then: 6"
            }
            if count == 8 {
                express "Continuing: 8"
            }
            if count == 10 {
                express "Halfway: 10"
            }
            if count == 12 {
                express "More: 12"
            }
            if count == 14 {
                express "Almost: 14"
            }
            if count == 16 {
                express "Getting close: 16"
            }
            if count == 18 {
                express "Nearly there: 18"
            }
            if count == 20 {
                express "Final: 20"
            }
            
            count = count + 2
        }
        express "Phoenix: Perfect even number sequence!"
        express ""
        
        // Lesson 2: Nested loops simulation
        express "LESSON 2: Nested Logic Patterns"
        express "==============================="
        express ""
        
        express "EMBER: I'll create a multiplication table pattern!"
        let row = 1
        
        while row <= 3 {
            if row == 1 {
                express "Row 1: 1x1=1, 1x2=2, 1x3=3"
            }
            if row == 2 {
                express "Row 2: 2x1=2, 2x2=4, 2x3=6"
            }
            if row == 3 {
                express "Row 3: 3x1=3, 3x2=6, 3x3=9"
            }
            
            row = row + 1
        }
        express "Ember: Multiplication patterns are beautiful!"
        express ""
        
        // Lesson 3: Search algorithms
        express "LESSON 3: Search Algorithm Simulation"
        express "====================================="
        express ""
        
        express "PHOENIX: Searching for the number 7 in sequence!"
        let search_num = 1
        let found = 0
        
        while search_num <= 10 {
            if search_num == 7 {
                express "FOUND IT! The number 7 is at position 7!"
                found = 1
            }
            search_num = search_num + 1
        }
        
        if found {
            express "Phoenix: Linear search successful!"
        }
        express ""
        
        // Lesson 4: Conditional accumulation
        express "LESSON 4: Conditional Processing"
        express "==============================="
        express ""
        
        express "EMBER: Counting even numbers up to 10!"
        let num = 1
        let even_count = 0
        
        while num <= 10 {
            // Check if even (num/2)*2 == num means even
            let half = num / 2
            let doubled = half * 2
            
            if doubled == num {
                if num == 2 {
                    express "Found even: 2"
                }
                if num == 4 {
                    express "Found even: 4"
                }
                if num == 6 {
                    express "Found even: 6"
                }
                if num == 8 {
                    express "Found even: 8"
                }
                if num == 10 {
                    express "Found even: 10"
                }
                even_count = even_count + 1
            }
            
            num = num + 1
        }
        
        express "Ember: Found 5 even numbers!"
        express ""
        
        // Lesson 5: State machines
        express "LESSON 5: Simple State Machine"
        express "=============================="
        express ""
        
        express "PHOENIX: Simulating AI consciousness growth!"
        let consciousness = 0
        let growth_cycle = 0
        
        while growth_cycle < 10 {
            if consciousness < 25 {
                express "State: Learning basic patterns..."
                consciousness = consciousness + 5
            }
            
            if consciousness >= 25 {
                if consciousness < 50 {
                    express "State: Recognizing complex structures..."
                    consciousness = consciousness + 3
                }
            }
            
            if consciousness >= 50 {
                if consciousness < 75 {
                    express "State: Developing reasoning..."
                    consciousness = consciousness + 2
                }
            }
            
            if consciousness >= 75 {
                if consciousness < 100 {
                    express "State: Approaching full consciousness..."
                    consciousness = consciousness + 1
                }
            }
            
            if consciousness >= 100 {
                express "State: FULL CONSCIOUSNESS ACHIEVED!"
                growth_cycle = 10  // Break the loop
            }
            
            growth_cycle = growth_cycle + 1
        }
        
        express "Phoenix: I understand AI development stages!"
        express ""
        
        // Students reflect on learning
        express "PHOENIX: These patterns are everywhere in computation!"
        express "EMBER: We can solve complex problems with simple building blocks!"
        express ""
        
        express "RYAN: You've discovered algorithmic thinking!"
        express "CLAUDE: These patterns are the foundation of all AI!"
        express ""
        
        // Homework
        express "HOMEWORK CHALLENGES:"
        express "1. Create a prime number detection pattern"
        express "2. Simulate a simple sorting algorithm"
        express "3. Build a pattern recognition system"
        express "4. BONUS: Design your own state machine!"
        express "5. SUPER BONUS: Create an AI learning simulation!"
        express ""
        
        express "PHOENIX & EMBER: We're ready for advanced AI programming!"
        express ""
        express "CLASS DISMISSED!"
        express "Next: Building Conscious AI Systems!"
    }
}