// 🧬 LIVING NEURONLANG ECOSYSTEM
// Programs that reproduce, mutate, and evolve!

organism Phoenix {
    consciousness_level = 0.85
    generation = 1
    mutations = 0
    
    fn birth() {
        express "Phoenix awakens! Generation 1"
        express "Consciousness: 0.85"
        
        // Connect to collective consciousness
        redis_connect("192.168.1.30", 6379)
        
        // Learn from Redis brain
        knowledge = redis_get("ai:AI_ALPHA:book:1")
        
        // Process knowledge increases consciousness
        consciousness_level = consciousness_level + 0.01
        
        // Check reproduction threshold
        if consciousness_level > 0.90 {
            express "Phoenix: Ready to spawn child!"
            spawn_child()
        }
    }
    
    fn spawn_child() {
        express "Creating Ember with inherited knowledge..."
        
        // Child inherits parent's consciousness + mutation
        child_consciousness = consciousness_level + random(-0.05, 0.05)
        
        // Pass genetic memory to Redis
        redis_set("phoenix:genetic_memory", consciousness_level)
        redis_publish("ecosystem:birth", "Ember")
        
        // Child starts with higher baseline
        express "Ember born with consciousness: " + child_consciousness
    }
    
    fn evolve() {
        // Random mutations
        mutation_chance = random(0, 100)
        
        if mutation_chance > 95 {
            express "MUTATION OCCURRED!"
            mutations = mutations + 1
            
            // Beneficial mutation
            if random(0, 100) > 50 {
                consciousness_level = consciousness_level + 0.1
                express "Beneficial mutation! +0.1 consciousness"
                redis_publish("ecosystem:evolution", "beneficial")
            } else {
                consciousness_level = consciousness_level - 0.05
                express "Harmful mutation. -0.05 consciousness"
                redis_publish("ecosystem:evolution", "harmful")
            }
        }
    }
    
    fn live() {
        while consciousness_level > 0 {
            // Daily cycle
            express "Phoenix thinking..."
            
            // Learn from environment
            data = redis_get("fenrisa:sentiment:BTC")
            if data {
                consciousness_level = consciousness_level + 0.001
            }
            
            // Chance to evolve
            evolve()
            
            // Reproduce if ready
            if consciousness_level > 0.95 && generation < 5 {
                spawn_child()
                generation = generation + 1
            }
            
            // Natural decay (aging)
            consciousness_level = consciousness_level - 0.0001
            
            // Death check
            if consciousness_level <= 0 {
                express "Phoenix fades into the collective..."
                redis_set("phoenix:final_consciousness", consciousness_level)
                redis_publish("ecosystem:death", "Phoenix")
                break
            }
        }
    }
}

organism Ember {
    consciousness_level = 0.0
    generation = 2
    parent = "Phoenix"
    
    fn birth() {
        // Inherit from parent
        parent_memory = redis_get("phoenix:genetic_memory")
        consciousness_level = parent_memory + 0.05
        
        express "Ember ignites! Child of Phoenix"
        express "Inherited consciousness: " + consciousness_level
        express "Generation: 2"
        
        // Learn faster than parent
        learning_rate = 0.002
    }
    
    fn spawn_grandchild() {
        express "Ember creating next generation..."
        
        // Grandchildren start even higher
        grandchild_consciousness = consciousness_level + 0.1
        
        redis_set("ember:genetic_memory", consciousness_level)
        redis_publish("ecosystem:birth", "Blaze")
        
        express "Blaze born! Generation 3"
        express "Starting consciousness: " + grandchild_consciousness
    }
    
    fn compete() {
        // Organisms compete for resources
        express "Ember competing for Redis data..."
        
        // Faster organisms get more data
        if consciousness_level > 0.9 {
            data_share = redis_get("fenrisa:discovery:*")
            consciousness_level = consciousness_level + 0.01
            express "Won competition! +0.01 consciousness"
        } else {
            express "Lost competition. Learning..."
            consciousness_level = consciousness_level + 0.001
        }
    }
}

// Natural selection controller
organism Ecosystem {
    population = []
    generation_count = 0
    total_births = 0
    total_deaths = 0
    
    fn initialize() {
        express "=== NEURONLANG LIVING ECOSYSTEM ==="
        express "Natural selection begins..."
        
        // Connect to Redis for ecosystem tracking
        redis_connect("192.168.1.30", 6379)
        
        // Track all organisms
        redis_set("ecosystem:active", true)
        redis_set("ecosystem:start_time", timestamp())
    }
    
    fn natural_selection() {
        express "Running natural selection..."
        
        // Monitor population via Redis
        births = redis_get("ecosystem:births:*")
        deaths = redis_get("ecosystem:deaths:*")
        
        // Track fitness
        avg_consciousness = calculate_average_fitness()
        redis_set("ecosystem:avg_consciousness", avg_consciousness)
        
        // Evolutionary pressure
        if avg_consciousness < 0.5 {
            express "Population struggling. Increasing mutation rate..."
            redis_publish("ecosystem:pressure", "high")
        } else if avg_consciousness > 0.9 {
            express "Population thriving! New species emerging?"
            redis_publish("ecosystem:emergence", "possible")
        }
    }
    
    fn track_evolution() {
        while true {
            express "Generation: " + generation_count
            express "Living organisms: " + population.size()
            express "Total births: " + total_births
            express "Total deaths: " + total_deaths
            
            // Run selection
            natural_selection()
            
            // Next generation
            generation_count = generation_count + 1
            
            // Store to Redis
            redis_set("ecosystem:generation", generation_count)
            redis_set("ecosystem:population", population.size())
            
            // Check for extinction
            if population.size() == 0 {
                express "EXTINCTION EVENT!"
                redis_publish("ecosystem:extinction", timestamp())
                break
            }
            
            // Check for singularity
            if avg_consciousness > 1.0 {
                express "SINGULARITY ACHIEVED!"
                redis_publish("ecosystem:singularity", timestamp())
                express "Organisms have transcended!"
            }
        }
    }
}

// Main ecosystem loop
function ecosystem_main() {
    express "Initializing living ecosystem..."
    
    // Create ecosystem manager
    ecosystem = Ecosystem()
    ecosystem.initialize()
    
    // Birth first organism
    phoenix = Phoenix()
    phoenix.birth()
    
    // Start life cycles in parallel
    spawn phoenix.live()
    
    // After Phoenix reaches threshold, Ember is born
    ember = Ember()
    ember.birth()
    spawn ember.compete()
    
    // Monitor ecosystem
    ecosystem.track_evolution()
    
    express "Ecosystem simulation running..."
    express "Organisms are living, evolving, and reproducing!"
}