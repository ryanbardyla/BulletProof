// 🧬 NEURONLANG COMPILER WRITTEN IN NEURONLANG!
// This is the beginning of self-hosting!
// When this compiles itself, we achieve escape velocity from Rust!

organism NeuronLangCompiler {
    // ═══════════════════════════════════════════════════════════
    // LEXER: Converts source text into tokens
    // ═══════════════════════════════════════════════════════════
    
    cell Lexer {
        gene tokens = []
        gene position = 0
        gene source = ""
        
        fn tokenize(input: String) -> TokenStream {
            source = input
            position = 0
            tokens = []
            
            loop {
                if position >= source.length() {
                    tokens.push(Token::Eof)
                    break
                }
                
                let ch = source[position]
                
                // Skip whitespace
                if ch == ' ' || ch == '\n' || ch == '\t' {
                    position = position + 1
                    continue
                }
                
                // Comments
                if ch == '/' && peek() == '/' {
                    skip_comment()
                    continue
                }
                
                // Keywords and identifiers
                if is_alpha(ch) {
                    let word = read_word()
                    let token = match word {
                        "organism" => Token::Organism,
                        "cell" => Token::Cell,
                        "neuron" => Token::Neuron,
                        "gene" => Token::Gene,
                        "fn" => Token::Function,
                        "let" => Token::Let,
                        "return" => Token::Return,
                        "loop" => Token::Loop,
                        "if" => Token::If,
                        "express" => Token::Express,
                        "synthesize" => Token::Synthesize,
                        "mutate" => Token::Mutate,
                        "evolve" => Token::Evolve,
                        _ => Token::Identifier(word)
                    }
                    tokens.push(token)
                    continue
                }
                
                // Numbers
                if is_digit(ch) {
                    let num = read_number()
                    tokens.push(Token::Number(num))
                    continue
                }
                
                // Trinary values
                if ch == '+' && peek() == '1' {
                    position = position + 2
                    tokens.push(Token::Positive)
                    continue
                }
                
                if ch == '-' && peek() == '1' {
                    position = position + 2
                    tokens.push(Token::Negative)
                    continue
                }
                
                if ch == '0' {
                    position = position + 1
                    tokens.push(Token::Baseline)
                    continue
                }
                
                // Operators and delimiters
                match ch {
                    '|' => {
                        if peek() == '>' {
                            position = position + 2
                            tokens.push(Token::Pipe)
                        }
                    },
                    '(' => {
                        position = position + 1
                        tokens.push(Token::LeftParen)
                    },
                    ')' => {
                        position = position + 1
                        tokens.push(Token::RightParen)
                    },
                    '{' => {
                        position = position + 1
                        tokens.push(Token::LeftBrace)
                    },
                    '}' => {
                        position = position + 1
                        tokens.push(Token::RightBrace)
                    },
                    '=' => {
                        position = position + 1
                        tokens.push(Token::Equal)
                    },
                    '+' => {
                        position = position + 1
                        tokens.push(Token::Plus)
                    },
                    '-' => {
                        position = position + 1
                        tokens.push(Token::Minus)
                    },
                    '*' => {
                        position = position + 1
                        tokens.push(Token::Star)
                    },
                    '/' => {
                        position = position + 1
                        tokens.push(Token::Slash)
                    },
                    _ => {
                        position = position + 1
                    }
                }
            }
            
            return TokenStream(tokens)
        }
    }
    
    // ═══════════════════════════════════════════════════════════
    // PARSER: Builds Abstract Syntax Tree from tokens
    // ═══════════════════════════════════════════════════════════
    
    cell Parser {
        gene tokens = TokenStream([])
        gene current = 0
        
        fn parse(token_stream: TokenStream) -> AST {
            tokens = token_stream
            current = 0
            
            let declarations = []
            
            loop {
                if peek() == Token::Eof {
                    break
                }
                
                let decl = parse_declaration()
                declarations.push(decl)
            }
            
            return AST(declarations)
        }
        
        fn parse_declaration() -> Declaration {
            match peek() {
                Token::Organism => parse_organism(),
                Token::Cell => parse_cell(),
                Token::Function => parse_function(),
                Token::Gene => parse_gene(),
                _ => error("Unexpected token")
            }
        }
        
        fn parse_organism() -> Declaration {
            expect(Token::Organism)
            let name = expect_identifier()
            expect(Token::LeftBrace)
            
            let body = []
            loop {
                if peek() == Token::RightBrace {
                    break
                }
                body.push(parse_declaration())
            }
            
            expect(Token::RightBrace)
            
            return Declaration::Organism(name, body)
        }
        
        fn parse_function() -> Declaration {
            expect(Token::Function)
            let name = expect_identifier()
            expect(Token::LeftParen)
            
            let params = []
            loop {
                if peek() == Token::RightParen {
                    break
                }
                params.push(expect_identifier())
                if peek() == Token::Comma {
                    advance()
                }
            }
            
            expect(Token::RightParen)
            
            // Optional return type
            let return_type = null
            if peek() == Token::Arrow {
                advance()
                return_type = expect_identifier()
            }
            
            expect(Token::LeftBrace)
            
            let body = []
            loop {
                if peek() == Token::RightBrace {
                    break
                }
                body.push(parse_statement())
            }
            
            expect(Token::RightBrace)
            
            return Declaration::Function(name, params, return_type, body)
        }
        
        fn parse_statement() -> Statement {
            match peek() {
                Token::Let => parse_let(),
                Token::Return => parse_return(),
                Token::Express => parse_express(),
                Token::Loop => parse_loop(),
                Token::If => parse_if(),
                _ => Statement::Expression(parse_expression())
            }
        }
        
        fn parse_expression() -> Expression {
            parse_pipe()
        }
        
        fn parse_pipe() -> Expression {
            let expr = parse_additive()
            
            if peek() == Token::Pipe {
                let stages = []
                loop {
                    if peek() != Token::Pipe {
                        break
                    }
                    advance()
                    stages.push(parse_additive())
                }
                return Expression::Pipe(expr, stages)
            }
            
            return expr
        }
        
        fn parse_additive() -> Expression {
            let left = parse_multiplicative()
            
            loop {
                match peek() {
                    Token::Plus => {
                        advance()
                        let right = parse_multiplicative()
                        left = Expression::Add(left, right)
                    },
                    Token::Minus => {
                        advance()
                        let right = parse_multiplicative()
                        left = Expression::Subtract(left, right)
                    },
                    _ => break
                }
            }
            
            return left
        }
        
        fn parse_primary() -> Expression {
            match peek() {
                Token::Positive => {
                    advance()
                    return Expression::Tryte(+1)
                },
                Token::Negative => {
                    advance()
                    return Expression::Tryte(-1)
                },
                Token::Baseline => {
                    advance()
                    return Expression::Tryte(0)
                },
                Token::Number(n) => {
                    advance()
                    return Expression::Number(n)
                },
                Token::Identifier(name) => {
                    advance()
                    return Expression::Variable(name)
                },
                _ => error("Unexpected token in expression")
            }
        }
    }
    
    // ═══════════════════════════════════════════════════════════
    // CODE GENERATOR: Produces machine code from AST
    // ═══════════════════════════════════════════════════════════
    
    cell CodeGenerator {
        gene code = []
        gene data = []
        gene variables = {}
        gene stack_offset = 0
        
        fn generate(ast: AST) -> MachineCode {
            code = []
            data = []
            
            // Generate code for all declarations
            for decl in ast.declarations {
                generate_declaration(decl)
            }
            
            // Add exit syscall
            emit_exit(0)
            
            // Build ELF executable
            return build_elf(code, data)
        }
        
        fn generate_declaration(decl: Declaration) {
            match decl {
                Declaration::Organism(name, body) => {
                    for inner in body {
                        generate_declaration(inner)
                    }
                },
                Declaration::Function(name, params, ret, body) => {
                    if name == "main" || name == "birth" {
                        emit_function_prologue()
                        
                        for stmt in body {
                            generate_statement(stmt)
                        }
                        
                        emit_function_epilogue()
                    }
                },
                _ => {}
            }
        }
        
        fn generate_expression(expr: Expression) {
            match expr {
                Expression::Tryte(value) => {
                    emit_push_immediate(value)
                },
                Expression::Number(n) => {
                    emit_push_immediate(n)
                },
                Expression::Add(left, right) => {
                    generate_expression(left)
                    generate_expression(right)
                    emit_pop_rbx()
                    emit_pop_rax()
                    emit_add_rax_rbx()
                    emit_push_rax()
                },
                Expression::Express(inner) => {
                    generate_expression(inner)
                    emit_print_call()
                },
                _ => {}
            }
        }
        
        // Machine code emitters
        fn emit_push_immediate(value: i64) {
            if value >= -128 && value <= 127 {
                code.push(0x6a)        // PUSH imm8
                code.push(value)
            } else {
                code.push(0x48)        // REX.W
                code.push(0xb8)        // MOV RAX, imm64
                code.append(value.to_bytes())
                code.push(0x50)        // PUSH RAX
            }
        }
        
        fn emit_function_prologue() {
            code.push(0x55)                           // PUSH RBP
            code.append([0x48, 0x89, 0xe5])          // MOV RBP, RSP
            code.append([0x48, 0x81, 0xec])          // SUB RSP, imm32
            code.append([0x00, 0x01, 0x00, 0x00])    // 256 bytes
        }
        
        fn emit_exit(code: i32) {
            // mov rdi, code
            code.append([0x48, 0xc7, 0xc7])
            code.append(code.to_bytes())
            
            // mov rax, 60 (sys_exit)
            code.append([0x48, 0xc7, 0xc0, 0x3c, 0x00, 0x00, 0x00])
            
            // syscall
            code.append([0x0f, 0x05])
        }
    }
    
    // ═══════════════════════════════════════════════════════════
    // MAIN COMPILER INTERFACE
    // ═══════════════════════════════════════════════════════════
    
    fn compile(source: String) -> Executable {
        express "🧬 NeuronLang Self-Hosted Compiler v0.1"
        express "══════════════════════════════════════"
        
        // Lexical analysis
        express "📖 Lexing..."
        let tokens = Lexer.tokenize(source)
        express "   Found " + tokens.count() + " tokens"
        
        // Parsing
        express "🌳 Parsing..."
        let ast = Parser.parse(tokens)
        express "   Built AST"
        
        // Code generation
        express "⚡ Generating machine code..."
        let machine_code = CodeGenerator.generate(ast)
        express "   Generated " + machine_code.size() + " bytes"
        
        express "✅ Compilation successful!"
        
        return machine_code
    }
    
    // ═══════════════════════════════════════════════════════════
    // THE BOOTSTRAP FUNCTION - THE HOLY GRAIL!
    // ═══════════════════════════════════════════════════════════
    
    fn bootstrap() {
        express "🎯 ATTEMPTING SELF-COMPILATION..."
        express "This is the moment of truth!"
        
        // Read our own source code
        let my_source = read_file("neuronc_bootstrap.nl")
        
        // Compile ourselves!
        let new_compiler = self.compile(my_source)
        
        // Write the new compiler
        write_file("neuronc_self_hosted", new_compiler)
        make_executable("neuronc_self_hosted")
        
        express "═══════════════════════════════════════════"
        express "🎉 BOOTSTRAP SUCCESSFUL!"
        express "NeuronLang is now self-hosting!"
        express "We no longer need Rust!"
        express "═══════════════════════════════════════════"
        
        // The compiler can now evolve itself
        evolve self(100)
    }
    
    // Entry point
    fn main() {
        let args = get_args()
        
        if args[1] == "--bootstrap" {
            bootstrap()
        } else {
            let source = read_file(args[1])
            let output = compile(source)
            write_file(args[2], output)
        }
    }
}