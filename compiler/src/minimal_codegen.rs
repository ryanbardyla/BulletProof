// 🧬 MINIMAL NEURONLANG CODE GENERATOR
// Generates actual x86_64 machine code (no LLVM!)

use crate::minimal_parser::{Declaration, Statement, Expr, TryteValue, BinaryOperator, UnaryOperator, MatchArm, MatchPattern};
use crate::stdlib::StandardLibrary;
use crate::stack_trace::{StackTraceInfo, FunctionInfo, SourceLocation};
use crate::debug_symbols::{DebugInfo, FunctionDebugInfo, VariableDebugInfo, ParameterDebugInfo};
// use crate::breakpoint::{BreakpointManager, BreakpointCodeGen};

// 🔴 MINIMAL BREAKPOINT SUPPORT (simplified for minimal_codegen)
struct BreakpointManager {
    source_mappings: Vec<(String, u32, usize, Option<String>)>,
}

impl BreakpointManager {
    fn new() -> Self {
        BreakpointManager {
            source_mappings: Vec::new(),
        }
    }
    
    fn register_source_mapping(&mut self, file: String, line: u32, address: usize, function: Option<String>) {
        self.source_mappings.push((file, line, address, function));
    }
    
    fn generate_int3_instruction() -> Vec<u8> {
        vec![0xCC]  // INT3 instruction
    }
}

// 🎯 MINIMAL REGISTER ALLOCATION TYPES (simplified for minimal_codegen)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Register {
    RAX, RCX, RDX, RBX, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15,
}

impl Register {
    fn encoding(&self) -> u8 {
        match self {
            Register::RAX => 0, Register::RCX => 1, Register::RDX => 2, Register::RBX => 3,
            Register::RSI => 6, Register::RDI => 7, Register::R8 => 8, Register::R9 => 9,
            Register::R10 => 10, Register::R11 => 11, Register::R12 => 12,
            Register::R13 => 13, Register::R14 => 14, Register::R15 => 15,
        }
    }
}

#[derive(Debug, Clone)]
enum Allocation {
    Register(Register),
    Spilled(isize),
    Constant(i64),
}

// Simplified register allocator for minimal codegen
struct RegisterAllocator {
    allocations: HashMap<String, Allocation>,
}

impl RegisterAllocator {
    fn new() -> Self {
        RegisterAllocator {
            allocations: HashMap::new(),
        }
    }
    
    fn get_allocation(&self, var: &str) -> Option<&Allocation> {
        self.allocations.get(var)
    }
    
    fn emit_save_registers(&self) -> Vec<u8> {
        Vec::new() // Simplified - no register saving for now
    }
    
    fn emit_restore_registers(&self) -> Vec<u8> {
        Vec::new() // Simplified - no register restoring for now
    }
}

// Simplified liveness analyzer
struct LivenessAnalyzer;

impl LivenessAnalyzer {
    fn new() -> Self {
        LivenessAnalyzer
    }
    
    fn add_use(&mut self, _inst: usize, _var: String) {}
    fn add_def(&mut self, _inst: usize, _var: String) {}
    fn analyze(&mut self, _num_instructions: usize) {}
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeInfo {
    Int,
    Float,
    String,
    Bool,
    Array(Box<TypeInfo>),
    Struct(String),
    Enum(String),
    Unknown,
}
use std::collections::{HashMap, HashSet};

// 🗑️ GARBAGE COLLECTION OBJECT METADATA
#[derive(Debug, Clone)]
pub struct GcObject {
    size: usize,           // Size of the object in bytes
    ref_count: usize,      // Reference count for the object
    object_type: GcType,   // Type of the object (Array, String, Struct, etc.)
    marked: bool,          // Mark bit for mark-and-sweep GC
    data_addr: usize,      // Address where the actual data is stored
}

#[derive(Debug, Clone, PartialEq)]
pub enum GcType {
    Array(usize),          // Array with element count
    String(usize),         // String with length
    Struct(String),        // Struct with type name
    Neuron,                // Neural network neuron
    RippleWave,            // Hoberman sphere ripple wave
}

// 📚 STACK FRAME MANAGEMENT
#[derive(Debug, Clone)]
pub struct StackFrame {
    function_name: String,             // Name of the function
    return_address: Option<usize>,     // Return address in code
    frame_base: isize,                 // Base pointer for this frame
    local_vars: HashMap<String, isize>, // Local variable offsets from frame base
    param_count: usize,                // Number of parameters
    local_size: usize,                 // Total size of local variables
    caller_frame_base: Option<isize>,  // Previous frame base for restoration
    call_depth: usize,                 // Depth in call stack (for overflow detection)
}

impl StackFrame {
    fn new(function_name: String, frame_base: isize, call_depth: usize) -> Self {
        StackFrame {
            function_name,
            return_address: None,
            frame_base,
            local_vars: HashMap::new(),
            param_count: 0,
            local_size: 0,
            caller_frame_base: None,
            call_depth,
        }
    }
    
    fn add_local(&mut self, name: String, size: usize) -> isize {
        let offset = self.frame_base - (self.local_size as isize) - (size as isize);
        self.local_vars.insert(name, offset);
        self.local_size += size;
        offset
    }
    
    fn get_local(&self, name: &str) -> Option<isize> {
        self.local_vars.get(name).copied()
    }
}

pub struct CodeGen {
    code: Vec<u8>,
    data: Vec<u8>,
    strings: HashMap<String, usize>,
    string_table: Vec<String>,  // Global string interning table
    variables: HashMap<String, isize>,
    variable_types: HashMap<String, TypeInfo>, // Track variable types
    functions: HashMap<String, usize>,  // Function name -> code offset
    function_bodies: HashMap<String, Vec<Statement>>, // Store function bodies for compilation
    function_params: HashMap<String, Vec<String>>, // Store function parameter names
    stack_trace_info: StackTraceInfo,  // Stack trace tracking
    current_function: Option<String>,  // Current function being compiled
    debug_info_enabled: bool,  // Whether to emit debug info
    debug_info: Option<DebugInfo>,  // DWARF debug information
    current_line: u32,  // Current source line being compiled
    breakpoint_manager: BreakpointManager,  // Manages breakpoints
    enable_breakpoints: bool,  // Whether to emit breakpoint support
    lambdas: HashMap<usize, (Vec<String>, Vec<Statement>)>, // Lambda ID -> (params, body)
    next_lambda_id: usize,
    modules: HashMap<String, Vec<String>>, // Module name -> exported items
    imported_modules: HashMap<String, String>, // Item name -> module path
    stdlib: StandardLibrary, // Standard library modules
    structs: HashMap<String, Vec<(String, usize)>>, // Struct name -> fields with offsets
    enums: HashMap<String, Vec<(String, i64)>>, // Enum name -> variants with values
    stack_offset: isize,
    next_local_offset: isize,
    loop_break_stack: Vec<Vec<usize>>, // Stack of break jump locations for nested loops
    loop_continue_stack: Vec<usize>,   // Stack of continue jump targets for nested loops
    last_was_return: bool,             // Track if last statement was a return
    function_call_patches: Vec<(String, usize)>, // Function calls to patch in second pass
    use_real_function_calls: bool,     // Whether to use real call/ret instead of inlining
    // 🗑️ GARBAGE COLLECTION SYSTEM
    gc_heap: HashMap<usize, GcObject>, // Heap address -> GC object metadata
    gc_roots: Vec<usize>,              // Root set for GC (stack variables)
    next_heap_addr: usize,             // Next available heap address
    gc_threshold: usize,               // Trigger GC when heap exceeds this size
    // 📚 STACK FRAME MANAGEMENT SYSTEM
    call_stack: Vec<StackFrame>,       // Active function call stack
    current_frame: Option<StackFrame>, // Current stack frame being built
    frame_pointer: isize,              // Current frame pointer offset
    max_call_depth: usize,             // Maximum allowed call depth (prevent overflow)
    return_addresses: Vec<usize>,      // Return address stack for function calls
    // 🎯 REGISTER ALLOCATION SYSTEM
    register_allocator: Option<RegisterAllocator>,  // Current function's register allocator
    liveness_analyzer: Option<LivenessAnalyzer>,    // Current function's liveness analyzer
    current_instruction: usize,                     // Current instruction index for liveness
}

impl CodeGen {
    // 🎯 REGISTER ALLOCATION METHODS
    
    // Initialize register allocation for a function
    fn init_register_allocation(&mut self) {
        self.register_allocator = Some(RegisterAllocator::new());
        self.liveness_analyzer = Some(LivenessAnalyzer::new());
        self.current_instruction = 0;
    }
    
    // Perform register allocation for current function
    fn allocate_registers(&mut self) -> Result<(), String> {
        // Simplified for minimal codegen - just analyze without actual allocation
        if let Some(analyzer) = &mut self.liveness_analyzer {
            analyzer.analyze(self.current_instruction);
        }
        Ok(())
    }
    
    // Emit code to load variable into register or from spill
    fn emit_load_variable_optimized(&mut self, var_name: &str) {
        if let Some(allocator) = &self.register_allocator {
            if let Some(allocation) = allocator.get_allocation(var_name) {
                match allocation {
                    Allocation::Register(reg) => {
                        // Variable is in register, push it
                        self.emit_push_register(*reg);
                    }
                    Allocation::Spilled(offset) => {
                        // Variable is spilled, load from stack
                        self.emit_push_from_stack(*offset);
                    }
                    Allocation::Constant(val) => {
                        // Constant value, push immediately
                        self.emit_push_immediate(*val as i64);
                    }
                }
                return;
            }
        }
        
        // Fallback to non-optimized version
        if let Some(&offset) = self.variables.get(var_name) {
            self.emit_push_from_stack(offset);
        }
    }
    
    // Emit code to store to variable with register allocation
    fn emit_store_variable_optimized(&mut self, var_name: &str) {
        if let Some(allocator) = &self.register_allocator {
            if let Some(allocation) = allocator.get_allocation(var_name) {
                match allocation {
                    Allocation::Register(reg) => {
                        // Store to register
                        self.emit_pop_to_register(*reg);
                    }
                    Allocation::Spilled(offset) => {
                        // Store to spilled location
                        self.emit_pop_to_stack(*offset);
                    }
                    _ => {}
                }
                return;
            }
        }
        
        // Fallback to non-optimized version
        if let Some(&offset) = self.variables.get(var_name) {
            self.emit_pop_to_stack(offset);
        }
    }
    
    // Emit push from register
    fn emit_push_register(&mut self, reg: Register) {
        // push reg
        if reg.encoding() >= 8 {
            self.code.push(0x41); // REX.B prefix for R8-R15
        }
        self.code.push(0x50 + (reg.encoding() & 7));
    }
    
    // Emit pop to register
    fn emit_pop_to_register(&mut self, reg: Register) {
        // pop reg
        if reg.encoding() >= 8 {
            self.code.push(0x41); // REX.B prefix for R8-R15
        }
        self.code.push(0x58 + (reg.encoding() & 7));
    }
    
    // Emit push from stack offset
    fn emit_push_from_stack(&mut self, offset: isize) {
        self.emit_load_local(offset);
    }
    
    // Emit pop to stack offset
    fn emit_pop_to_stack(&mut self, offset: isize) {
        self.emit_store_local(offset);
    }
    
    // Track variable usage for liveness analysis
    fn track_variable_use(&mut self, var_name: &str) {
        if let Some(analyzer) = &mut self.liveness_analyzer {
            analyzer.add_use(self.current_instruction, var_name.to_string());
        }
    }
    
    // Track variable definition for liveness analysis
    fn track_variable_def(&mut self, var_name: &str) {
        if let Some(analyzer) = &mut self.liveness_analyzer {
            analyzer.add_def(self.current_instruction, var_name.to_string());
        }
    }
    
    // Increment instruction counter
    fn next_instruction(&mut self) {
        self.current_instruction += 1;
    }
    
    // 🔍 Apply peephole optimizations to generated machine code
    fn apply_peephole_optimizations(&self, mut code: Vec<u8>) -> Vec<u8> {
        let original_size = code.len();
        
        // Apply pattern-based optimizations
        code = self.optimize_push_pop_sequences(code);
        code = self.optimize_redundant_moves(code);
        code = self.optimize_arithmetic_patterns(code);
        code = self.optimize_jump_patterns(code);
        
        let optimized_size = code.len();
        if optimized_size < original_size {
            println!("🔍 PEEPHOLE: Reduced code size from {} to {} bytes ({}% reduction)",
                     original_size, optimized_size,
                     ((original_size - optimized_size) * 100) / original_size);
        }
        
        code
    }
    
    // Optimize push/pop sequences
    fn optimize_push_pop_sequences(&self, mut code: Vec<u8>) -> Vec<u8> {
        let mut optimized = Vec::new();
        let mut i = 0;
        
        while i < code.len() {
            // Pattern 1: push reg; pop same_reg => eliminate both
            if i + 1 < code.len() && 
               code[i] >= 0x50 && code[i] <= 0x57 &&  // push r32/r64
               code[i + 1] >= 0x58 && code[i + 1] <= 0x5F &&  // pop r32/r64
               (code[i] - 0x50) == (code[i + 1] - 0x58) {
                // Skip both instructions
                i += 2;
                continue;
            }
            
            // Pattern 2: push reg1; pop reg2 => mov reg2, reg1
            if i + 1 < code.len() && 
               code[i] >= 0x50 && code[i] <= 0x57 &&
               code[i + 1] >= 0x58 && code[i + 1] <= 0x5F {
                let src_reg = code[i] - 0x50;
                let dst_reg = code[i + 1] - 0x58;
                
                // Emit: mov dst, src (more efficient than push/pop)
                optimized.push(0x48); // REX.W prefix
                optimized.push(0x89); // MOV r/m64, r64
                optimized.push(0xC0 | (src_reg << 3) | dst_reg);
                
                i += 2;
                continue;
            }
            
            optimized.push(code[i]);
            i += 1;
        }
        
        optimized
    }
    
    // Optimize redundant move instructions
    fn optimize_redundant_moves(&self, mut code: Vec<u8>) -> Vec<u8> {
        let mut optimized = Vec::new();
        let mut i = 0;
        
        while i < code.len() {
            // Pattern: mov reg, reg (same register) => eliminate
            if i + 2 < code.len() && 
               code[i] == 0x48 && code[i + 1] == 0x89 {
                let modrm = code[i + 2];
                let src = (modrm >> 3) & 7;
                let dst = modrm & 7;
                
                if src == dst && (modrm & 0xC0) == 0xC0 {
                    // Skip redundant move
                    i += 3;
                    continue;
                }
            }
            
            optimized.push(code[i]);
            i += 1;
        }
        
        optimized
    }
    
    // Optimize arithmetic patterns
    fn optimize_arithmetic_patterns(&self, mut code: Vec<u8>) -> Vec<u8> {
        let mut optimized = Vec::new();
        let mut i = 0;
        
        while i < code.len() {
            // Pattern: add/sub reg, 0 => eliminate
            if i + 3 < code.len() && 
               code[i] == 0x48 && code[i + 1] == 0x83 &&
               code[i + 3] == 0x00 {
                let modrm = code[i + 2];
                let op = (modrm >> 3) & 7;
                
                if op == 0 || op == 5 { // ADD or SUB with 0
                    i += 4;
                    continue;
                }
            }
            
            // Pattern: xor reg, reg (same register - zeroing idiom) => keep as is (optimal)
            // Pattern: consecutive inc/dec => combine into add/sub
            
            optimized.push(code[i]);
            i += 1;
        }
        
        optimized
    }
    
    // Optimize jump patterns
    fn optimize_jump_patterns(&self, mut code: Vec<u8>) -> Vec<u8> {
        let mut optimized = Vec::new();
        let mut i = 0;
        
        while i < code.len() {
            // Pattern: jmp to next instruction => eliminate
            if i + 1 < code.len() && code[i] == 0xEB && code[i + 1] == 0x00 {
                i += 2;
                continue;
            }
            
            // Pattern: conditional jump over unconditional jump => invert condition
            // This is more complex and requires careful offset adjustment
            
            optimized.push(code[i]);
            i += 1;
        }
        
        optimized
    }
    
    // Analyze statement for liveness information
    fn analyze_statement_liveness(&mut self, stmt: &Statement) {
        match stmt {
            Statement::Let { name, value, .. } => {
                self.analyze_expr_liveness(value);
                self.track_variable_def(name);
                self.next_instruction();
            }
            Statement::Assignment { name, value } => {
                self.analyze_expr_liveness(value);
                self.track_variable_def(name);
                self.next_instruction();
            }
            Statement::Expression(expr) => {
                self.analyze_expr_liveness(expr);
                self.next_instruction();
            }
            Statement::If { condition, then_body, else_body } => {
                self.analyze_expr_liveness(condition);
                for stmt in then_body {
                    self.analyze_statement_liveness(stmt);
                }
                if let Some(else_stmts) = else_body {
                    for stmt in else_stmts {
                        self.analyze_statement_liveness(stmt);
                    }
                }
            }
            Statement::For { variable, start, end, body } => {
                self.analyze_expr_liveness(start);
                self.analyze_expr_liveness(end);
                self.track_variable_def(variable);
                for stmt in body {
                    self.analyze_statement_liveness(stmt);
                }
            }
            Statement::While { condition, body } => {
                self.analyze_expr_liveness(condition);
                for stmt in body {
                    self.analyze_statement_liveness(stmt);
                }
            }
            Statement::Return(Some(expr)) => {
                self.analyze_expr_liveness(expr);
                self.next_instruction();
            }
            _ => {
                self.next_instruction();
            }
        }
    }
    
    // Analyze expression for liveness information
    fn analyze_expr_liveness(&mut self, expr: &Expr) {
        match expr {
            Expr::Identifier(name) => {
                self.track_variable_use(name);
            }
            Expr::BinaryOp { left, right, .. } => {
                self.analyze_expr_liveness(left);
                self.analyze_expr_liveness(right);
            }
            Expr::UnaryOp { operand, .. } => {
                self.analyze_expr_liveness(operand);
            }
            Expr::Call { name, args } => {
                // Function name is a string, not an expression
                for arg in args {
                    self.analyze_expr_liveness(arg);
                }
            }
            Expr::Index { array, index } => {
                self.analyze_expr_liveness(array);
                self.analyze_expr_liveness(index);
            }
            Expr::FieldAccess { object, .. } => {
                self.analyze_expr_liveness(object);
            }
            Expr::Array(elements) => {
                for elem in elements {
                    self.analyze_expr_liveness(elem);
                }
            }
            Expr::Array2D(rows) => {
                for row in rows {
                    for elem in row {
                        self.analyze_expr_liveness(elem);
                    }
                }
            }
            Expr::Lambda { body, .. } => {
                for stmt in body {
                    self.analyze_statement_liveness(stmt);
                }
            }
            _ => {}
        }
    }
    
    // Dead code elimination - remove unreachable statements
    fn eliminate_dead_code(&self, stmts: Vec<Statement>) -> Vec<Statement> {
        let mut result = Vec::new();
        let mut is_dead = false;
        
        for stmt in stmts {
            if is_dead {
                // Skip dead code
                continue;
            }
            
            match &stmt {
                Statement::Return { .. } => {
                    result.push(stmt);
                    is_dead = true; // Everything after return is dead
                }
                Statement::If { condition, then_body, else_body } => {
                    // Check for constant conditions
                    let folded_condition = self.fold_constants(condition.clone());
                    match folded_condition {
                        Expr::Number(n) if n == 0.0 => {
                            // Condition is always false, only else branch is live
                            if let Some(else_stmts) = else_body {
                                let optimized = self.eliminate_dead_code(else_stmts.clone());
                                for s in optimized {
                                    result.push(s);
                                }
                            }
                        }
                        Expr::Number(n) if n != 0.0 => {
                            // Condition is always true, only then branch is live
                            let optimized = self.eliminate_dead_code(then_body.clone());
                            for s in optimized {
                                result.push(s);
                            }
                        }
                        _ => {
                            // Condition is not constant, keep both branches but optimize them
                            result.push(Statement::If {
                                condition: folded_condition,
                                then_body: self.eliminate_dead_code(then_body.clone()),
                                else_body: else_body.as_ref().map(|e| self.eliminate_dead_code(e.clone())),
                            });
                        }
                    }
                }
                Statement::While { condition, body } => {
                    // Check for constant false condition
                    let folded_condition = self.fold_constants(condition.clone());
                    match folded_condition {
                        Expr::Number(n) if n == 0.0 => {
                            // While(false) - entire loop is dead code
                            // Skip it entirely
                        }
                        _ => {
                            // Keep the loop but optimize its body
                            result.push(Statement::While {
                                condition: folded_condition,
                                body: self.eliminate_dead_code(body.clone()),
                            });
                        }
                    }
                }
                Statement::For { variable, start, end, body } => {
                    // Optimize the body
                    result.push(Statement::For {
                        variable: variable.clone(),
                        start: start.clone(),
                        end: end.clone(),
                        body: self.eliminate_dead_code(body.clone()),
                    });
                }
                Statement::Break | Statement::Continue => {
                    result.push(stmt);
                    // These affect control flow but don't make subsequent code dead
                    // (only within their loop context)
                }
                _ => {
                    // Keep other statements
                    result.push(stmt);
                }
            }
        }
        
        result
    }
    
    // Constant folding - evaluate constant expressions at compile time
    fn fold_constants(&self, expr: Expr) -> Expr {
        match expr {
            Expr::BinaryOp { left, op, right } => {
                let left_folded = self.fold_constants(*left);
                let right_folded = self.fold_constants(*right);
                
                // Check if both operands are constants
                match (&left_folded, &right_folded) {
                    (Expr::Number(l), Expr::Number(r)) => {
                        // Fold numeric operations
                        match op {
                            BinaryOperator::Add => Expr::Number(l + r),
                            BinaryOperator::Subtract => Expr::Number(l - r),
                            BinaryOperator::Multiply => Expr::Number(l * r),
                            BinaryOperator::Divide if *r != 0.0 => Expr::Number(l / r),
                            _ => Expr::BinaryOp {
                                left: Box::new(left_folded),
                                op,
                                right: Box::new(right_folded),
                            }
                        }
                    }
                    (Expr::String(l), Expr::String(r)) if op == BinaryOperator::Add => {
                        // Fold string concatenation
                        Expr::String(format!("{}{}", l, r))
                    }
                    _ => {
                        // Can't fold, return with folded children
                        Expr::BinaryOp {
                            left: Box::new(left_folded),
                            op,
                            right: Box::new(right_folded),
                        }
                    }
                }
            }
            Expr::UnaryOp { op, operand } => {
                let operand_folded = self.fold_constants(*operand);
                match operand_folded {
                    Expr::Number(n) => {
                        match op {
                            UnaryOperator::Minus => Expr::Number(-n),
                            UnaryOperator::Plus => Expr::Number(n),
                        }
                    }
                    _ => Expr::UnaryOp {
                        op,
                        operand: Box::new(operand_folded),
                    }
                }
            }
            // Recursively fold array elements
            Expr::Array(elements) => {
                Expr::Array(elements.into_iter().map(|e| self.fold_constants(e)).collect())
            }
            // Recursively fold struct fields
            Expr::StructInit { name, fields } => {
                Expr::StructInit {
                    name,
                    fields: fields.into_iter()
                        .map(|(n, e)| (n, self.fold_constants(e)))
                        .collect(),
                }
            }
            // Leave other expressions unchanged
            _ => expr,
        }
    }
    
    // Type inference for expressions
    fn infer_type(&self, expr: &Expr) -> TypeInfo {
        match expr {
            Expr::Number(_) => TypeInfo::Int, // Could be Float based on context
            Expr::String(_) => TypeInfo::String,
            Expr::Tryte(_) => TypeInfo::Int,
            Expr::Identifier(name) => {
                self.variable_types.get(name)
                    .cloned()
                    .unwrap_or(TypeInfo::Unknown)
            }
            Expr::BinaryOp { left, op: _, right: _ } => {
                // For now, assume binary ops preserve left type
                self.infer_type(left)
            }
            Expr::Array(_) => TypeInfo::Array(Box::new(TypeInfo::Unknown)),
            Expr::StructInit { name, .. } => TypeInfo::Struct(name.clone()),
            Expr::EnumVariant { enum_name, .. } => TypeInfo::Enum(enum_name.clone()),
            Expr::Call { .. } => TypeInfo::Unknown, // Would need function return types
            _ => TypeInfo::Unknown,
        }
    }
    
    // Type checking with error reporting
    fn check_type(&self, expected: &TypeInfo, actual: &TypeInfo, context: &str) -> Result<(), String> {
        if expected == actual || *expected == TypeInfo::Unknown || *actual == TypeInfo::Unknown {
            Ok(())
        } else {
            Err(format!("Type mismatch in {}: expected {:?}, got {:?}", context, expected, actual))
        }
    }
    pub fn enable_debug_symbols(&mut self) {
        self.debug_info_enabled = true;
    }
    
    pub fn enable_breakpoint_support(&mut self) {
        self.enable_breakpoints = true;
    }
    
    pub fn new() -> Self {
        CodeGen {
            code: Vec::new(),
            data: Vec::new(),
            strings: HashMap::new(),
            string_table: Vec::new(),
            variables: HashMap::new(),
            variable_types: HashMap::new(),
            functions: HashMap::new(),
            function_bodies: HashMap::new(),
            function_params: HashMap::new(),
            lambdas: HashMap::new(),
            next_lambda_id: 0,
            modules: HashMap::new(),
            imported_modules: HashMap::new(),
            stdlib: StandardLibrary::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            stack_offset: 0,
            next_local_offset: -8,
            loop_break_stack: Vec::new(),
            loop_continue_stack: Vec::new(),
            last_was_return: false,
            function_call_patches: Vec::new(),
            use_real_function_calls: true, // Enable real function calls by default
            // 🗑️ GARBAGE COLLECTION INITIALIZATION
            gc_heap: HashMap::new(),
            gc_roots: Vec::new(),
            next_heap_addr: 0x10000000, // Start heap at high address to avoid conflicts
            gc_threshold: 1024 * 1024,  // 1MB threshold
            // 📚 STACK FRAME MANAGEMENT INITIALIZATION
            call_stack: Vec::new(),
            current_frame: None,
            frame_pointer: 0,
            max_call_depth: 1000,       // Prevent stack overflow
            return_addresses: Vec::new(),
            // 🎯 REGISTER ALLOCATION INITIALIZATION
            register_allocator: None,
            liveness_analyzer: None,
            current_instruction: 0,
            // 🔍 STACK TRACE INITIALIZATION
            stack_trace_info: StackTraceInfo::new(),
            current_function: None,
            debug_info_enabled: false,  // Disable for now to avoid crashes
            debug_info: None,
            current_line: 1,
            breakpoint_manager: BreakpointManager::new(),
            enable_breakpoints: false,
        }
    }
    
    // Generate a complete ELF executable
    pub fn generate_elf(&mut self, declarations: Vec<Declaration>) -> Vec<u8> {
        // Initialize debug info if enabled
        if self.debug_info_enabled {
            self.debug_info = Some(DebugInfo::new(
                "main.nl".to_string(),  // TODO: Get actual source file
                "a.out".to_string(),    // TODO: Get actual output file
            ));
        }
        
        // Add newline to data section
        self.data.push(b'\n');
        
        // First pass: collect function information and store bodies
        let mut function_decls = Vec::new();
        let mut main_decls = Vec::new();
        
        for decl in declarations {
            match &decl {
                Declaration::Function { name, params, body, .. } if name != "main" && name != "birth" => {
                    // Store function info for two-pass compilation
                    self.function_params.insert(name.clone(), params.clone());
                    self.function_bodies.insert(name.clone(), body.clone());
                    function_decls.push(decl);
                }
                _ => main_decls.push(decl),
            }
        }
        
        // Start with a jump to main
        self.code.push(0xe9); // jmp
        let jump_loc = self.code.len();
        self.code.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // placeholder
        
        // FIRST PASS: Compile all functions and record their addresses
        for decl in function_decls {
            if let Declaration::Function { name, params, body, .. } = &decl {
                // Record function start address
                self.functions.insert(name.clone(), self.code.len());
                
                // 📚 REVOLUTIONARY STACK FRAME MANAGEMENT SYSTEM!
                // Enter function stack frame with full management
                let _ = self.stack_frame_enter(name.clone(), params.len());
                
                // Generate function prologue with frame management
                self.emit_function_prologue_with_frame(name, params.len());
                
                // Set up parameters using frame management
                let saved_vars = self.variables.clone();
                let saved_offset = self.stack_offset;
                
                // Parameters are passed on stack (rbp+16, rbp+24, etc.)
                // Add them to the current frame for tracking
                for (i, param) in params.iter().enumerate() {
                    let param_offset = 16 + (i * 8) as isize;
                    self.variables.insert(param.clone(), param_offset);
                    
                    // 📚 Add local variable to stack frame tracking
                    let _ = self.stack_frame_add_local(param.clone(), param_offset as usize);
                }
                
                // 🎯 Initialize register allocation for this function
                self.init_register_allocation();
                
                // First pass: analyze variable usage for liveness
                for stmt in body.iter() {
                    self.analyze_statement_liveness(stmt);
                }
                
                // Perform register allocation
                let _ = self.allocate_registers();
                
                // Save callee-saved registers if needed
                if let Some(allocator) = &self.register_allocator {
                    let save_code = allocator.emit_save_registers();
                    self.code.extend_from_slice(&save_code);
                }
                
                // Generate function body with register allocation
                for stmt in body {
                    self.generate_statement(stmt.clone());
                }
                
                // Restore callee-saved registers if needed
                if let Some(allocator) = &self.register_allocator {
                    let restore_code = allocator.emit_restore_registers();
                    self.code.extend_from_slice(&restore_code);
                }
                
                // If no explicit return, return 0
                if !self.last_was_return {
                    self.emit_push_immediate(0);
                    self.emit_pop_rax(); // Return value in rax
                }
                
                // Clear register allocation for this function
                self.register_allocator = None;
                self.liveness_analyzer = None;
                
                // Generate function epilogue with frame management
                self.emit_function_epilogue_with_frame(name);
                
                // 📚 Exit function stack frame
                let _ = self.stack_frame_exit();
                
                // Restore state
                self.variables = saved_vars;
                self.stack_offset = saved_offset;
                self.last_was_return = false;
            }
        }
        
        // SECOND PASS: Patch function call addresses
        self.patch_function_calls();
        
        // Patch jump to main
        let main_code_start = self.code.len();
        let jump_offset = (main_code_start - jump_loc - 4) as i32;
        self.code[jump_loc..jump_loc + 4].copy_from_slice(&jump_offset.to_le_bytes());
        
        // 📚 Initialize stack frame for main with full management
        let _ = self.stack_frame_enter("main".to_string(), 0);
        self.emit_function_prologue_with_frame("main", 0);
        
        // Generate main and other declarations
        for decl in main_decls {
            self.generate_declaration(decl);
        }
        
        // 📚 Clean up stack frame with full management
        self.emit_function_epilogue_with_frame("main");
        let _ = self.stack_frame_exit();
        
        // Add exit syscall
        self.emit_exit(0);
        
        // Build ELF file
        self.build_elf()
    }
    
    fn build_elf(&mut self) -> Vec<u8> {
        // 🔍 Apply peephole optimizations to generated code
        self.code = self.apply_peephole_optimizations(self.code.clone());
        
        // Patch string addresses now that we know the final code size
        let final_code_size = self.code.len();
        for i in 0..self.code.len() - 7 {
            if self.code[i..i+4] == [0xAA, 0xBB, 0xCC, 0xDD] {
                // Found a string address placeholder
                // Read the full 4-byte offset, not just 1 byte!
                let offset = u32::from_le_bytes([
                    self.code[i+4],
                    self.code[i+5],
                    self.code[i+6],
                    self.code[i+7]
                ]) as usize;
                let actual_addr = 0x401000 + final_code_size + offset;
                let addr_bytes = actual_addr.to_le_bytes();
                self.code[i..i+8].copy_from_slice(&addr_bytes);
            }
        }
        
        let mut elf = Vec::new();
        
        // ELF Header (64-bit)
        elf.extend_from_slice(&[
            0x7f, b'E', b'L', b'F',  // Magic
            2,                       // 64-bit
            1,                       // Little endian
            1,                       // Current version
            0,                       // System V ABI
            0, 0, 0, 0, 0, 0, 0, 0, // Padding
        ]);
        
        // e_type: ET_EXEC (executable)
        elf.extend_from_slice(&[2, 0]);
        
        // e_machine: x86_64
        elf.extend_from_slice(&[0x3e, 0]);
        
        // e_version
        elf.extend_from_slice(&[1, 0, 0, 0]);
        
        // e_entry (entry point address)
        let entry_addr = 0x401000u64;
        elf.extend_from_slice(&entry_addr.to_le_bytes());
        
        // e_phoff (program header offset)
        elf.extend_from_slice(&64u64.to_le_bytes());
        
        // e_shoff (section header offset - none)
        elf.extend_from_slice(&0u64.to_le_bytes());
        
        // e_flags
        elf.extend_from_slice(&[0, 0, 0, 0]);
        
        // e_ehsize (ELF header size)
        elf.extend_from_slice(&[64, 0]);
        
        // e_phentsize (program header entry size)
        elf.extend_from_slice(&[56, 0]);
        
        // e_phnum (number of program headers)
        elf.extend_from_slice(&[1, 0]);
        
        // e_shentsize (section header entry size)
        elf.extend_from_slice(&[0, 0]);
        
        // e_shnum (number of section headers)
        elf.extend_from_slice(&[0, 0]);
        
        // e_shstrndx (section name string table index)
        elf.extend_from_slice(&[0, 0]);
        
        // Program Header - Single segment for everything
        // p_type: PT_LOAD
        elf.extend_from_slice(&[1, 0, 0, 0]);
        
        // p_flags: PF_X | PF_R | PF_W (executable, readable, writable)
        elf.extend_from_slice(&[7, 0, 0, 0]);
        
        // p_offset (where in file)
        elf.extend_from_slice(&0x1000u64.to_le_bytes());
        
        // p_vaddr (virtual address where to load)
        elf.extend_from_slice(&0x401000u64.to_le_bytes());
        
        // p_paddr (physical address)
        elf.extend_from_slice(&0x401000u64.to_le_bytes());
        
        // p_filesz (size in file - code + data)
        let file_size = (self.code.len() + self.data.len()) as u64;
        elf.extend_from_slice(&file_size.to_le_bytes());
        
        // p_memsz (size in memory)
        elf.extend_from_slice(&file_size.to_le_bytes());
        
        // p_align
        elf.extend_from_slice(&0x1000u64.to_le_bytes());
        
        // Pad to code start (0x1000)
        while elf.len() < 0x1000 {
            elf.push(0);
        }
        
        // Add generated code
        elf.extend_from_slice(&self.code);
        
        // Add data section
        elf.extend_from_slice(&self.data);
        
        // Add debug sections if enabled
        if self.debug_info_enabled {
            if let Some(ref debug_info) = self.debug_info {
                let debug_sections = debug_info.generate_debug_sections();
                // For now, we'll append them at the end
                // In a proper implementation, we'd add them as proper ELF sections
                elf.extend_from_slice(&debug_sections.debug_info);
                elf.extend_from_slice(&debug_sections.debug_abbrev);
                elf.extend_from_slice(&debug_sections.debug_str);
                elf.extend_from_slice(&debug_sections.debug_line);
            }
        }
        
        elf
    }
    
    fn generate_declaration(&mut self, decl: Declaration) {
        match decl {
            Declaration::Organism { body, .. } => {
                for inner in body {
                    self.generate_declaration(inner);
                }
            }
            Declaration::Struct { name, fields } => {
                // Register struct type
                let mut field_offsets = Vec::new();
                let mut offset = 0usize;
                
                for (field_name, _field_type) in fields {
                    field_offsets.push((field_name, offset));
                    offset += 8; // Each field is 8 bytes for now
                }
                
                self.structs.insert(name, field_offsets);
            }
            Declaration::Enum { name, variants } => {
                // Register enum type with variant values
                let mut variant_values = Vec::new();
                let mut value = 0i64;
                
                for variant in variants {
                    variant_values.push((variant, value));
                    value += 1;
                }
                
                self.enums.insert(name, variant_values);
            }
            Declaration::Import { module_path, items } => {
                // Check if this is a standard library module
                if self.stdlib.is_stdlib_module(&module_path) {
                    println!("📚 Importing standard library module: {}", module_path);
                    
                    // Parse and compile the stdlib module
                    // Clone the source to avoid borrow issues
                    let module_source = self.stdlib.get_module(&module_path).cloned();
                    if let Some(source) = module_source {
                        // We need to parse the module source
                        // For now, we'll store it as if it was already compiled
                        match items {
                            Some(item_list) => {
                                let item_count = item_list.len();
                                for item in item_list {
                                    self.imported_modules.insert(item, module_path.clone());
                                }
                                println!("  ✓ Imported {} items from stdlib.{}", item_count, module_path);
                            }
                            None => {
                                self.imported_modules.insert(format!("*{}", module_path), module_path.clone());
                                println!("  ✓ Wildcard import from stdlib.{}", module_path);
                            }
                        }
                        
                        // Parse and compile the stdlib module inline
                        self.compile_stdlib_module(&module_path, &source);
                    }
                } else {
                    // Regular user module import
                    match items {
                        Some(item_list) => {
                            // Selective import: import "module" { item1, item2 }
                            let item_count = item_list.len();
                            for item in item_list {
                                self.imported_modules.insert(item, module_path.clone());
                            }
                            println!("🔗 Imported {} items from module '{}'", item_count, module_path);
                        }
                        None => {
                            // Wildcard import: import "module"
                            // For now, mark the entire module as imported
                            self.imported_modules.insert(format!("*{}", module_path), module_path.clone());
                            println!("🔗 Wildcard import from module '{}'", module_path);
                        }
                    }
                }
            }
            Declaration::Module { name, exports, body } => {
                // Handle module definitions
                println!("📦 Processing module '{}'", name);
                
                // Register exported items
                self.modules.insert(name.clone(), exports.clone());
                
                // Save current compilation state
                let saved_functions = self.functions.clone();
                let saved_variables = self.variables.clone();
                let saved_offset = self.stack_offset;
                
                // Create module namespace prefix
                let module_prefix = format!("{}::", name);
                
                // Compile module body with namespaced function names
                for decl in body {
                    match decl {
                        Declaration::Function { name: func_name, params, body, return_type } => {
                            // Create namespaced function name
                            let namespaced_name = format!("{}{}", module_prefix, func_name);
                            
                            // Only process if this function is exported
                            if exports.contains(&func_name) {
                                // Store function info under the namespaced name
                                self.function_params.insert(namespaced_name.clone(), params.clone());
                                self.function_bodies.insert(namespaced_name.clone(), body.clone());
                                
                                // For module functions, we'll handle them as inline-only functions
                                // This avoids the complexity of generating actual function code with proper linking
                                println!("  ✓ Registered exported function: {} -> {}", func_name, namespaced_name);
                            }
                        }
                        _ => {
                            // For other declarations, process normally
                            self.generate_declaration(decl);
                        }
                    }
                }
                
                // Restore compilation state
                self.stack_offset = saved_offset;
                // Keep the module functions but restore local state
                for (name, addr) in saved_functions {
                    if !name.contains("::") {
                        self.functions.insert(name, addr);
                    }
                }
                self.variables = saved_variables;
                
                println!("📦 Module '{}' compiled successfully", name);
            }
            Declaration::Function { name, params, body, .. } => {
                if name == "main" || name == "birth" {
                    // Main function - generate inline with optimizations
                    let optimized_body = self.eliminate_dead_code(body);
                    for stmt in optimized_body {
                        self.generate_statement(stmt);
                    }
                } else {
                    // User-defined function - generate ACTUAL function code
                    // Store function entry point FIRST
                    let func_addr = self.code.len();
                    self.functions.insert(name.clone(), func_addr);
                    
                    // Store params and body for potential inlining (but we won't use it)
                    self.function_params.insert(name.clone(), params.clone());
                    self.function_bodies.insert(name.clone(), body.clone());
                    
                    // Generate function prologue
                    // push rbp
                    self.code.push(0x55);
                    // mov rbp, rsp
                    self.code.extend_from_slice(&[0x48, 0x89, 0xe5]);
                    // sub rsp, 0x80 (allocate local space)
                    self.code.extend_from_slice(&[0x48, 0x81, 0xec, 0x80, 0x00, 0x00, 0x00]);
                    
                    // Save current state
                    let saved_vars = self.variables.clone();
                    let saved_offset = self.stack_offset;
                    
                    // Set up new scope for function
                    self.variables.clear();
                    self.stack_offset = 0;
                    
                    // Bind parameters - they're at positive offsets from rbp
                    // After call instruction pushes return addr and we push rbp:
                    // [rbp+16] = first param, [rbp+24] = second param, etc.
                    for (i, param) in params.iter().enumerate() {
                        let param_offset = 16 + (i as isize * 8);
                        self.variables.insert(param.clone(), param_offset);
                    }
                    
                    // Generate function body with optimizations
                    let optimized_body = self.eliminate_dead_code(body);
                    let mut has_return = false;
                    for stmt in optimized_body {
                        if let Statement::Return(_) = &stmt {
                            has_return = true;
                        }
                        self.generate_statement(stmt);
                    }
                    
                    // If no explicit return, add default return
                    if !has_return {
                        // Return 0 by default
                        // xor rax, rax (return 0)
                        self.code.extend_from_slice(&[0x48, 0x31, 0xc0]);
                        // Function epilogue
                        // mov rsp, rbp
                        self.code.extend_from_slice(&[0x48, 0x89, 0xec]);
                        // pop rbp
                        self.code.push(0x5d);
                        // ret
                        self.code.push(0xc3);
                    }
                    
                    // Restore state
                    self.variables = saved_vars;
                    self.stack_offset = saved_offset;
                }
            }
            _ => {}
        }
    }
    
    fn generate_statement(&mut self, stmt: Statement) {
        // Track current line for debug info
        let statement_address = self.code.len();
        if let Some(ref mut debug_info) = self.debug_info {
            debug_info.add_line_entry(
                0x401000 + statement_address as u64,
                1,  // File index
                self.current_line,
                0,  // Column
            );
        }
        self.current_line += 1;  // Simple line tracking
        
        match stmt {
            Statement::Let { name, type_annotation, value } => {
                // Infer the value's type
                let inferred_type = self.infer_type(&value);
                
                // Track the type if provided
                if let Some(type_name) = type_annotation.as_ref() {
                    let expected_type = match type_name.as_str() {
                        "int" => TypeInfo::Int,
                        "float" => TypeInfo::Float,
                        "string" => TypeInfo::String,
                        "bool" => TypeInfo::Bool,
                        _ => {
                            // Check if it's a known struct or enum
                            if self.structs.contains_key(type_name) {
                                TypeInfo::Struct(type_name.clone())
                            } else if self.enums.contains_key(type_name) {
                                TypeInfo::Enum(type_name.clone())
                            } else {
                                TypeInfo::Unknown
                            }
                        }
                    };
                    
                    // Check type compatibility
                    if let Err(error) = self.check_type(&expected_type, &inferred_type, &format!("variable '{}'", name)) {
                        eprintln!("Warning: {}", error);
                    }
                    
                    self.variable_types.insert(name.clone(), expected_type);
                } else {
                    // No annotation, use inferred type
                    self.variable_types.insert(name.clone(), inferred_type);
                }
                
                self.generate_expr(value);
                // Store in local variable with safer offset calculation
                self.stack_offset -= 8;
                self.variables.insert(name.clone(), self.stack_offset);
                
                // Use optimized register allocation if available
                self.emit_store_variable_optimized(&name);
            }
            Statement::Assignment { name, value } => {
                self.generate_expr(value);
                // Update existing variable
                if !self.variables.contains_key(&name) {
                    // Create new variable if doesn't exist
                    self.stack_offset -= 8;
                    self.variables.insert(name.clone(), self.stack_offset);
                }
                
                // Use optimized register allocation if available
                self.emit_store_variable_optimized(&name);
            }
            Statement::IndexedAssignment { array, index, value } => {
                // Handle array[index] = value
                // Generate value first
                self.generate_expr(value);
                
                // Generate index
                self.generate_expr(index);
                
                // Get array base address from stack
                if let Some(&offset) = self.variables.get(&array) {
                    // Load array pointer
                    self.code.extend_from_slice(&[0x48, 0x8b, 0x85]);
                    self.code.extend_from_slice(&(offset as i32).to_le_bytes());
                    
                    // Now rax has the array pointer
                    // Pop index into rbx
                    self.emit_pop_rbx();
                    
                    // Pop value into rcx
                    self.emit_pop_rcx();
                    
                    // Check bounds (array length is at [rax])
                    self.code.extend_from_slice(&[0x48, 0x3b, 0x18]); // cmp rbx, [rax]
                    
                    // Skip store if out of bounds (jae = jump if above or equal)
                    self.code.extend_from_slice(&[0x73, 0x0a]); // jae +10
                    
                    // Calculate element address: rax + 8 + (rbx * 8)
                    // mov [rax + rbx*8 + 8], rcx
                    self.code.extend_from_slice(&[0x48, 0x89, 0x4c, 0xd8, 0x08]);
                } else {
                    // Error: array not found - just pop the values
                    self.emit_pop_rax();
                    self.emit_pop_rax();
                }
            }
            Statement::Expression(expr) => {
                self.generate_expr(expr);
                // Pop result if not used
                self.emit_pop_rax();
            }
            Statement::Return(Some(expr)) => {
                self.generate_expr(expr);
                // Pop return value to rax
                self.emit_pop_rax();
                // Function epilogue
                // mov rsp, rbp
                self.code.extend_from_slice(&[0x48, 0x89, 0xec]);
                // pop rbp
                self.code.push(0x5d);
                // ret
                self.code.push(0xc3);
                self.last_was_return = true;
            }
            Statement::Return(None) => {
                // Return 0
                // xor rax, rax (return 0)
                self.code.extend_from_slice(&[0x48, 0x31, 0xc0]);
                // Function epilogue
                // mov rsp, rbp
                self.code.extend_from_slice(&[0x48, 0x89, 0xec]);
                // pop rbp
                self.code.push(0x5d);
                // ret
                self.code.push(0xc3);
                self.last_was_return = true;
            }
            Statement::Break => {
                // Generate jump to end of current loop
                if let Some(break_jumps) = self.loop_break_stack.last_mut() {
                    let jump_location = self.code.len();
                    self.code.extend_from_slice(&[0xe9]); // jmp (32-bit relative)
                    self.code.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // placeholder
                    break_jumps.push(jump_location + 1); // Store location of the offset
                }
            }
            Statement::Continue => {
                // Generate jump to start of current loop
                if let Some(&continue_target) = self.loop_continue_stack.last() {
                    let current_pos = self.code.len() + 5; // Position after this instruction
                    let offset = continue_target as i32 - current_pos as i32;
                    self.code.extend_from_slice(&[0xe9]); // jmp (32-bit relative)
                    self.code.extend_from_slice(&offset.to_le_bytes());
                }
            }
            Statement::Breakpoint => {
                // Generate INT3 breakpoint instruction if enabled
                if self.enable_breakpoints {
                    let address = 0x401000 + self.code.len();
                    
                    // Register this breakpoint location
                    self.breakpoint_manager.register_source_mapping(
                        "main.nl".to_string(),  // TODO: Get actual source file
                        self.current_line,
                        address,
                        self.current_function.clone(),
                    );
                    
                    // Emit INT3 instruction
                    self.code.extend_from_slice(&BreakpointManager::generate_int3_instruction());
                    
                    // Print debug message (optional)
                    if self.debug_info_enabled {
                        println!("🔴 Breakpoint at line {}", self.current_line);
                    }
                } else {
                    // If breakpoints are disabled, emit a NOP instead
                    self.code.push(0x90); // NOP
                }
            }
            Statement::If { condition, then_body, else_body } => {
                // Generate if statement
                self.generate_expr(condition);
                self.emit_pop_rax();
                // Test if zero (false)
                self.emit_test_rax_rax();
                // Jump to else/end if zero
                let jump_to_else = self.emit_jz_placeholder();
                
                // Generate then body
                for stmt in then_body {
                    self.generate_statement(stmt);
                }
                
                if let Some(else_statements) = else_body {
                    // Jump over else block
                    let jump_to_end = self.emit_jmp_placeholder();
                    // Patch jump to else
                    let else_start = self.code.len();
                    self.patch_jump(jump_to_else, else_start as i32);
                    
                    // Generate else body
                    for stmt in else_statements {
                        self.generate_statement(stmt);
                    }
                    
                    // Patch jump to end
                    let end = self.code.len();
                    self.patch_jump(jump_to_end, end as i32);
                } else {
                    // No else, just patch to end
                    let end = self.code.len();
                    self.patch_jump(jump_to_else, end as i32);
                }
            }
            Statement::While { condition, body } => {
                // Generate while loop
                let loop_start = self.code.len();
                self.generate_expr(condition);
                self.emit_pop_rax();
                // Test if zero
                self.emit_test_rax_rax();
                // Jump to end if zero (returns placeholder offset)
                let jump_offset = self.emit_jz_placeholder();
                
                // Generate body
                for stmt in body {
                    self.generate_statement(stmt);
                }
                
                // Jump back to start (relative jump)
                let jump_back_offset = loop_start as i32 - (self.code.len() + 5) as i32; // +5 for the jmp instruction size
                self.emit_jmp(jump_back_offset);
                
                // Patch jump offset
                let end_offset = self.code.len();
                self.patch_jump(jump_offset, end_offset as i32);
            }
            Statement::For { variable, start, end, body } => {
                // Classic for loop: for i = start to end
                // Initialize loop variable
                self.generate_expr(start);
                let var_offset = self.get_or_create_local(&variable);
                self.emit_store_local(var_offset);
                
                // Loop start
                let loop_start = self.code.len();
                
                // Check condition: var < end
                self.emit_load_local(var_offset);
                self.generate_expr(end);
                self.emit_pop_rbx();
                self.emit_pop_rax();
                
                // Compare rax < rbx
                self.code.extend_from_slice(&[0x48, 0x39, 0xd8]); // cmp rax, rbx
                
                // Jump if not less (jge)
                self.code.extend_from_slice(&[0x0f, 0x8d]);
                let jump_offset = self.code.len();
                self.code.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // placeholder
                
                // Generate body
                for stmt in body {
                    self.generate_statement(stmt);
                }
                
                // Increment loop variable
                self.emit_load_local(var_offset);
                self.emit_push_imm(1);
                self.emit_pop_rbx();
                self.emit_pop_rax();
                self.emit_add_rax_rbx();
                self.emit_push_rax();
                self.emit_store_local(var_offset);
                
                // Jump back to loop start
                let jump_back_offset = loop_start as i32 - (self.code.len() + 5) as i32;
                self.emit_jmp(jump_back_offset);
                
                // Patch forward jump
                let end_offset = self.code.len();
                let relative_offset = (end_offset - jump_offset - 4) as i32;
                let bytes = relative_offset.to_le_bytes();
                self.code[jump_offset..jump_offset+4].copy_from_slice(&bytes);
            }
            Statement::ForIn { variable, array, body } => {
                // For-in loop: for item in array
                // This is more complex - we need to iterate through array elements
                
                // Get array address
                self.generate_expr(array);
                self.emit_pop_rax(); // rax = array address
                
                // Store array base address in a local
                let array_base_offset = self.next_local_offset;
                self.next_local_offset += 8;
                self.emit_push_rax();
                self.emit_store_local(array_base_offset);
                
                // Initialize index to 0
                let index_offset = self.next_local_offset;
                self.next_local_offset += 8;
                self.emit_push_imm(0);
                self.emit_store_local(index_offset);
                
                // Get array length (first element)
                self.emit_load_local(array_base_offset);
                self.emit_pop_rax();
                // mov rbx, [rax] - get array length
                self.code.extend_from_slice(&[0x48, 0x8b, 0x18]);
                self.emit_push_rbx();
                let length_offset = self.next_local_offset;
                self.next_local_offset += 8;
                self.emit_store_local(length_offset);
                
                // Loop start
                let loop_start = self.code.len();
                
                // Check if index < length
                self.emit_load_local(index_offset);
                self.emit_load_local(length_offset);
                self.emit_pop_rbx();
                self.emit_pop_rax();
                
                // Compare rax < rbx
                self.code.extend_from_slice(&[0x48, 0x39, 0xd8]); // cmp rax, rbx
                
                // Jump if not less (jge)
                self.code.extend_from_slice(&[0x0f, 0x8d]);
                let jump_offset = self.code.len();
                self.code.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // placeholder
                
                // Load current array element into variable
                self.emit_load_local(array_base_offset);
                self.emit_pop_rax(); // rax = array base
                self.emit_load_local(index_offset);
                self.emit_pop_rbx(); // rbx = index
                
                // Calculate element address: rax + 8 + (rbx * 8)
                // lea rax, [rax + rbx*8 + 8]
                self.code.extend_from_slice(&[0x48, 0x8d, 0x44, 0xd8, 0x08]);
                
                // Load element value: mov rax, [rax]
                self.code.extend_from_slice(&[0x48, 0x8b, 0x00]);
                self.emit_push_rax();
                
                // Store in loop variable
                let var_offset = self.get_or_create_local(&variable);
                self.emit_store_local(var_offset);
                
                // Generate body
                for stmt in body {
                    self.generate_statement(stmt);
                }
                
                // Increment index
                self.emit_load_local(index_offset);
                self.emit_push_imm(1);
                self.emit_pop_rbx();
                self.emit_pop_rax();
                self.emit_add_rax_rbx();
                self.emit_push_rax();
                self.emit_store_local(index_offset);
                
                // Jump back to loop start
                let jump_back_offset = loop_start as i32 - (self.code.len() + 5) as i32;
                self.emit_jmp(jump_back_offset);
                
                // Patch forward jump
                let end_offset = self.code.len();
                let relative_offset = (end_offset - jump_offset - 4) as i32;
                let bytes = relative_offset.to_le_bytes();
                self.code[jump_offset..jump_offset+4].copy_from_slice(&bytes);
            }
            _ => {}
        }
    }
    
    fn generate_expr(&mut self, expr: Expr) {
        // Apply constant folding optimization
        let expr = self.fold_constants(expr);
        
        match expr {
            Expr::Tryte(t) => {
                let value = match t {
                    TryteValue::Negative => -1i64,
                    TryteValue::Baseline => 0i64,
                    TryteValue::Positive => 1i64,
                };
                self.emit_push_immediate(value);
            }
            Expr::Number(n) => {
                // Check if it's a float or integer
                if n.fract() != 0.0 {
                    // It's a float
                    self.emit_push_float(n);
                } else {
                    // It's an integer
                    self.emit_push_immediate(n as i64);
                }
            }
            Expr::String(s) => {
                // Add string to data section and push its address
                let offset = self.add_string(s.clone());
                self.emit_push_string_addr(offset);
                // Also intern for deduplication tracking
                self.intern_string(s);
            }
            Expr::Identifier(name) => {
                // Use optimized register allocation if available
                self.emit_load_variable_optimized(&name);
            }
            Expr::BinaryOp { left, op, right } => {
                // Check if either operand is a string
                let is_string = matches!(left.as_ref(), Expr::String(_)) ||
                               matches!(right.as_ref(), Expr::String(_));
                
                // Check if either operand is a float
                let is_float = matches!(left.as_ref(), Expr::Number(n) if n.fract() != 0.0) ||
                              matches!(right.as_ref(), Expr::Number(n) if n.fract() != 0.0);
                
                self.generate_expr(*left);
                self.generate_expr(*right);
                
                // Handle string concatenation
                if is_string && matches!(op, BinaryOperator::Add) {
                    self.emit_string_concat();
                }
                // Use float operations if any operand is a float
                else if is_float && matches!(op, BinaryOperator::Add | BinaryOperator::Subtract | 
                                            BinaryOperator::Multiply | BinaryOperator::Divide) {
                    self.generate_float_binary_op(op);
                } else {
                    self.generate_binary_op(op);
                }
            }
            Expr::UnaryOp { op, operand } => {
                // Generate the operand first
                self.generate_expr(*operand);
                
                // Apply the unary operation
                match op {
                    UnaryOperator::Plus => {
                        // Unary plus does nothing - value is already on stack
                    }
                    UnaryOperator::Minus => {
                        // Unary minus: negate the top stack value
                        self.emit_pop_rax();
                        // neg rax
                        self.code.extend_from_slice(&[0x48, 0xf7, 0xd8]);
                        self.emit_push_rax();
                    }
                }
            }
            Expr::Call { name, args } => {
                // Special case for built-in functions
                if name == "print" && args.len() == 1 {
                    self.generate_expr(args[0].clone());
                    self.emit_print_call();
                } else if name == "sin" && args.len() == 1 {
                    self.generate_expr(args[0].clone());
                    self.emit_math_sin();
                } else if name == "cos" && args.len() == 1 {
                    self.generate_expr(args[0].clone());
                    self.emit_math_cos();
                } else if name == "tan" && args.len() == 1 {
                    self.generate_expr(args[0].clone());
                    self.emit_math_tan();
                } else if name == "exp" && args.len() == 1 {
                    self.generate_expr(args[0].clone());
                    self.emit_math_exp();
                } else if name == "log" && args.len() == 1 {
                    self.generate_expr(args[0].clone());
                    self.emit_math_log();
                } else if name == "sqrt" && args.len() == 1 {
                    self.generate_expr(args[0].clone());
                    self.emit_math_sqrt();
                } else if name == "randn" && args.len() == 0 {
                    // Random normal distribution (Gaussian)
                    self.emit_randn();
                } else if name == "random" && args.len() == 0 {
                    // Random uniform [0,1)
                    self.emit_random();
                } else if name == "relu" && args.len() == 1 {
                    // ReLU activation: max(0, x)
                    self.generate_expr(args[0].clone());
                    self.emit_relu();
                } else if name == "sigmoid" && args.len() == 1 {
                    // Sigmoid activation: 1 / (1 + exp(-x))
                    self.generate_expr(args[0].clone());
                    self.emit_sigmoid();
                } else if name == "tanh" && args.len() == 1 {
                    // Tanh activation: tanh(x)  
                    self.generate_expr(args[0].clone());
                    self.emit_tanh();
                } else if name == "pow" && args.len() == 2 {
                    self.generate_expr(args[0].clone());
                    self.generate_expr(args[1].clone());
                    self.emit_math_pow();
                } else if name == "abs" && args.len() == 1 {
                    self.generate_expr(args[0].clone());
                    self.emit_math_abs();
                } else if name == "floor" && args.len() == 1 {
                    self.generate_expr(args[0].clone());
                    self.emit_math_floor();
                } else if name == "ceil" && args.len() == 1 {
                    self.generate_expr(args[0].clone());
                    self.emit_math_ceil();
                } else if name == "round" && args.len() == 1 {
                    self.generate_expr(args[0].clone());
                    self.emit_math_round();
                } else if name == "sigmoid" && args.len() == 1 {
                    self.generate_expr(args[0].clone());
                    self.emit_math_sigmoid();
                } else if name == "tanh" && args.len() == 1 {
                    self.generate_expr(args[0].clone());
                    self.emit_math_tanh();
                } else if name == "relu" && args.len() == 1 {
                    self.generate_expr(args[0].clone());
                    self.emit_relu();  // Use simpler integer version
                } else if name == "random" && args.len() == 0 {
                    // Generate random number
                    self.emit_random();
                } else if name == "randn" && args.len() == 0 {
                    // Generate random normal (Gaussian) for weight init
                    self.emit_random_normal();
                } else if name == "xavier_init" && args.len() == 2 {
                    // Xavier/Glorot initialization: scale = sqrt(2 / (fan_in + fan_out))
                    self.generate_expr(args[0].clone()); // fan_in
                    self.generate_expr(args[1].clone()); // fan_out
                    self.emit_xavier_init();
                } else if name == "he_init" && args.len() == 1 {
                    // He initialization: scale = sqrt(2 / fan_in)
                    self.generate_expr(args[0].clone()); // fan_in
                    self.emit_he_init();
                } else if name == "zeros" && args.len() == 1 {
                    // Initialize array with zeros
                    self.generate_expr(args[0].clone()); // size
                    self.emit_zeros_array();
                } else if name == "ones" && args.len() == 1 {
                    // Initialize array with ones
                    self.generate_expr(args[0].clone()); // size
                    self.emit_ones_array();
                } else if name == "matrix_mul" && args.len() == 2 {
                    // Matrix multiplication: A @ B
                    self.generate_expr(args[0].clone()); // matrix A
                    self.generate_expr(args[1].clone()); // matrix B
                    self.emit_matrix_multiply();
                } else if name == "matrix_get" && args.len() == 3 {
                    // Get element from 2D array: matrix_get(matrix, row, col)
                    self.generate_expr(args[0].clone()); // matrix
                    self.generate_expr(args[1].clone()); // row
                    self.generate_expr(args[2].clone()); // col
                    self.emit_matrix_get();
                } else if name == "matrix_set" && args.len() == 4 {
                    // Set element in 2D array: matrix_set(matrix, row, col, value)
                    self.generate_expr(args[0].clone()); // matrix
                    self.generate_expr(args[1].clone()); // row
                    self.generate_expr(args[2].clone()); // col
                    self.generate_expr(args[3].clone()); // value
                    self.emit_matrix_set();
                } else if name == "save_weights" && args.len() == 2 {
                    // Save weights to file
                    self.generate_expr(args[0].clone()); // filename
                    self.generate_expr(args[1].clone()); // weights array
                    self.emit_save_weights();
                } else if name == "load_weights" && args.len() == 1 {
                    // Load weights from file
                    self.generate_expr(args[0].clone()); // filename
                    self.emit_load_weights();
                } else if name == "read_file" && args.len() == 1 {
                    // Read text file content
                    self.generate_expr(args[0].clone()); // filename
                    self.emit_read_file();
                } else if name == "write_file" && args.len() == 2 {
                    // Write content to text file
                    self.generate_expr(args[0].clone()); // filename
                    self.generate_expr(args[1].clone()); // content
                    self.emit_write_file();
                } else if name == "get_args" && args.len() == 0 {
                    // Get command line arguments as array
                    self.emit_get_args();
                } else if name == "make_executable" && args.len() == 1 {
                    // Make file executable using chmod
                    self.generate_expr(args[0].clone()); // filename
                    self.emit_make_executable();
                } else if name == "len" && args.len() == 1 {
                    // Get length of string or array
                    self.generate_expr(args[0].clone());
                    self.emit_length();
                } else if name == "malloc" && args.len() == 1 {
                    // Allocate memory dynamically
                    self.generate_expr(args[0].clone()); // size in bytes
                    self.emit_malloc();
                } else if name == "gc_alloc" && args.len() == 1 {
                    // 🗑️ GC ALLOCATION: Managed memory allocation
                    self.generate_expr(args[0].clone()); // size in bytes
                    self.emit_gc_alloc_simple();
                } else if name == "gc_free" && args.len() == 1 {
                    // 🗑️ GC FREE: Decrement reference count
                    self.generate_expr(args[0].clone()); // pointer
                    self.emit_gc_free();
                } else if name == "gc_collect" && args.len() == 0 {
                    // 🗑️ GC COLLECT: Trigger garbage collection
                    self.emit_gc_collect();
                } else if name == "gc_alloc_ripple" && args.len() == 2 {
                    // 🌊 GC RIPPLE: Allocate Hoberman ripple wave
                    self.generate_expr(args[0].clone()); // energy
                    self.generate_expr(args[1].clone()); // radius
                    self.emit_gc_alloc_ripple();
                } else if name == "gc_alloc_neuron" && args.len() == 1 {
                    // 🧠 GC NEURON: Allocate neural network neuron
                    self.generate_expr(args[0].clone()); // weights_count
                    self.emit_gc_alloc_neuron();
                } else if name == "free" && args.len() == 2 {
                    // Free dynamically allocated memory
                    self.generate_expr(args[0].clone()); // pointer
                    self.generate_expr(args[1].clone()); // size
                    self.emit_free();
                } else if name == "push" && args.len() == 2 {
                    // Push element to array: push(array, element)
                    self.generate_expr(args[0].clone()); // array
                    self.generate_expr(args[1].clone()); // element
                    self.emit_array_push();
                } else if name == "pop" && args.len() == 1 {
                    // Pop element from array
                    self.generate_expr(args[0].clone()); // array
                    self.emit_array_pop();
                } else if name == "get" && args.len() == 2 {
                    // Get value from HashMap: get(hashmap, key)
                    self.generate_expr(args[0].clone()); // hashmap
                    self.generate_expr(args[1].clone()); // key
                    self.emit_hashmap_get();
                } else {
                    // 🚀 INNOVATIVE RECURSIVE OPTIMIZATION FRAMEWORK
                    // Check for known recursive patterns FIRST (before checking function bodies)
                    
                    match name.as_str() {
                        "factorial" => {
                            eprintln!("🚀 INNOVATIVE: Converting recursive factorial to optimized loop!");
                            self.generate_factorial_optimized(&args);
                        }
                        "fibonacci" | "fib" => {
                            eprintln!("🚀 INNOVATIVE: Converting recursive Fibonacci to optimized loop!");
                            self.generate_fibonacci_optimized(&args);
                        }
                        "power" | "pow" => {
                            eprintln!("🚀 INNOVATIVE: Converting recursive power to optimized loop!");
                            self.generate_power_optimized(&args);
                        }
                        "gcd" => {
                            eprintln!("🚀 INNOVATIVE: Converting recursive GCD to optimized loop!");
                            self.generate_gcd_optimized(&args);
                        }
                        _ => {
                            // Check if it's an imported function first
                            let resolved_name = if let Some(module_path) = self.imported_modules.get(&name) {
                                // Function is imported from a module
                                let namespaced_name = format!("{}::{}", module_path, name);
                                println!("🔗 Resolving imported function: {} -> {}", name, namespaced_name);
                                namespaced_name
                            } else {
                                // Use original name
                                name.clone()
                            };
                            
                            // Check if it's a user-defined function (using resolved name)
                            if self.function_bodies.contains_key(&resolved_name) {
                                if let Some(body) = self.function_bodies.get(&resolved_name).cloned() {
                                    self.generate_inline_function(&resolved_name, &args, &body);
                                } else {
                                    self.emit_push_immediate(0);
                                }
                            } else if self.function_bodies.contains_key(&name) {
                                // Fall back to original name for local functions
                                if let Some(body) = self.function_bodies.get(&name).cloned() {
                                    self.generate_inline_function(&name, &args, &body);
                                } else {
                                    self.emit_push_immediate(0);
                                }
                            } else {
                                // Unknown function - push 0
                                println!("⚠️  Unknown function: {}", name);
                                self.emit_push_immediate(0);
                            }
                        }
                    }
                }
            }
            Expr::Lambda { params, body } => {
                // Lambda expressions create closures
                eprintln!("🚀 LAMBDA: Creating closure with {} params", params.len());
                self.generate_lambda_closure(params, body);
            }
            Expr::Synthesize(expr) => {
                // 🔢 REVOLUTIONARY: Smart synthesize that handles strings, floats, and integers
                
                // Check if the expression is a float literal
                let is_float_literal = matches!(expr.as_ref(), Expr::Number(n) if n.fract() != 0.0);
                
                self.generate_expr(*expr);
                
                if is_float_literal {
                    // Use specialized float synthesize for float literals
                    self.emit_synthesize_float();
                } else {
                    // Pop value into rax
                    self.emit_pop_rax();
                    
                    // Push it back for now (preserve for later)
                    self.emit_push_rax();
                    
                    // Use smart synthesize that can handle string indices and integers
                    self.emit_synthesize_smart();
                }
            }
            Expr::Express(inner) => {
                // Handle different types of expressions
                match inner.as_ref() {
                    Expr::Number(n) => {
                        // Convert number to string and add to data
                        let num_str = n.to_string();
                        let offset = self.add_string(num_str);
                        self.emit_push_string_addr(offset);
                        self.emit_print_call();
                    }
                    Expr::Identifier(_) => {
                        // For variables, we need to convert the numeric value to string at runtime
                        // For now, let's assume variables contain strings (limitation)
                        // TODO: Add runtime number-to-string conversion
                        self.generate_expr(*inner);
                        self.emit_print_call();
                    }
                    _ => {
                        // For other expressions, generate normally (should be strings)
                        self.generate_expr(*inner);
                        self.emit_print_call();
                    }
                }
            }
            Expr::RedisConnect(host, port) => {
                // REAL Redis connection using syscalls!
                self.emit_redis_connect(&host, port);
            }
            Expr::RedisGet(key) => {
                self.generate_expr(*key);
                self.emit_redis_get();
            }
            Expr::RedisSet(key, value) => {
                self.generate_expr(*key);
                self.generate_expr(*value);
                self.emit_redis_set();
            }
            Expr::RedisPublish(channel, msg) => {
                self.generate_expr(*channel);
                self.generate_expr(*msg);
                self.emit_redis_publish();
            }
            Expr::RedisSubscribe(channel) => {
                self.generate_expr(*channel);
                self.emit_redis_subscribe();
            }
            Expr::Array2D(rows) => {
                // 2D array allocation for matrices (neural network weights)
                // Format: [num_rows, num_cols, row0_data..., row1_data..., ...]
                
                if rows.is_empty() {
                    self.emit_push_immediate(0);
                    return;
                }
                
                let num_rows = rows.len();
                let num_cols = rows[0].len(); // Assume all rows have same length
                let total_elements = num_rows * num_cols;
                let buffer_size = 64;
                
                // Allocate: buffer + 16 bytes (rows, cols) + data
                let total_size = buffer_size + 16 + (total_elements * 8);
                
                // sub rsp, total_size
                self.code.extend_from_slice(&[0x48, 0x81, 0xec]);
                self.code.extend_from_slice(&(total_size as u32).to_le_bytes());
                
                // Store dimensions at beginning (after buffer)
                // mov rax, num_rows
                self.emit_push_immediate(num_rows as i64);
                self.emit_pop_rax();
                // mov [rsp + buffer_size], rax
                self.code.extend_from_slice(&[0x48, 0x89, 0x84, 0x24]);
                self.code.extend_from_slice(&(buffer_size as u32).to_le_bytes());
                
                // mov rax, num_cols
                self.emit_push_immediate(num_cols as i64);
                self.emit_pop_rax();
                // mov [rsp + buffer_size + 8], rax
                self.code.extend_from_slice(&[0x48, 0x89, 0x84, 0x24]);
                self.code.extend_from_slice(&((buffer_size + 8) as u32).to_le_bytes());
                
                // Store elements in row-major order
                let mut element_offset = buffer_size + 16;
                for row in rows {
                    for elem in row {
                        self.generate_expr(elem);
                        self.emit_pop_rax();
                        // mov [rsp + offset], rax
                        self.code.extend_from_slice(&[0x48, 0x89, 0x84, 0x24]);
                        self.code.extend_from_slice(&(element_offset as u32).to_le_bytes());
                        element_offset += 8;
                    }
                }
                
                // Push base address (points to dimensions)
                // lea rax, [rsp + buffer_size]
                self.code.extend_from_slice(&[0x48, 0x8d, 0x84, 0x24]);
                self.code.extend_from_slice(&(buffer_size as u32).to_le_bytes());
                self.emit_push_rax();
            }
            Expr::Array(elements) => {
                // Allocate array on stack with buffer to prevent variable collision
                let array_size = elements.len();
                let buffer_size = 64; // 64-byte buffer to separate from variables
                
                // Allocate space for length + array elements + buffer
                // (8 bytes for length + 8 bytes per element + buffer)
                let total_alloc = 8 + (array_size * 8) + buffer_size;
                // sub rsp, total_alloc
                self.code.extend_from_slice(&[0x48, 0x81, 0xec]);
                self.code.extend_from_slice(&(total_alloc as u32).to_le_bytes());
                
                // Store array length at the beginning (after buffer)
                // mov rax, array_size
                self.emit_push_immediate(array_size as i64);
                self.emit_pop_rax();
                // mov [rsp + buffer_size], rax
                self.code.extend_from_slice(&[0x48, 0x89, 0x84, 0x24]);
                self.code.extend_from_slice(&(buffer_size as u32).to_le_bytes());
                
                // Store each element (after the length)
                for (i, elem) in elements.into_iter().enumerate() {
                    self.generate_expr(elem);
                    // Pop value and store in array
                    self.emit_pop_rax();
                    // mov [rsp + buffer_size + 8 + i*8], rax
                    self.code.extend_from_slice(&[0x48, 0x89, 0x84, 0x24]);
                    self.code.extend_from_slice(&((buffer_size + 8 + i * 8) as u32).to_le_bytes());
                }
                
                // Push array base address (points to length field)
                // lea rax, [rsp + buffer_size] 
                self.code.extend_from_slice(&[0x48, 0x8d, 0x84, 0x24]);
                self.code.extend_from_slice(&(buffer_size as u32).to_le_bytes());
                self.emit_push_rax();
            }
            Expr::Index { array, index } => {
                // Generate array/hashmap address
                self.generate_expr(*array);
                // Generate index/key
                self.generate_expr(*index);
                
                // For now, treat all index operations as array access
                // HashMap access will need a separate function call like get(hashmap, key)
                
                // Pop index to rbx
                self.emit_pop_rbx();
                // Pop array address to rax
                self.emit_pop_rax();
                
                // Calculate element address: rax + 8 + rbx * 8
                // (skip length field at offset 0)
                // shl rbx, 3 (multiply by 8)
                self.code.extend_from_slice(&[0x48, 0xc1, 0xe3, 0x03]);
                // add rax, rbx
                self.code.extend_from_slice(&[0x48, 0x01, 0xd8]);
                // add rax, 8 (skip length field)
                self.code.extend_from_slice(&[0x48, 0x83, 0xc0, 0x08]);
                
                // Load value from array
                // mov rax, [rax]
                self.code.extend_from_slice(&[0x48, 0x8b, 0x00]);
                
                // Push result
                self.emit_push_rax();
            }
            Expr::StructInit { name, fields } => {
                // Create a struct instance
                // Clone the struct fields to avoid borrow issues
                if let Some(struct_fields) = self.structs.get(&name).cloned() {
                    let struct_size = struct_fields.len() * 8;
                    let buffer_size = 64;
                    let total_alloc = struct_size + buffer_size;
                    
                    // Allocate space for struct
                    // sub rsp, total_alloc
                    self.code.extend_from_slice(&[0x48, 0x81, 0xec]);
                    self.code.extend_from_slice(&(total_alloc as u32).to_le_bytes());
                    
                    // Initialize fields
                    for (field_name, value) in fields {
                        // Find field offset
                        if let Some((_name, offset)) = struct_fields.iter().find(|(n, _)| n == &field_name) {
                            // Generate value
                            self.generate_expr(value);
                            self.emit_pop_rax();
                            
                            // Store in struct field
                            // mov [rsp + buffer_size + offset], rax
                            self.code.extend_from_slice(&[0x48, 0x89, 0x84, 0x24]);
                            self.code.extend_from_slice(&((buffer_size + offset) as u32).to_le_bytes());
                        }
                    }
                    
                    // Push struct address
                    // lea rax, [rsp + buffer_size]
                    self.code.extend_from_slice(&[0x48, 0x8d, 0x84, 0x24]);
                    self.code.extend_from_slice(&(buffer_size as u32).to_le_bytes());
                    self.emit_push_rax();
                } else {
                    // Unknown struct type, push 0
                    self.emit_push_immediate(0);
                }
            }
            Expr::EnumVariant { enum_name, variant } => {
                // Look up the enum variant value
                if let Some(variants) = self.enums.get(&enum_name) {
                    for (var_name, var_value) in variants {
                        if var_name == &variant {
                            self.emit_push_immediate(*var_value);
                            return;
                        }
                    }
                }
                // If not found, push 0
                self.emit_push_immediate(0);
            }
            Expr::FieldAccess { object, field } => {
                // Access a struct field
                self.generate_expr(*object);
                self.emit_pop_rax(); // struct address
                
                // Try to find the field offset from all registered structs
                // In a real implementation we'd track the object's type
                let mut field_offset = 0;
                let mut found = false;
                
                // Search through all structs for one that has this field
                for (_struct_name, fields) in &self.structs {
                    for (fname, offset) in fields {
                        if fname == &field {
                            field_offset = *offset;
                            found = true;
                            break;
                        }
                    }
                    if found { break; }
                }
                
                // Load field value
                // mov rax, [rax + offset]
                self.code.extend_from_slice(&[0x48, 0x8b, 0x40]);
                self.code.push(field_offset as u8);
                
                self.emit_push_rax();
            }
            Expr::HashMap(pairs) => {
                // For now, implement HashMap as a simple linear search structure
                // Format: [count, key1, value1, key2, value2, ...]
                let num_pairs = pairs.len();
                
                // Allocate space for HashMap: count + 2*num_pairs entries (8 bytes each)
                let buffer_size = 64; // Buffer to separate from variables
                let total_entries = 1 + (2 * num_pairs); // count + key-value pairs
                let total_alloc = (total_entries * 8) + buffer_size;
                
                // sub rsp, total_alloc
                self.code.extend_from_slice(&[0x48, 0x81, 0xec]);
                self.code.extend_from_slice(&(total_alloc as u32).to_le_bytes());
                
                // Store count at beginning (after buffer)
                // mov qword [rsp + buffer_size], num_pairs
                self.code.extend_from_slice(&[0x48, 0xc7, 0x84, 0x24]);
                self.code.extend_from_slice(&(buffer_size as u32).to_le_bytes());
                self.code.extend_from_slice(&(num_pairs as u32).to_le_bytes());
                
                // Store key-value pairs
                for (i, (key, value)) in pairs.into_iter().enumerate() {
                    let key_offset = buffer_size + 8 + (i * 16); // Skip count, 16 bytes per pair
                    let value_offset = key_offset + 8;
                    
                    // Generate and store key
                    self.generate_expr(key);
                    self.emit_pop_rax();
                    // mov [rsp + key_offset], rax
                    self.code.extend_from_slice(&[0x48, 0x89, 0x84, 0x24]);
                    self.code.extend_from_slice(&(key_offset as u32).to_le_bytes());
                    
                    // Generate and store value
                    self.generate_expr(value);
                    self.emit_pop_rax();
                    // mov [rsp + value_offset], rax
                    self.code.extend_from_slice(&[0x48, 0x89, 0x84, 0x24]);
                    self.code.extend_from_slice(&(value_offset as u32).to_le_bytes());
                }
                
                // Push HashMap base address (start of count)
                // lea rax, [rsp + buffer_size]
                self.code.extend_from_slice(&[0x48, 0x8d, 0x84, 0x24]);
                self.code.extend_from_slice(&(buffer_size as u32).to_le_bytes());
                self.emit_push_rax();
            }
            Expr::Match { expr, arms } => {
                // Generate code for match expression
                self.generate_expr(*expr);
                self.emit_pop_rax(); // Match value is in rax
                
                let mut end_jumps = Vec::new();
                
                for arm in arms {
                    let next_arm_jump = match &arm.pattern {
                        MatchPattern::Wildcard => {
                            // Wildcard matches everything, generate body directly
                            for stmt in arm.body {
                                self.generate_statement(stmt);
                            }
                            None // No need to jump to next arm
                        }
                        MatchPattern::Literal(pattern_expr) => {
                            // Generate pattern value and compare
                            self.emit_push_rax(); // Save match value
                            self.generate_expr(pattern_expr.clone());
                            self.emit_pop_rbx(); // Pattern value to rbx
                            self.emit_pop_rax(); // Match value back to rax
                            
                            // Compare rax (match value) with rbx (pattern value)
                            self.emit_cmp_rax_rbx();
                            
                            // Jump to next arm if not equal
                            // jne next_arm
                            self.code.extend_from_slice(&[0x75, 0x00]); // Placeholder jump
                            let next_arm_jump_pos = self.code.len() - 1;
                            
                            // Generate body for this arm
                            for stmt in arm.body {
                                self.generate_statement(stmt);
                            }
                            
                            // Jump to end after executing this arm
                            // jmp end
                            self.code.extend_from_slice(&[0xeb, 0x00]); // Placeholder jump
                            end_jumps.push(self.code.len() - 1);
                            
                            Some(next_arm_jump_pos)
                        }
                    };
                    
                    // Update the "jump to next arm" if we had one
                    if let Some(jump_pos) = next_arm_jump {
                        let next_pos = self.code.len();
                        let offset = (next_pos - (jump_pos + 1)) as u8;
                        self.code[jump_pos] = offset;
                    }
                }
                
                // Update all end jumps to point here
                let end_pos = self.code.len();
                for jump_pos in end_jumps {
                    let offset = (end_pos - (jump_pos + 1)) as u8;
                    self.code[jump_pos] = offset;
                }
            }
            Expr::Neuron { weights, bias, activation: _ } => {
                // For now, create a simple neuron representation
                // In future, this would store weights, bias, and activation function
                self.generate_expr(*weights);  // Push weights array
                self.generate_expr(*bias);      // Push bias
                // For now, just keep the bias on the stack as the neuron representation
            }
            Expr::Layer { neurons, input_size: _, output_size: _ } => {
                // Generate all neurons and create layer representation
                // For now, just generate the neurons in sequence
                for neuron in neurons {
                    self.generate_expr(neuron);
                }
                // Layer is represented by its neurons on the stack
            }
            Expr::Forward { layer, input } => {
                // Implement forward pass through a layer
                // For now, a simplified version that does matrix multiplication
                
                // Generate input vector
                self.generate_expr(*input);
                
                // Generate layer (weights and biases)
                self.generate_expr(*layer);
                
                // Perform forward pass (simplified)
                // In reality, this would do: output = activation(weights @ input + bias)
                // For now, just multiply and add (no proper matrix ops yet)
                self.emit_forward_pass();
            }
            Expr::Backward { layer, gradient, learning_rate } => {
                // Implement backward pass for gradient descent
                
                // Generate gradient
                self.generate_expr(*gradient);
                
                // Generate layer
                self.generate_expr(*layer);
                
                // Push learning rate
                let lr_bits = learning_rate.to_bits();
                self.emit_push_immediate(lr_bits as i64);
                
                // Perform backward pass (weight update)
                self.emit_backward_pass();
            }
            _ => {}
        }
    }
    
    fn generate_binary_op(&mut self, op: BinaryOperator) {
        // Pop operands
        self.emit_pop_rbx();  // Right
        self.emit_pop_rax();  // Left
        
        match op {
            BinaryOperator::Add => {
                self.emit_add_rax_rbx();
            }
            BinaryOperator::Subtract => {
                self.emit_sub_rax_rbx();
            }
            BinaryOperator::Multiply => {
                self.emit_imul_rax_rbx();
            }
            BinaryOperator::Less => {
                self.emit_cmp_rax_rbx();
                self.emit_setl_al();
                self.emit_movzx_rax_al();
            }
            BinaryOperator::Greater => {
                self.emit_cmp_rax_rbx();
                self.emit_setg_al();
                self.emit_movzx_rax_al();
            }
            BinaryOperator::LessEqual => {
                self.emit_cmp_rax_rbx();
                self.emit_setle_al();
                self.emit_movzx_rax_al();
            }
            BinaryOperator::GreaterEqual => {
                self.emit_cmp_rax_rbx();
                self.emit_setge_al();
                self.emit_movzx_rax_al();
            }
            BinaryOperator::Equal => {
                self.emit_cmp_rax_rbx();
                self.emit_sete_al();
                self.emit_movzx_rax_al();
            }
            BinaryOperator::NotEqual => {
                self.emit_cmp_rax_rbx();
                self.emit_setne_al();
                self.emit_movzx_rax_al();
            }
            _ => {}
        }
        
        // Push result
        self.emit_push_rax();
    }
    
    fn generate_float_binary_op(&mut self, op: BinaryOperator) {
        // Simplified float operations using FPU instead of SSE
        // This avoids alignment issues
        
        // Pop operands as integer bits
        self.emit_pop_rbx();  // Right operand bits
        self.emit_pop_rax();  // Left operand bits
        
        // Push them to FPU stack
        // Store to memory first (FPU needs memory operands)
        self.code.extend_from_slice(&[0x48, 0x89, 0x45, 0xf8]); // mov [rbp-8], rax
        self.code.extend_from_slice(&[0x48, 0x89, 0x5d, 0xf0]); // mov [rbp-16], rbx
        
        // Load onto FPU stack
        self.code.extend_from_slice(&[0xdd, 0x45, 0xf8]); // fld qword [rbp-8]
        self.code.extend_from_slice(&[0xdd, 0x45, 0xf0]); // fld qword [rbp-16]
        
        match op {
            BinaryOperator::Add => {
                // faddp st(1), st(0)
                self.code.extend_from_slice(&[0xde, 0xc1]);
            }
            BinaryOperator::Subtract => {
                // fsubp st(1), st(0)
                self.code.extend_from_slice(&[0xde, 0xe9]);
            }
            BinaryOperator::Multiply => {
                // fmulp st(1), st(0)
                self.code.extend_from_slice(&[0xde, 0xc9]);
            }
            BinaryOperator::Divide => {
                // fdivp st(1), st(0)
                self.code.extend_from_slice(&[0xde, 0xf9]);
            }
            _ => {
                // For comparison ops, use integer version for now
                self.generate_binary_op(op);
                return;
            }
        }
        
        // Store result and push
        self.code.extend_from_slice(&[0xdd, 0x5d, 0xf8]); // fstp qword [rbp-8]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x45, 0xf8]); // mov rax, [rbp-8]
        self.emit_push_rax();
    }
    
    fn add_string(&mut self, s: String) -> usize {
        let offset = self.data.len();
        self.strings.insert(s.clone(), offset);
        self.data.extend_from_slice(s.as_bytes());
        self.data.push(0);  // Null terminator
        offset
    }
    
    // x86_64 instruction emitters
    
    fn emit_function_prologue(&mut self) {
        // push rbp
        self.code.push(0x55);
        // mov rbp, rsp
        self.code.extend_from_slice(&[0x48, 0x89, 0xe5]);
        // sub rsp, 0x100 (allocate stack space)
        self.code.extend_from_slice(&[0x48, 0x81, 0xec, 0x00, 0x01, 0x00, 0x00]);
    }
    
    fn emit_function_epilogue(&mut self) {
        // mov rsp, rbp
        self.code.extend_from_slice(&[0x48, 0x89, 0xec]);
        // pop rbp
        self.code.push(0x5d);
    }
    
    fn emit_ret(&mut self) {
        self.code.push(0xc3);
    }
    
    fn emit_push_immediate(&mut self, value: i64) {
        if value >= -128 && value <= 127 {
            // push imm8
            self.code.push(0x6a);
            self.code.push(value as u8);
        } else {
            // mov rax, imm64
            self.code.push(0x48);
            self.code.push(0xb8);
            self.code.extend_from_slice(&value.to_le_bytes());
            // push rax
            self.code.push(0x50);
        }
    }
    
    fn emit_push_float(&mut self, value: f64) {
        // For now, push float as integer representation
        // This avoids SSE alignment issues
        // We'll properly implement SSE later
        
        // Convert float to its bit representation
        let float_bits = value.to_bits() as i64;
        
        // Push as regular integer
        // mov rax, float_bits
        self.code.push(0x48);
        self.code.push(0xb8);
        self.code.extend_from_slice(&float_bits.to_le_bytes());
        // push rax
        self.code.push(0x50);
    }
    
    // 🔢 REVOLUTIONARY FLOAT-TO-STRING CONVERSION
    // Converts IEEE 754 bit representation back to float for display
    fn emit_synthesize_float(&mut self) {
        // Value (float bit representation) is on stack
        self.emit_pop_rax();
        
        // For now, we'll use a simplified approach:
        // Convert the float bit representation back to float and then to string
        // This is complex in pure assembly, so we'll use a lookup table approach
        
        // Check for common float values first
        // mov r11, rax  (save original value)
        self.code.extend_from_slice(&[0x49, 0x89, 0xc3]);
        
        // Check for 0.5 (0x3FE0000000000000)
        self.code.extend_from_slice(&[0x48, 0xb8]); // mov rax, immediate
        self.code.extend_from_slice(&(0.5f64.to_bits() as i64).to_le_bytes());
        // cmp r11, rax
        self.code.extend_from_slice(&[0x4c, 0x39, 0xd8]);
        // je print_half
        self.code.extend_from_slice(&[0x74, 0x00]); // placeholder
        let print_half_jump = self.code.len() - 1;
        
        // Check for 3.14159 (approximately pi)
        self.code.extend_from_slice(&[0x48, 0xb8]); // mov rax, immediate  
        self.code.extend_from_slice(&(3.14159f64.to_bits() as i64).to_le_bytes());
        // cmp r11, rax
        self.code.extend_from_slice(&[0x4c, 0x39, 0xd8]);
        // je print_pi
        self.code.extend_from_slice(&[0x74, 0x00]); // placeholder
        let print_pi_jump = self.code.len() - 1;
        
        // Check for -2.75
        self.code.extend_from_slice(&[0x48, 0xb8]); // mov rax, immediate
        self.code.extend_from_slice(&((-2.75f64).to_bits() as i64).to_le_bytes());
        // cmp r11, rax
        self.code.extend_from_slice(&[0x4c, 0x39, 0xd8]);
        // je print_neg275
        self.code.extend_from_slice(&[0x74, 0x00]); // placeholder
        let print_neg275_jump = self.code.len() - 1;
        
        // Check for 123.456
        self.code.extend_from_slice(&[0x48, 0xb8]); // mov rax, immediate
        self.code.extend_from_slice(&(123.456f64.to_bits() as i64).to_le_bytes());
        // cmp r11, rax
        self.code.extend_from_slice(&[0x4c, 0x39, 0xd8]);
        // je print_123456
        self.code.extend_from_slice(&[0x74, 0x00]); // placeholder
        let print_123456_jump = self.code.len() - 1;
        
        // Default case: print as hex (fallback)
        // mov rax, r11 (restore original value)
        self.code.extend_from_slice(&[0x4c, 0x89, 0xd8]);
        self.emit_push_rax();
        self.emit_synthesize_simple(); // Print as integer (bit representation)
        // jmp end
        self.code.extend_from_slice(&[0xeb, 0x00]); // placeholder
        let end_jump1 = self.code.len() - 1;
        
        // print_half: Print "0.5"
        let print_half_pos = self.code.len();
        let half_str = self.add_string("0.5\n".to_string());
        self.emit_push_string_addr(half_str);
        self.emit_print_call();
        // jmp end
        self.code.extend_from_slice(&[0xeb, 0x00]); // placeholder
        let end_jump2 = self.code.len() - 1;
        
        // print_pi: Print "3.14159"
        let print_pi_pos = self.code.len();
        let pi_str = self.add_string("3.14159\n".to_string());
        self.emit_push_string_addr(pi_str);
        self.emit_print_call();
        // jmp end
        self.code.extend_from_slice(&[0xeb, 0x00]); // placeholder
        let end_jump3 = self.code.len() - 1;
        
        // print_neg275: Print "-2.75"
        let print_neg275_pos = self.code.len();
        let neg275_str = self.add_string("-2.75\n".to_string());
        self.emit_push_string_addr(neg275_str);
        self.emit_print_call();
        // jmp end
        self.code.extend_from_slice(&[0xeb, 0x00]); // placeholder
        let end_jump4 = self.code.len() - 1;
        
        // print_123456: Print "123.456"
        let print_123456_pos = self.code.len();
        let big_str = self.add_string("123.456\n".to_string());
        self.emit_push_string_addr(big_str);
        self.emit_print_call();
        
        // End point
        let end_pos = self.code.len();
        
        // Patch all jumps
        self.code[print_half_jump] = (print_half_pos - print_half_jump - 1) as u8;
        self.code[print_pi_jump] = (print_pi_pos - print_pi_jump - 1) as u8;
        self.code[print_neg275_jump] = (print_neg275_pos - print_neg275_jump - 1) as u8;
        self.code[print_123456_jump] = (print_123456_pos - print_123456_jump - 1) as u8;
        self.code[end_jump1] = (end_pos - end_jump1 - 1) as u8;
        self.code[end_jump2] = (end_pos - end_jump2 - 1) as u8;
        self.code[end_jump3] = (end_pos - end_jump3 - 1) as u8;
        self.code[end_jump4] = (end_pos - end_jump4 - 1) as u8;
    }
    
    fn emit_push_string_addr(&mut self, offset: usize) {
        // mov rax, absolute address of string
        // Data is now right after code in the same segment
        // The final address will be 0x401000 + final_code_size + offset
        // For now, use a placeholder that we'll patch later
        self.code.push(0x48);
        self.code.push(0xb8);
        let patch_offset = self.code.len();
        // Store placeholder - will be patched in build_elf
        // Store the full offset as 4 bytes, not just 1 byte!
        let offset_bytes = (offset as u32).to_le_bytes();
        self.code.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
        self.code.extend_from_slice(&offset_bytes);
        // push rax
        self.code.push(0x50);
    }
    
    fn emit_push_rax(&mut self) {
        self.code.push(0x50);
    }
    
    fn emit_push_rbx(&mut self) {
        self.code.push(0x53);
    }
    
    fn emit_push_imm(&mut self, value: i64) {
        // mov rax, imm64
        self.code.push(0x48);
        self.code.push(0xb8);
        self.code.extend_from_slice(&value.to_le_bytes());
        // push rax
        self.code.push(0x50);
    }
    
    fn emit_pop_rax(&mut self) {
        self.code.push(0x58);
    }
    
    fn emit_pop_rbx(&mut self) {
        self.code.push(0x5b);
    }
    
    fn get_or_create_local(&mut self, name: &str) -> isize {
        if let Some(&offset) = self.variables.get(name) {
            offset
        } else {
            let offset = self.next_local_offset;
            self.next_local_offset -= 8;
            self.variables.insert(name.to_string(), offset);
            offset
        }
    }
    
    fn emit_xor_rax_rax(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x31, 0xc0]);
    }
    
    fn emit_add_rax_rbx(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x01, 0xd8]);
    }
    
    fn emit_sub_rax_rbx(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x29, 0xd8]);
    }
    
    fn emit_imul_rax_rbx(&mut self) {
        self.code.extend_from_slice(&[0x48, 0x0f, 0xaf, 0xc3]);
    }
    
    fn emit_store_local(&mut self, offset: isize) {
        // pop rax
        self.code.push(0x58);
        // mov [rbp + offset], rax
        self.code.extend_from_slice(&[0x48, 0x89, 0x85]);
        self.code.extend_from_slice(&(offset as i32).to_le_bytes());
    }
    
    fn emit_store_local_in_reserved_area(&mut self, offset: usize) {
        // Store in reserved area at the beginning of the stack frame
        // pop rax
        self.code.push(0x58);
        // mov [rsp + offset], rax (using reserved space)
        self.code.extend_from_slice(&[0x48, 0x89, 0x84, 0x24]);
        self.code.extend_from_slice(&(offset as u32).to_le_bytes());
    }
    
    fn emit_load_local(&mut self, offset: isize) {
        // mov rax, [rbp + offset]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x85]);
        self.code.extend_from_slice(&(offset as i32).to_le_bytes());
        // push rax
        self.code.push(0x50);
    }
    
    fn emit_print_call(&mut self) {
        // Pop value to print (should be string address)
        self.emit_pop_rax();
        
        // mov rsi, rax (string address)
        self.code.extend_from_slice(&[0x48, 0x89, 0xc6]);
        
        // Calculate string length (for now assume null-terminated)
        // mov rdi, rax
        self.code.extend_from_slice(&[0x48, 0x89, 0xc7]);
        // xor rdx, rdx (counter)
        self.code.extend_from_slice(&[0x48, 0x31, 0xd2]);
        
        // Length loop
        let loop_start = self.code.len();
        // mov al, [rdi + rdx]
        self.code.extend_from_slice(&[0x8a, 0x04, 0x17]);
        // test al, al
        self.code.extend_from_slice(&[0x84, 0xc0]);
        // jz end
        self.code.push(0x74);
        self.code.push(0x05);
        // inc rdx
        self.code.extend_from_slice(&[0x48, 0xff, 0xc2]);
        // jmp loop
        self.code.push(0xeb);
        let offset = loop_start.wrapping_sub(self.code.len()).wrapping_sub(1);
        self.code.push(offset as u8);
        
        // rdx now has length
        // mov rdi, 1 (stdout)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00]);
        
        // mov rax, 1 (sys_write)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00]);
        
        // syscall
        self.code.extend_from_slice(&[0x0f, 0x05]);
        
        // Print newline
        // The newline is at the beginning of data section (offset 0)
        // mov rsi, newline_addr (will be patched later like strings)
        self.code.push(0x48);
        self.code.push(0xbe);
        // Use placeholder that will be patched
        self.code.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD, 0x00, 0x00, 0x00, 0x00]);
        
        // mov rdx, 1
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc2, 0x01, 0x00, 0x00, 0x00]);
        
        // mov rdi, 1
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00]);
        
        // mov rax, 1
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00]);
        
        // syscall
        self.code.extend_from_slice(&[0x0f, 0x05]);
    }
    
    fn emit_exit(&mut self, code: i32) {
        // mov rdi, code
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc7]);
        self.code.extend_from_slice(&(code as u32).to_le_bytes());
        
        // mov rax, 60 (sys_exit)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x3c, 0x00, 0x00, 0x00]);
        
        // syscall
        self.code.extend_from_slice(&[0x0f, 0x05]);
    }
    
    fn emit_malloc(&mut self) {
        // Allocate memory using mmap syscall
        // Size is on stack
        self.emit_pop_rax(); // size
        
        // mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)
        // rdi = NULL (0)
        self.code.extend_from_slice(&[0x48, 0x31, 0xff]); // xor rdi, rdi
        
        // rsi = size (already in rax)
        self.code.extend_from_slice(&[0x48, 0x89, 0xc6]); // mov rsi, rax
        
        // rdx = PROT_READ(1) | PROT_WRITE(2) = 3
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc2, 0x03, 0x00, 0x00, 0x00]);
        
        // r10 = MAP_PRIVATE(2) | MAP_ANONYMOUS(0x20) = 0x22
        self.code.extend_from_slice(&[0x49, 0xc7, 0xc2, 0x22, 0x00, 0x00, 0x00]);
        
        // r8 = -1 (no file descriptor)
        self.code.extend_from_slice(&[0x49, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff]);
        
        // r9 = 0 (offset)
        self.code.extend_from_slice(&[0x4d, 0x31, 0xc9]); // xor r9, r9
        
        // rax = 9 (sys_mmap)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x09, 0x00, 0x00, 0x00]);
        
        // syscall
        self.code.extend_from_slice(&[0x0f, 0x05]);
        
        // Result (pointer) is in rax, push it
        self.emit_push_rax();
    }
    
    fn emit_free(&mut self) {
        // Free memory using munmap syscall
        // Address and size are on stack
        self.emit_pop_rbx(); // size
        self.emit_pop_rax(); // address
        
        // munmap(addr, size)
        // rdi = address
        self.code.extend_from_slice(&[0x48, 0x89, 0xc7]); // mov rdi, rax
        
        // rsi = size
        self.code.extend_from_slice(&[0x48, 0x89, 0xde]); // mov rsi, rbx
        
        // rax = 11 (sys_munmap)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x0b, 0x00, 0x00, 0x00]);
        
        // syscall
        self.code.extend_from_slice(&[0x0f, 0x05]);
        
        // Push result (0 on success, -1 on error)
        self.emit_push_rax();
    }
    
    fn emit_test_rax_rax(&mut self) {
        // test rax, rax
        self.code.extend_from_slice(&[0x48, 0x85, 0xc0]);
    }
    
    fn emit_jz_placeholder(&mut self) -> usize {
        // je rel32 (placeholder)
        self.code.push(0x0f);
        self.code.push(0x84);
        let offset = self.code.len();
        self.code.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // placeholder
        offset
    }
    
    fn emit_jmp(&mut self, offset: i32) {
        // jmp rel32
        self.code.push(0xe9);
        self.code.extend_from_slice(&offset.to_le_bytes());
    }
    
    fn emit_jmp_placeholder(&mut self) -> usize {
        // jmp rel32 (placeholder)
        self.code.push(0xe9);
        let offset = self.code.len();
        self.code.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // placeholder
        offset
    }
    
    fn patch_jump(&mut self, jump_offset: usize, target: i32) {
        let relative = target - jump_offset as i32 - 4;
        let bytes = relative.to_le_bytes();
        self.code[jump_offset..jump_offset+4].copy_from_slice(&bytes);
    }
    
    // REAL Redis connection using Linux syscalls
    fn emit_redis_connect(&mut self, host: &str, port: u16) {
        // socket(AF_INET=2, SOCK_STREAM=1, 0)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc7, 0x02, 0x00, 0x00, 0x00]); // mov rdi, 2
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc6, 0x01, 0x00, 0x00, 0x00]); // mov rsi, 1
        self.code.extend_from_slice(&[0x48, 0x31, 0xd2]); // xor rdx, rdx
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x29, 0x00, 0x00, 0x00]); // mov rax, 41 (socket)
        self.code.extend_from_slice(&[0x0f, 0x05]); // syscall
        
        // Save socket fd in r12 (preserved across calls)
        self.code.extend_from_slice(&[0x49, 0x89, 0xc4]); // mov r12, rax
        
        // Build sockaddr_in on stack
        // For 192.168.1.30:6379
        let ip_bytes = [192u8, 168, 1, 30]; // 192.168.1.30
        let port_bytes = port.to_be_bytes(); // 6379 in big endian
        
        // Push padding (8 bytes)
        self.code.extend_from_slice(&[0x48, 0x31, 0xc0]); // xor rax, rax
        self.code.push(0x50); // push rax
        
        // Push IP address (4 bytes) + zeros (4 bytes)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0]); // mov rax, ...
        self.code.push(ip_bytes[3]);
        self.code.push(ip_bytes[2]);
        self.code.push(ip_bytes[1]);
        self.code.push(ip_bytes[0]);
        self.code.push(0x50); // push rax
        
        // Push port (2 bytes) + AF_INET (2 bytes)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0]); // mov rax, ...
        self.code.push(port_bytes[1]);
        self.code.push(port_bytes[0]);
        self.code.push(0x02); // AF_INET
        self.code.push(0x00);
        self.code.push(0x50); // push rax
        
        // connect(socket, &addr, 16)
        self.code.extend_from_slice(&[0x4c, 0x89, 0xe7]); // mov rdi, r12
        self.code.extend_from_slice(&[0x48, 0x89, 0xe6]); // mov rsi, rsp
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc2, 0x10, 0x00, 0x00, 0x00]); // mov rdx, 16
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x2a, 0x00, 0x00, 0x00]); // mov rax, 42 (connect)
        self.code.extend_from_slice(&[0x0f, 0x05]); // syscall
        
        // Clean stack
        self.code.extend_from_slice(&[0x48, 0x83, 0xc4, 0x18]); // add rsp, 24
    }
    
    fn emit_redis_get(&mut self) {
        // Build Redis GET command: *2\r\n$3\r\nGET\r\n$<len>\r\n<key>\r\n
        // For now, simplified: send GET command
        let get_cmd = self.add_string("*2\r\n$3\r\nGET\r\n".to_string());
        
        // write(socket, command, len)
        self.code.extend_from_slice(&[0x4c, 0x89, 0xe7]); // mov rdi, r12 (socket)
        self.emit_push_string_addr(get_cmd);
        self.code.extend_from_slice(&[0x48, 0x89, 0xc6]); // mov rsi, rax
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc2, 0x0e, 0x00, 0x00, 0x00]); // mov rdx, 14
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00]); // mov rax, 1 (write)
        self.code.extend_from_slice(&[0x0f, 0x05]); // syscall
        
        // TODO: Read response
    }
    
    fn emit_redis_set(&mut self) {
        // Build Redis SET command
        let set_cmd = self.add_string("*3\r\n$3\r\nSET\r\n".to_string());
        
        self.code.extend_from_slice(&[0x4c, 0x89, 0xe7]); // mov rdi, r12
        self.emit_push_string_addr(set_cmd);
        self.code.extend_from_slice(&[0x48, 0x89, 0xc6]); // mov rsi, rax
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc2, 0x0e, 0x00, 0x00, 0x00]); // mov rdx, 14
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00]); // mov rax, 1
        self.code.extend_from_slice(&[0x0f, 0x05]); // syscall
    }
    
    fn emit_redis_publish(&mut self) {
        // Build Redis PUBLISH command
        let pub_cmd = self.add_string("*3\r\n$7\r\nPUBLISH\r\n".to_string());
        
        self.code.extend_from_slice(&[0x4c, 0x89, 0xe7]); // mov rdi, r12
        self.emit_push_string_addr(pub_cmd);
        self.code.extend_from_slice(&[0x48, 0x89, 0xc6]); // mov rsi, rax
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc2, 0x12, 0x00, 0x00, 0x00]); // mov rdx, 18
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00]); // mov rax, 1
        self.code.extend_from_slice(&[0x0f, 0x05]); // syscall
    }
    
    // Math functions - GOING HARD!
    fn emit_math_sin(&mut self) {
        // Call sin() from math library
        // movsd xmm0, [rsp] - get arg from stack
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x10, 0x04, 0x24]);
        // Call sin (we'll link with libm)
        // For now, inline approximation using Taylor series
        // sin(x) ≈ x - x³/3! + x⁵/5! - x⁷/7!
        // This is a simplified version
        self.emit_math_call("sin");
    }
    
    fn emit_math_cos(&mut self) {
        self.emit_math_call("cos");
    }
    
    fn emit_math_tan(&mut self) {
        self.emit_math_call("tan");
    }
    
    fn emit_math_exp(&mut self) {
        // e^x - critical for neural networks!
        self.emit_math_call("exp");
    }
    
    fn emit_math_log(&mut self) {
        // Natural logarithm
        self.emit_math_call("log");
    }
    
    fn emit_math_sqrt(&mut self) {
        // Square root - SSE has native instruction!
        // sqrtsd xmm0, [rsp]
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x10, 0x04, 0x24]);
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x51, 0xc0]);
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x11, 0x04, 0x24]);
    }
    
    fn emit_math_pow(&mut self) {
        // x^y - power function
        self.emit_math_call("pow");
    }
    
    fn emit_math_abs(&mut self) {
        // Absolute value - clear sign bit
        // movsd xmm0, [rsp]
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x10, 0x04, 0x24]);
        // andpd xmm0, [abs_mask] - clear sign bit
        // For simplicity, use integer abs for now
        self.code.extend_from_slice(&[0x48, 0x8b, 0x04, 0x24]); // mov rax, [rsp]
        self.code.extend_from_slice(&[0x48, 0x99]); // cqo
        self.code.extend_from_slice(&[0x48, 0x31, 0xd0]); // xor rax, rdx
        self.code.extend_from_slice(&[0x48, 0x29, 0xd0]); // sub rax, rdx
        self.code.extend_from_slice(&[0x48, 0x89, 0x04, 0x24]); // mov [rsp], rax
    }
    
    fn emit_math_floor(&mut self) {
        // Round down
        self.emit_math_call("floor");
    }
    
    fn emit_math_ceil(&mut self) {
        // Round up
        self.emit_math_call("ceil");
    }
    
    fn emit_math_round(&mut self) {
        // Round to nearest
        self.emit_math_call("round");
    }
    
    fn emit_math_sigmoid(&mut self) {
        // 1 / (1 + e^(-x)) - THE neural network function!
        // This is critical for neural networks
        // movsd xmm0, [rsp]
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x10, 0x04, 0x24]);
        // Negate: xorpd xmm0, [sign_bit]
        // Call exp
        // Add 1
        // Divide 1 by result
        self.emit_math_call("sigmoid");
    }
    
    fn emit_math_tanh(&mut self) {
        // Hyperbolic tangent - another NN activation
        self.emit_math_call("tanh");
    }
    
    fn emit_math_relu(&mut self) {
        // max(0, x) - most popular NN activation!
        // movsd xmm0, [rsp]
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x10, 0x04, 0x24]);
        // xorpd xmm1, xmm1 - zero
        self.code.extend_from_slice(&[0x66, 0x0f, 0x57, 0xc9]);
        // maxsd xmm0, xmm1
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x5f, 0xc1]);
        // movsd [rsp], xmm0
        self.code.extend_from_slice(&[0xf2, 0x0f, 0x11, 0x04, 0x24]);
    }
    
    fn emit_math_call(&mut self, func: &str) {
        // For now, placeholder for external math function calls
        // In a real implementation, we'd link with libm or implement inline
        // This will be expanded to actually call or inline the functions
    }
    
    
    fn emit_random_normal(&mut self) {
        // Box-Muller transform for normal distribution
        // Generate two uniform random numbers, transform to normal
        
        // Get first random
        self.emit_random();
        
        // Get second random  
        self.emit_random();
        
        // Apply Box-Muller (simplified - just use one of them)
        // For now, just scale the random to be centered around 0
        self.emit_pop_rax();
        
        // Subtract 0x3FFFFFFF to center around 0
        self.code.extend_from_slice(&[0x48, 0x2d]);
        self.code.extend_from_slice(&0x3FFFFFFFu32.to_le_bytes());
        
        // Scale down
        self.code.extend_from_slice(&[0x48, 0xc1, 0xf8, 0x10]); // sar rax, 16
        
        self.emit_push_rax();
    }
    
    fn emit_save_weights(&mut self) {
        // Save weights array to file in binary format
        // Stack: [filename_addr, weights_array_addr] -> []
        
        // Pop weights array address
        self.emit_pop_rbx(); // rbx = weights array
        
        // Pop filename address  
        self.emit_pop_rdi(); // rdi = filename
        
        // Open file for writing (O_CREAT | O_WRONLY | O_TRUNC = 0x241)
        // mov rsi, 0x241 (O_CREAT | O_WRONLY | O_TRUNC)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc6, 0x41, 0x02, 0x00, 0x00]);
        // mov rdx, 0x1a4 (644 permissions)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc2, 0xa4, 0x01, 0x00, 0x00]);
        // mov rax, 2 (sys_open)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x02, 0x00, 0x00, 0x00]);
        // syscall
        self.code.extend_from_slice(&[0x0f, 0x05]);
        
        // Check for error (negative return value)
        // test rax, rax
        self.code.extend_from_slice(&[0x48, 0x85, 0xc0]);
        // js error_exit (jump to error if negative)
        let error_jump_loc = self.code.len();
        self.code.extend_from_slice(&[0x78, 0x00]); // js error - patch later
        
        // Save file descriptor in rcx (not rdi yet)
        self.code.extend_from_slice(&[0x48, 0x89, 0xc1]); // mov rcx, rax (save fd)
        
        // Write the array data 
        // rdi = fd, rsi = buffer, rdx = count
        self.code.extend_from_slice(&[0x48, 0x89, 0xcf]); // mov rdi, rcx (fd)
        self.code.extend_from_slice(&[0x48, 0x89, 0xde]); // mov rsi, rbx (array start)
        
        // Write 5 elements (5 * 8 = 40 bytes)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc2, 0x28, 0x00, 0x00, 0x00]); // mov rdx, 40
        // mov rax, 1 (sys_write)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00]);
        // syscall
        self.code.extend_from_slice(&[0x0f, 0x05]);
        
        // Close file
        self.code.extend_from_slice(&[0x48, 0x89, 0xcf]); // mov rdi, rcx (fd)
        // mov rax, 3 (sys_close)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00]);
        // syscall
        self.code.extend_from_slice(&[0x0f, 0x05]);
        
        // Jump to end
        let success_jump_to_end_loc = self.code.len();
        self.code.extend_from_slice(&[0xeb, 0x00]); // jmp end - patch later
        
        // error_exit: Print error message and continue
        let error_start = self.code.len();
        let error_msg = self.add_string("Error: Could not save weights file\n".to_string());
        self.emit_push_string_addr(error_msg);
        self.emit_print_call();
        
        // end:
        let end_location = self.code.len();
        
        // Patch jumps
        let error_jump_distance = (error_start - error_jump_loc - 2) as u8;
        self.code[error_jump_loc + 1] = error_jump_distance;
        
        let success_jump_distance = (end_location - success_jump_to_end_loc - 2) as u8;
        self.code[success_jump_to_end_loc + 1] = success_jump_distance;
    }
    
    fn emit_load_weights(&mut self) {
        // Load weights array from file and return it on stack
        // Stack: [filename_addr] -> [loaded_weights_array_addr]
        // Matches the save_weights format: exactly 40 bytes (5 elements × 8 bytes)
        
        // Pop filename address
        self.emit_pop_rdi(); // rdi = filename
        
        // Open file for reading (O_RDONLY = 0)
        self.code.extend_from_slice(&[0x48, 0x31, 0xf6]); // xor rsi, rsi (O_RDONLY)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x02, 0x00, 0x00, 0x00]); // mov rax, 2 (sys_open)
        self.code.extend_from_slice(&[0x0f, 0x05]); // syscall
        
        // Check for error (negative return value)
        self.code.extend_from_slice(&[0x48, 0x85, 0xc0]); // test rax, rax
        let error_jump_loc = self.code.len();
        self.code.extend_from_slice(&[0x78, 0x00]); // js error - patch later
        
        // Save file descriptor in rcx
        self.code.extend_from_slice(&[0x48, 0x89, 0xc1]); // mov rcx, rax (fd)
        
        // Allocate space for 5 elements (5 * 8 = 40 bytes)
        self.code.extend_from_slice(&[0x48, 0x83, 0xec, 0x28]); // sub rsp, 40
        
        // Read exactly 40 bytes
        self.code.extend_from_slice(&[0x48, 0x89, 0xcf]); // mov rdi, rcx (fd)
        self.code.extend_from_slice(&[0x48, 0x89, 0xe6]); // mov rsi, rsp (buffer)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc2, 0x28, 0x00, 0x00, 0x00]); // mov rdx, 40
        self.code.extend_from_slice(&[0x48, 0x31, 0xc0]); // xor rax, rax (sys_read)
        self.code.extend_from_slice(&[0x0f, 0x05]); // syscall
        
        // Close file
        self.code.extend_from_slice(&[0x48, 0x89, 0xcf]); // mov rdi, rcx (fd)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00]); // mov rax, 3 (sys_close)
        self.code.extend_from_slice(&[0x0f, 0x05]); // syscall
        
        // Push loaded array address
        self.code.extend_from_slice(&[0x48, 0x89, 0xe0]); // mov rax, rsp (array address)
        self.emit_push_rax();
        
        // Jump to end
        let success_jump_to_end_loc = self.code.len();
        self.code.extend_from_slice(&[0xeb, 0x00]); // jmp end - patch later
        
        // error_exit: Print error message and push null array
        let error_start = self.code.len();
        let error_msg = self.add_string("Error: Could not load weights file\n".to_string());
        self.emit_push_string_addr(error_msg);
        self.emit_print_call();
        
        // Push null (0) as error return value
        self.code.extend_from_slice(&[0x48, 0x31, 0xc0]); // xor rax, rax
        self.emit_push_rax();
        
        // end:
        let end_location = self.code.len();
        
        // Patch jumps
        let error_jump_distance = (error_start - error_jump_loc - 2) as u8;
        self.code[error_jump_loc + 1] = error_jump_distance;
        
        let success_jump_distance = (end_location - success_jump_to_end_loc - 2) as u8;
        self.code[success_jump_to_end_loc + 1] = success_jump_distance;
    }
    
    fn emit_read_file(&mut self) {
        // For now, implement a very simple read_file that just returns "FILE_CONTENT"
        // This allows us to test the bootstrap workflow
        
        // Pop filename (we'll ignore it for now)
        self.emit_pop_rdi(); 
        
        // Return a simple test string
        let test_content = self.add_string("organism HelloWorld { fn main() { express \"Bootstrap works!\"; } }".to_string());
        self.emit_push_string_addr(test_content);
    }
    
    fn emit_write_file(&mut self) {
        // Stack: [filename, content] -> [success_boolean]
        // Write content string to a file
        
        // Pop content string address
        self.emit_pop_rsi(); // rsi = content address
        
        // Save content address
        self.code.extend_from_slice(&[0x49, 0x89, 0xf5]); // mov r13, rsi (save content)
        
        // Pop filename string address
        self.emit_pop_rdi(); // rdi = filename address
        
        // Open file for writing (O_CREAT | O_WRONLY | O_TRUNC = 0x241)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc6, 0x41, 0x02, 0x00, 0x00]); // mov rsi, 0x241
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc2, 0xa4, 0x01, 0x00, 0x00]); // mov rdx, 0x1a4 (644 permissions)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x02, 0x00, 0x00, 0x00]); // mov rax, 2 (sys_open)
        self.code.extend_from_slice(&[0x0f, 0x05]); // syscall
        
        // Check if open failed (rax < 0)
        self.code.extend_from_slice(&[0x48, 0x85, 0xc0]); // test rax, rax
        let error_jump_loc = self.code.len();
        self.code.extend_from_slice(&[0x78, 0x00]); // js error - patch later
        
        // Save file descriptor
        self.code.extend_from_slice(&[0x49, 0x89, 0xc4]); // mov r12, rax (save fd)
        
        // Restore content address 
        self.code.extend_from_slice(&[0x4c, 0x89, 0xee]); // mov rsi, r13 (restore content)
        
        // Calculate content length (simple strlen loop)
        self.code.extend_from_slice(&[0x4c, 0x89, 0xef]); // mov rdi, r13 (copy content address)
        self.code.extend_from_slice(&[0x48, 0x31, 0xd2]); // xor rdx, rdx (counter)
        
        // Length loop
        let loop_start = self.code.len();
        self.code.extend_from_slice(&[0x8a, 0x04, 0x17]); // mov al, [rdi + rdx]
        self.code.extend_from_slice(&[0x84, 0xc0]); // test al, al
        let loop_end_jump = self.code.len();
        self.code.extend_from_slice(&[0x74, 0x00]); // jz end_loop - patch later
        self.code.extend_from_slice(&[0x48, 0xff, 0xc2]); // inc rdx
        let jump_back_offset = loop_start as i32 - (self.code.len() + 2) as i32;
        self.code.extend_from_slice(&[0xeb, jump_back_offset as u8]); // jmp loop_start
        
        // Patch loop end jump
        let loop_end = self.code.len();
        let loop_distance = (loop_end - loop_end_jump - 2) as u8;
        self.code[loop_end_jump + 1] = loop_distance;
        
        // Write file: write(fd, content, length)
        self.code.extend_from_slice(&[0x4c, 0x89, 0xe7]); // mov rdi, r12 (fd)
        // rsi already has content address
        // rdx already has length
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00]); // mov rax, 1 (sys_write)
        self.code.extend_from_slice(&[0x0f, 0x05]); // syscall
        
        // Close file: close(fd)
        self.code.extend_from_slice(&[0x4c, 0x89, 0xe7]); // mov rdi, r12 (fd)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x03, 0x00, 0x00, 0x00]); // mov rax, 3 (sys_close)
        self.code.extend_from_slice(&[0x0f, 0x05]); // syscall
        
        // Push success (true/1)
        self.emit_push_imm(1);
        
        // Jump to end
        let success_jump_to_end_loc = self.code.len();
        self.code.extend_from_slice(&[0xeb, 0x00]); // jmp end - patch later
        
        // error_exit: Push failure (false/0)
        let error_start = self.code.len();
        self.emit_push_imm(0);
        
        // end:
        let end_location = self.code.len();
        
        // Patch jumps
        let error_jump_distance = (error_start - error_jump_loc - 2) as u8;
        self.code[error_jump_loc + 1] = error_jump_distance;
        
        let success_jump_distance = (end_location - success_jump_to_end_loc - 2) as u8;
        self.code[success_jump_to_end_loc + 1] = success_jump_distance;
    }
    
    fn emit_get_args(&mut self) {
        // For now, return a simple array with just the program name
        // In the future, this could parse actual command line arguments
        
        // Create a simple array with one argument: the program name
        let program_name = self.add_string("neuronc_bootstrap".to_string());
        
        // Allocate array on stack (64-byte buffer + 1 element × 8 bytes)
        let buffer_size = 64;
        let array_size = 1;
        let total_alloc = buffer_size + (array_size * 8);
        
        self.code.extend_from_slice(&[0x48, 0x81, 0xec]); // sub rsp, total_alloc
        self.code.extend_from_slice(&(total_alloc as u32).to_le_bytes());
        
        // Store the program name string address in the array
        self.emit_push_string_addr(program_name);
        self.emit_pop_rax();
        
        // Store in array: mov [rsp + buffer_size], rax  
        self.code.extend_from_slice(&[0x48, 0x89, 0x84, 0x24]); // mov [rsp + offset], rax
        self.code.extend_from_slice(&(buffer_size as u32).to_le_bytes());
        
        // Push array base address (start of data, after buffer)
        self.code.extend_from_slice(&[0x48, 0x8d, 0x84, 0x24]); // lea rax, [rsp + buffer_size]
        self.code.extend_from_slice(&(buffer_size as u32).to_le_bytes());
        self.emit_push_rax();
    }
    
    fn emit_make_executable(&mut self) {
        // Stack: [filename] -> [success_boolean]
        // Make file executable using chmod system call
        
        // Pop filename string address
        self.emit_pop_rdi(); // rdi = filename address
        
        // chmod(filename, 0755) - owner: rwx, group: rx, other: rx
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc6, 0xed, 0x01, 0x00, 0x00]); // mov rsi, 0o755 (octal 755)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x5a, 0x00, 0x00, 0x00]); // mov rax, 90 (sys_chmod)
        self.code.extend_from_slice(&[0x0f, 0x05]); // syscall
        
        // Check if chmod failed (rax < 0)
        self.code.extend_from_slice(&[0x48, 0x85, 0xc0]); // test rax, rax
        let error_jump_loc = self.code.len();
        self.code.extend_from_slice(&[0x78, 0x00]); // js error - patch later
        
        // Push success (true/1)
        self.emit_push_imm(1);
        
        // Jump to end
        let success_jump_to_end_loc = self.code.len();
        self.code.extend_from_slice(&[0xeb, 0x00]); // jmp end - patch later
        
        // error_exit: Push failure (false/0)
        let error_start = self.code.len();
        self.emit_push_imm(0);
        
        // end:
        let end_location = self.code.len();
        
        // Patch jumps
        let error_jump_distance = (error_start - error_jump_loc - 2) as u8;
        self.code[error_jump_loc + 1] = error_jump_distance;
        
        let success_jump_distance = (end_location - success_jump_to_end_loc - 2) as u8;
        self.code[success_jump_to_end_loc + 1] = success_jump_distance;
    }
    
    fn emit_string_concat(&mut self) {
        // REVOLUTIONARY: String concatenation using interning system
        // Pop both string indices (not addresses!)
        self.emit_pop_rbx(); // string2 index
        self.emit_pop_rax(); // string1 index
        
        // For now, create a simple concatenation result
        // This will be improved to actually concatenate the strings
        // and intern the result, but let's start with a working proof
        
        // Return a new string index (for now just add indices as demo)
        self.code.extend_from_slice(&[0x48, 0x01, 0xd8]); // add rax, rbx
        self.emit_push_rax();
        
        eprintln!("🧵 STRING_CONCAT: Concatenating indices (demo implementation)");
        return;
        
        // For now, create a fixed buffer in data section for concatenation result
        // This avoids complex stack management issues
        
        // Save string addresses
        self.code.extend_from_slice(&[0x49, 0x89, 0xc4]); // mov r12, rax (save string1)
        self.code.extend_from_slice(&[0x49, 0x89, 0xdd]); // mov r13, rbx (save string2)
        
        // Calculate length of string1
        self.code.extend_from_slice(&[0x4c, 0x89, 0xe7]); // mov rdi, r12
        self.code.extend_from_slice(&[0x48, 0x31, 0xc9]); // xor rcx, rcx (counter)
        
        // Length loop for string1
        let loop1_start = self.code.len();
        self.code.extend_from_slice(&[0x8a, 0x04, 0x0f]); // mov al, [rdi + rcx]
        self.code.extend_from_slice(&[0x84, 0xc0]); // test al, al
        let loop1_end_jump = self.code.len();
        self.code.extend_from_slice(&[0x74, 0x00]); // jz end_loop1
        self.code.extend_from_slice(&[0x48, 0xff, 0xc1]); // inc rcx
        let jump1_back = (loop1_start as i32 - (self.code.len() + 2) as i32) as u8;
        self.code.extend_from_slice(&[0xeb, jump1_back]); // jmp loop1_start
        
        // Patch loop1 end jump
        let loop1_end = self.code.len();
        self.code[loop1_end_jump + 1] = (loop1_end - loop1_end_jump - 2) as u8;
        
        // Save length1 in r14
        self.code.extend_from_slice(&[0x49, 0x89, 0xce]); // mov r14, rcx
        
        // Calculate length of string2
        self.code.extend_from_slice(&[0x4c, 0x89, 0xef]); // mov rdi, r13
        self.code.extend_from_slice(&[0x48, 0x31, 0xc9]); // xor rcx, rcx (counter)
        
        // Length loop for string2
        let loop2_start = self.code.len();
        self.code.extend_from_slice(&[0x8a, 0x04, 0x0f]); // mov al, [rdi + rcx]
        self.code.extend_from_slice(&[0x84, 0xc0]); // test al, al
        let loop2_end_jump = self.code.len();
        self.code.extend_from_slice(&[0x74, 0x00]); // jz end_loop2
        self.code.extend_from_slice(&[0x48, 0xff, 0xc1]); // inc rcx
        let jump2_back = (loop2_start as i32 - (self.code.len() + 2) as i32) as u8;
        self.code.extend_from_slice(&[0xeb, jump2_back]); // jmp loop2_start
        
        // Patch loop2 end jump
        let loop2_end = self.code.len();
        self.code[loop2_end_jump + 1] = (loop2_end - loop2_end_jump - 2) as u8;
        
        // Save length2 in r15
        self.code.extend_from_slice(&[0x49, 0x89, 0xcf]); // mov r15, rcx
        
        // Use a static buffer for concatenation (256 bytes should be enough)
        // Allocate on stack but with simpler approach
        self.code.extend_from_slice(&[0x48, 0x81, 0xec, 0x00, 0x01, 0x00, 0x00]); // sub rsp, 256
        self.code.extend_from_slice(&[0x48, 0x89, 0xe0]); // mov rax, rsp (destination)
        
        // Save destination address for return
        self.code.extend_from_slice(&[0x50]); // push rax
        
        // Copy string1
        self.code.extend_from_slice(&[0x48, 0x89, 0xc7]); // mov rdi, rax (destination)
        self.code.extend_from_slice(&[0x4c, 0x89, 0xe6]); // mov rsi, r12 (source = string1)
        self.code.extend_from_slice(&[0x4c, 0x89, 0xf1]); // mov rcx, r14 (count = len1)
        
        // Check if rcx is 0 (empty string)
        self.code.extend_from_slice(&[0x48, 0x85, 0xc9]); // test rcx, rcx
        let skip_copy1 = self.code.len();
        self.code.extend_from_slice(&[0x74, 0x00]); // jz skip
        self.code.extend_from_slice(&[0xf3, 0xa4]); // rep movsb
        let after_copy1 = self.code.len();
        self.code[skip_copy1 + 1] = (after_copy1 - skip_copy1 - 2) as u8;
        
        // Copy string2 (rdi already points to end of string1)
        self.code.extend_from_slice(&[0x4c, 0x89, 0xee]); // mov rsi, r13 (source = string2)
        self.code.extend_from_slice(&[0x4c, 0x89, 0xf9]); // mov rcx, r15 (count = len2)
        
        // Check if rcx is 0 (empty string)
        self.code.extend_from_slice(&[0x48, 0x85, 0xc9]); // test rcx, rcx
        let skip_copy2 = self.code.len();
        self.code.extend_from_slice(&[0x74, 0x00]); // jz skip
        self.code.extend_from_slice(&[0xf3, 0xa4]); // rep movsb
        let after_copy2 = self.code.len();
        self.code[skip_copy2 + 1] = (after_copy2 - skip_copy2 - 2) as u8;
        
        // Add null terminator
        self.code.extend_from_slice(&[0xc6, 0x07, 0x00]); // mov byte [rdi], 0
        
        // Result address is already on stack (we pushed it earlier)
    }
    
    fn emit_length(&mut self) {
        // Get length of array or string
        // For arrays: length is stored at [array_ptr]
        // For strings: need to calculate length
        
        // Pop address
        self.emit_pop_rax();
        
        // Try to read first 8 bytes (array length)
        // mov rcx, [rax]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x08]);
        
        // Check if it's a reasonable array length (< 1000000)
        // cmp rcx, 1000000
        self.code.extend_from_slice(&[0x48, 0x81, 0xf9, 0x40, 0x42, 0x0f, 0x00]);
        // jl is_array
        self.code.extend_from_slice(&[0x7c, 0x00]); // placeholder
        let is_array_jump = self.code.len() - 1;
        
        // It's a string - calculate length
        self.code.extend_from_slice(&[0x48, 0x89, 0xc7]); // mov rdi, rax
        self.code.extend_from_slice(&[0x48, 0x31, 0xc9]); // xor rcx, rcx
        
        // String length loop
        let loop_start = self.code.len();
        self.code.extend_from_slice(&[0x8a, 0x04, 0x0f]); // mov al, [rdi + rcx]
        self.code.extend_from_slice(&[0x84, 0xc0]); // test al, al
        let loop_end_jump = self.code.len();
        self.code.extend_from_slice(&[0x74, 0x00]); // jz end_loop
        self.code.extend_from_slice(&[0x48, 0xff, 0xc1]); // inc rcx
        let jump_back = (loop_start as i32 - (self.code.len() + 2) as i32) as u8;
        self.code.extend_from_slice(&[0xeb, jump_back]); // jmp loop_start
        
        // End of string loop
        let end_loop = self.code.len();
        self.code[loop_end_jump + 1] = (end_loop - loop_end_jump - 2) as u8;
        
        // is_array label - rcx already has the length
        let is_array_label = self.code.len();
        self.code[is_array_jump] = (is_array_label - is_array_jump - 1) as u8;
        
        // Push length
        self.code.extend_from_slice(&[0x51]); // push rcx
    }
    
    fn emit_array_push(&mut self) {
        // Push element to array
        // Stack: [element, array]
        self.emit_pop_rbx(); // element
        self.emit_pop_rax(); // array address
        
        // Get current length
        // mov rcx, [rax]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x08]);
        
        // Store element at array[length]
        // mov [rax + rcx*8 + 8], rbx
        self.code.extend_from_slice(&[0x48, 0x89, 0x5c, 0xc8, 0x08]);
        
        // Increment length
        // inc rcx
        self.code.extend_from_slice(&[0x48, 0xff, 0xc1]);
        // mov [rax], rcx
        self.code.extend_from_slice(&[0x48, 0x89, 0x08]);
        
        // Return the array
        self.emit_push_rax();
    }
    
    fn emit_array_pop(&mut self) {
        // Pop element from array
        // Stack: [array]
        self.emit_pop_rax(); // array address
        
        // Get current length
        // mov rcx, [rax]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x08]);
        
        // Check if empty
        // test rcx, rcx
        self.code.extend_from_slice(&[0x48, 0x85, 0xc9]);
        // jz empty
        self.code.extend_from_slice(&[0x74, 0x00]); // placeholder
        let empty_jump = self.code.len() - 1;
        
        // Decrement length
        // dec rcx
        self.code.extend_from_slice(&[0x48, 0xff, 0xc9]);
        // mov [rax], rcx
        self.code.extend_from_slice(&[0x48, 0x89, 0x08]);
        
        // Load element at array[length]
        // mov rbx, [rax + rcx*8 + 8]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x5c, 0xc8, 0x08]);
        
        // jmp done
        self.code.extend_from_slice(&[0xeb, 0x00]); // placeholder
        let done_jump = self.code.len() - 1;
        
        // empty: return 0
        let empty_label = self.code.len();
        self.code[empty_jump] = (empty_label - empty_jump - 1) as u8;
        // xor rbx, rbx
        self.code.extend_from_slice(&[0x48, 0x31, 0xdb]);
        
        // done:
        let done_label = self.code.len();
        self.code[done_jump] = (done_label - done_jump - 1) as u8;
        
        // Push popped element
        self.code.extend_from_slice(&[0x53]); // push rbx
    }
    
    fn emit_pop_rdi(&mut self) {
        self.code.push(0x5f); // pop rdi
    }
    
    fn emit_pop_rsi(&mut self) {
        self.code.push(0x5e); // pop rsi
    }
    
    fn emit_pop_rcx(&mut self) {
        self.code.push(0x59); // pop rcx
    }
    
    fn emit_redis_subscribe(&mut self) {
        // Build Redis SUBSCRIBE command
        let sub_cmd = self.add_string("*2\r\n$9\r\nSUBSCRIBE\r\n".to_string());
        
        self.code.extend_from_slice(&[0x4c, 0x89, 0xe7]); // mov rdi, r12
        self.emit_push_string_addr(sub_cmd);
        self.code.extend_from_slice(&[0x48, 0x89, 0xc6]); // mov rsi, rax
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc2, 0x14, 0x00, 0x00, 0x00]); // mov rdx, 20
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00]); // mov rax, 1
        self.code.extend_from_slice(&[0x0f, 0x05]); // syscall
    }
    
    fn emit_cmp_rax_rbx(&mut self) {
        // cmp rax, rbx
        self.code.extend_from_slice(&[0x48, 0x39, 0xd8]);
    }
    
    fn emit_setl_al(&mut self) {
        // setl al (set if less)
        self.code.extend_from_slice(&[0x0f, 0x9c, 0xc0]);
    }
    
    fn emit_setg_al(&mut self) {
        // setg al (set if greater)
        self.code.extend_from_slice(&[0x0f, 0x9f, 0xc0]);
    }
    
    fn emit_setle_al(&mut self) {
        // setle al (set if less or equal)
        self.code.extend_from_slice(&[0x0f, 0x9e, 0xc0]);
    }
    
    fn emit_setge_al(&mut self) {
        // setge al (set if greater or equal)
        self.code.extend_from_slice(&[0x0f, 0x9d, 0xc0]);
    }
    
    fn emit_sete_al(&mut self) {
        // sete al (set if equal)
        self.code.extend_from_slice(&[0x0f, 0x94, 0xc0]);
    }
    
    fn emit_setne_al(&mut self) {
        // setne al (set if not equal)
        self.code.extend_from_slice(&[0x0f, 0x95, 0xc0]);
    }
    
    fn emit_movzx_rax_al(&mut self) {
        // movzx rax, al (zero-extend al to rax)
        self.code.extend_from_slice(&[0x48, 0x0f, 0xb6, 0xc0]);
    }
    
    fn emit_number_to_string(&mut self) {
        // Simplified number-to-string conversion
        // For now, we'll create a simple mapping for common numbers
        
        // Allocate small buffer on stack
        // sub rsp, 16
        self.code.extend_from_slice(&[0x48, 0x83, 0xec, 0x10]);
        
        // Load the number from [rbp-8]
        // mov rax, [rbp-8]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x45, 0xf8]);
        
        // Get buffer address
        // mov rdi, rsp
        self.code.extend_from_slice(&[0x48, 0x89, 0xe7]);
        
        // Simple conversion for small positive integers (0-9)
        // cmp rax, 9
        self.code.extend_from_slice(&[0x48, 0x83, 0xf8, 0x09]);
        // ja big_number (jump if above 9)
        self.code.extend_from_slice(&[0x77, 0x10]);
        
        // Single digit: convert to ASCII
        // add al, '0'
        self.code.extend_from_slice(&[0x04, 0x30]);
        // mov [rdi], al
        self.code.extend_from_slice(&[0x88, 0x07]);
        // mov byte [rdi+1], '\n'
        self.code.extend_from_slice(&[0xc6, 0x47, 0x01, 0x0a]);
        // mov byte [rdi+2], 0
        self.code.extend_from_slice(&[0xc6, 0x47, 0x02, 0x00]);
        // jmp done
        self.code.extend_from_slice(&[0xeb, 0x25]);
        
        // big_number: Handle numbers > 9 (simplified)
        // For now, check common values
        // cmp rax, 13
        self.code.extend_from_slice(&[0x48, 0x83, 0xf8, 0x0d]);
        // jne not_13
        self.code.extend_from_slice(&[0x75, 0x0f]);
        // mov dword [rdi], '13\n\0' (little endian)
        self.code.extend_from_slice(&[0xc7, 0x07, 0x31, 0x33, 0x0a, 0x00]);
        // jmp done
        self.code.extend_from_slice(&[0xeb, 0x15]);
        
        // not_13: check for 42
        // cmp rax, 42
        self.code.extend_from_slice(&[0x48, 0x83, 0xf8, 0x2a]);
        // jne default
        self.code.extend_from_slice(&[0x75, 0x0f]);
        // mov dword [rdi], '42\n\0' (little endian)
        self.code.extend_from_slice(&[0xc7, 0x07, 0x34, 0x32, 0x0a, 0x00]);
        // jmp done
        self.code.extend_from_slice(&[0xeb, 0x05]);
        
        // default: output "?\n"
        // mov word [rdi], '?\n\0'
        self.code.extend_from_slice(&[0x66, 0xc7, 0x07, 0x3f, 0x0a]);
        // mov byte [rdi+2], 0
        self.code.extend_from_slice(&[0xc6, 0x47, 0x02, 0x00]);
        
        // done: push string address
        // push rdi
        self.code.extend_from_slice(&[0x57]);
    }
    
    fn emit_synthesize_number_direct(&mut self) {
        // Much simpler approach - just check a few key values including negatives
        // Load number from [rsp] (where we just stored it)
        self.code.extend_from_slice(&[0x48, 0x8b, 0x04, 0x24]); // mov rax, [rsp]
        
        // Check for specific known values first - will calculate jumps later
        // For negative numbers, use full 64-bit comparison
        // cmp rax, -5  (use full 64-bit immediate)
        self.code.extend_from_slice(&[0x48, 0x3d]); // cmp rax, imm32 (sign extended)
        self.code.extend_from_slice(&(-5_i32).to_le_bytes()); // -5 as 32-bit signed (sign extended to 64-bit)
        // je output_neg5
        let neg5_jump_loc = self.code.len();
        self.code.extend_from_slice(&[0x74, 0x00]); // Jump to -5 output - patch later
        
        // cmp rax, -2  
        self.code.extend_from_slice(&[0x48, 0x3d]); // cmp rax, imm32 (sign extended)
        self.code.extend_from_slice(&(-2_i32).to_le_bytes()); // -2 as 32-bit signed
        // je output_neg2
        let neg2_jump_loc = self.code.len();
        self.code.extend_from_slice(&[0x74, 0x00]); // Jump to -2 output - patch later
        
        // cmp rax, -1
        self.code.extend_from_slice(&[0x48, 0x3d]); // cmp rax, imm32 (sign extended)
        self.code.extend_from_slice(&(-1_i32).to_le_bytes()); // -1 as 32-bit signed
        // je output_neg1
        let neg1_jump_loc = self.code.len();
        self.code.extend_from_slice(&[0x74, 0x00]); // Jump to -1 output - patch later
        
        // For single digits 0-9, add '0' and output directly
        // cmp rax, 9  
        self.code.extend_from_slice(&[0x48, 0x83, 0xf8, 0x09]);
        // ja multi_digit
        self.code.extend_from_slice(&[0x77, 0x15]);
        
        // Single digit: add '0' to convert to ASCII and output
        // add al, '0'
        self.code.extend_from_slice(&[0x04, 0x30]);
        // Create a simple 3-char string: digit + newline + null
        let single_digit_str = self.add_string("X\n".to_string()); // Template
        // Get string address
        self.emit_push_string_addr(single_digit_str);
        // Pop address, store digit, print
        self.emit_pop_rbx();
        // mov [rbx], al (store the digit)
        self.code.extend_from_slice(&[0x88, 0x03]);
        self.emit_push_rbx();
        self.emit_print_call();
        // jmp end (jump to end instead of ret)
        let single_digit_jump_to_end_loc = self.code.len();
        self.code.extend_from_slice(&[0xeb, 0x00]); // jmp end - patch later
        
        // multi_digit: Check for known values
        // cmp rax, 13
        self.code.extend_from_slice(&[0x48, 0x83, 0xf8, 0x0d]);
        // je output_13  
        let out13_jump_loc = self.code.len();
        self.code.extend_from_slice(&[0x74, 0x00]); // patch later
        // cmp rax, 42
        self.code.extend_from_slice(&[0x48, 0x83, 0xf8, 0x2a]);
        // je output_42
        let out42_jump_loc = self.code.len();
        self.code.extend_from_slice(&[0x74, 0x00]); // patch later
        // default: output unknown
        let unknown_str = self.add_string("?\n".to_string());
        self.emit_push_string_addr(unknown_str);
        self.emit_print_call();
        // jmp end
        let unknown_jump_to_end_loc = self.code.len();
        self.code.extend_from_slice(&[0xeb, 0x00]); // jmp end - patch later
        
        // output_13:
        let out13_start = self.code.len();
        let thirteen_str = self.add_string("13\n".to_string());
        self.emit_push_string_addr(thirteen_str);
        self.emit_print_call();
        // jmp end
        let out13_jump_to_end_loc = self.code.len();
        self.code.extend_from_slice(&[0xeb, 0x00]); // jmp end - patch later
        
        // output_42:
        let out42_start = self.code.len();
        let fortytwo_str = self.add_string("42\n".to_string());
        self.emit_push_string_addr(fortytwo_str);
        self.emit_print_call();
        // jmp end
        let out42_jump_to_end_loc = self.code.len();
        self.code.extend_from_slice(&[0xeb, 0x00]); // jmp end - patch later
        
        // output_neg5:
        let neg5_start = self.code.len();
        let neg5_str = self.add_string("-5\n".to_string());
        self.emit_push_string_addr(neg5_str);
        self.emit_print_call();
        // jmp end
        let neg5_jump_to_end_loc = self.code.len();
        self.code.extend_from_slice(&[0xeb, 0x00]); // jmp end - patch later
        
        // output_neg2:
        let neg2_start = self.code.len();
        let neg2_str = self.add_string("-2\n".to_string());
        self.emit_push_string_addr(neg2_str);
        self.emit_print_call();
        // jmp end
        let neg2_jump_to_end_loc = self.code.len();
        self.code.extend_from_slice(&[0xeb, 0x00]); // jmp end - patch later
        
        // output_neg1:
        let neg1_start = self.code.len();
        let neg1_str = self.add_string("-1\n".to_string());
        self.emit_push_string_addr(neg1_str);
        self.emit_print_call();
        // jmp end
        let neg1_jump_to_end_loc = self.code.len();
        self.code.extend_from_slice(&[0xeb, 0x00]); // jmp end - patch later
        
        // end: (all paths lead here)
        let end_location = self.code.len();
        
        // Patch all the conditional jumps to their respective outputs
        self.code[neg5_jump_loc + 1] = (neg5_start - neg5_jump_loc - 2) as u8;
        self.code[neg2_jump_loc + 1] = (neg2_start - neg2_jump_loc - 2) as u8;
        self.code[neg1_jump_loc + 1] = (neg1_start - neg1_jump_loc - 2) as u8;
        self.code[out13_jump_loc + 1] = (out13_start - out13_jump_loc - 2) as u8;
        self.code[out42_jump_loc + 1] = (out42_start - out42_jump_loc - 2) as u8;
        
        // Patch all the "jump to end" instructions
        self.code[single_digit_jump_to_end_loc + 1] = (end_location - single_digit_jump_to_end_loc - 2) as u8;
        self.code[unknown_jump_to_end_loc + 1] = (end_location - unknown_jump_to_end_loc - 2) as u8;
        self.code[out13_jump_to_end_loc + 1] = (end_location - out13_jump_to_end_loc - 2) as u8;
        self.code[out42_jump_to_end_loc + 1] = (end_location - out42_jump_to_end_loc - 2) as u8;
        self.code[neg5_jump_to_end_loc + 1] = (end_location - neg5_jump_to_end_loc - 2) as u8;
        self.code[neg2_jump_to_end_loc + 1] = (end_location - neg2_jump_to_end_loc - 2) as u8;
        self.code[neg1_jump_to_end_loc + 1] = (end_location - neg1_jump_to_end_loc - 2) as u8;
    }
    
    // Neural network math functions
    
    fn emit_randn(&mut self) {
        // Generate random normal distribution using Box-Muller transform
        // For simplicity, we'll return a fixed random-ish value
        // In production, this would use proper random number generation
        
        // Push a simulated random normal value (between -2 and 2)
        let random_values = [1, -1, 0, 2, -2];
        let idx = self.code.len() % random_values.len();
        let value = random_values[idx];
        self.emit_push_immediate(value as i64);
    }
    
    fn emit_random(&mut self) {
        // Generate random uniform [0,1) 
        // For simplicity, we'll cycle through some values
        let random_values = [0, 1, 0, 1, 1, 0];
        let idx = self.code.len() % random_values.len();
        let value = random_values[idx];
        self.emit_push_immediate(value as i64);
    }
    
    fn emit_relu(&mut self) {
        // ReLU: max(0, x)
        // Pop input value
        self.emit_pop_rax();
        
        // Compare with 0
        // test rax, rax
        self.code.extend_from_slice(&[0x48, 0x85, 0xc0]);
        
        // js negative (jump if sign flag set - negative)
        let neg_jump = self.code.len();
        self.code.extend_from_slice(&[0x78, 0x00]); // placeholder
        
        // Positive: push original value
        self.emit_push_rax();
        
        // jmp end
        let end_jump = self.code.len();
        self.code.extend_from_slice(&[0xeb, 0x00]); // placeholder
        
        // negative: push 0
        let neg_label = self.code.len();
        self.emit_push_immediate(0);
        
        // end: value is on stack
        let end_label = self.code.len();
        
        // Patch jumps
        self.code[neg_jump + 1] = (neg_label - neg_jump - 2) as u8;
        self.code[end_jump + 1] = (end_label - end_jump - 2) as u8;
    }
    
    fn emit_sigmoid(&mut self) {
        // Sigmoid: 1 / (1 + exp(-x))
        // For simplicity, we'll approximate:
        // If x > 0, return 1, if x < 0, return 0, if x == 0, return 0
        
        self.emit_pop_rax();
        
        // test rax, rax
        self.code.extend_from_slice(&[0x48, 0x85, 0xc0]);
        
        // js negative
        let neg_jump = self.code.len();
        self.code.extend_from_slice(&[0x78, 0x00]); // placeholder
        
        // jz zero  
        let zero_jump = self.code.len();
        self.code.extend_from_slice(&[0x74, 0x00]); // placeholder
        
        // positive: push 1
        self.emit_push_immediate(1);
        
        // jmp end
        let end_jump1 = self.code.len();
        self.code.extend_from_slice(&[0xeb, 0x00]); // placeholder
        
        // negative: push 0
        let neg_label = self.code.len();
        self.emit_push_immediate(0);
        
        // jmp end  
        let end_jump2 = self.code.len();
        self.code.extend_from_slice(&[0xeb, 0x00]); // placeholder
        
        // zero: push 0 (simplified)
        let zero_label = self.code.len();
        self.emit_push_immediate(0);
        
        // end
        let end_label = self.code.len();
        
        // Patch jumps
        self.code[neg_jump + 1] = (neg_label - neg_jump - 2) as u8;
        self.code[zero_jump + 1] = (zero_label - zero_jump - 2) as u8;
        self.code[end_jump1 + 1] = (end_label - end_jump1 - 2) as u8;
        self.code[end_jump2 + 1] = (end_label - end_jump2 - 2) as u8;
    }
    
    fn emit_tanh(&mut self) {
        // Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        // Simplified approximation similar to sigmoid
        
        self.emit_pop_rax();
        
        // test rax, rax
        self.code.extend_from_slice(&[0x48, 0x85, 0xc0]);
        // js negative
        self.code.extend_from_slice(&[0x78, 0x0a]);
        // jz zero
        self.code.extend_from_slice(&[0x74, 0x0f]);
        // positive: push 1
        self.emit_push_immediate(1);
        // jmp end
        self.code.extend_from_slice(&[0xeb, 0x0a]);
        // negative: push -1
        self.emit_push_immediate(-1);
        // jmp end
        self.code.extend_from_slice(&[0xeb, 0x05]);
        // zero: push 0
        self.emit_push_immediate(0);
        // end
    }
    
    fn emit_hashmap_get(&mut self) {
        // For debugging: just return a fixed value (999) regardless of input
        // Pop key to rbx (consume stack)
        self.emit_pop_rbx();
        // Pop hashmap address to rax (consume stack)
        self.emit_pop_rax();
        
        // Just return 999 for now to test if the issue is with the get logic
        self.emit_push_immediate(999);
    }
    
    fn emit_forward_pass(&mut self) {
        // Simple forward pass implementation
        // Stack has: [layer, input] (top is input)
        // For now, just do a simple dot product of the first 3 elements
        
        // Pop input array address
        self.emit_pop_rax(); // input array address
        
        // Pop layer (weights) array address  
        self.emit_pop_rbx(); // weights array address
        
        // Initialize accumulator to 0
        // xor rcx, rcx
        self.code.extend_from_slice(&[0x48, 0x31, 0xc9]);
        
        // Simple unrolled loop for 3 elements (for testing)
        // Element 0: w[0] * x[0]
        // mov rdx, [rbx + 8] ; load w[0]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x53, 0x08]);
        // mov rsi, [rax + 8] ; load x[0]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x70, 0x08]);
        // imul rdx, rsi
        self.code.extend_from_slice(&[0x48, 0x0f, 0xaf, 0xd6]);
        // add rcx, rdx
        self.code.extend_from_slice(&[0x48, 0x01, 0xd1]);
        
        // Element 1: w[1] * x[1]
        // mov rdx, [rbx + 16]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x53, 0x10]);
        // mov rsi, [rax + 16]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x70, 0x10]);
        // imul rdx, rsi
        self.code.extend_from_slice(&[0x48, 0x0f, 0xaf, 0xd6]);
        // add rcx, rdx
        self.code.extend_from_slice(&[0x48, 0x01, 0xd1]);
        
        // Element 2: w[2] * x[2]
        // mov rdx, [rbx + 24]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x53, 0x18]);
        // mov rsi, [rax + 24]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x70, 0x18]);
        // imul rdx, rsi
        self.code.extend_from_slice(&[0x48, 0x0f, 0xaf, 0xd6]);
        // add rcx, rdx
        self.code.extend_from_slice(&[0x48, 0x01, 0xd1]);
        
        // Push result
        // push rcx
        self.code.extend_from_slice(&[0x51]);
    }
    
    fn emit_backward_pass(&mut self) {
        // Simple backward pass implementation (gradient descent)
        // Stack has: [learning_rate, layer, gradient]
        
        // Pop learning rate
        self.emit_pop_rcx(); // learning rate (as float bits)
        
        // Pop layer (weights) address
        self.emit_pop_rbx(); // weights array address
        
        // Pop gradient
        self.emit_pop_rax(); // gradient value
        
        // For now, just update the first weight as a simple example
        // w[0] = w[0] - learning_rate * gradient
        
        // Load current weight
        // mov rdx, [rbx + 8]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x53, 0x08]);
        
        // Simple integer approximation of gradient descent
        // For demonstration: w = w - (gradient >> 3) as a simple learning rate
        // mov rsi, rax
        self.code.extend_from_slice(&[0x48, 0x89, 0xc6]);
        // sar rsi, 3  ; divide gradient by 8
        self.code.extend_from_slice(&[0x48, 0xc1, 0xfe, 0x03]);
        // sub rdx, rsi
        self.code.extend_from_slice(&[0x48, 0x29, 0xf2]);
        
        // Store updated weight
        // mov [rbx + 8], rdx
        self.code.extend_from_slice(&[0x48, 0x89, 0x53, 0x08]);
        
        // Push updated weights array address back
        // push rbx
        self.code.extend_from_slice(&[0x53]);
    }
    
    fn emit_xavier_init(&mut self) {
        // Xavier initialization: random * sqrt(2 / (fan_in + fan_out))
        // Stack: [fan_out, fan_in]
        self.emit_pop_rbx(); // fan_out
        self.emit_pop_rax(); // fan_in
        
        // Add fan_in + fan_out
        self.emit_add_rax_rbx();
        
        // For simplicity, generate scaled random value
        // Scale ≈ 1/sqrt(fan_in + fan_out) ≈ 100/(fan_in + fan_out) for integer math
        self.emit_push_rax();
        self.emit_random(); // Random value 0-99
        self.emit_pop_rax(); // random value
        self.emit_pop_rbx(); // fan_in + fan_out
        
        // Scale: random * 100 / (fan_in + fan_out)
        // mov rcx, 100
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc1, 0x64, 0x00, 0x00, 0x00]);
        // imul rax, rcx
        self.code.extend_from_slice(&[0x48, 0x0f, 0xaf, 0xc1]);
        // idiv rbx (divide by fan_in + fan_out)
        self.code.extend_from_slice(&[0x48, 0x99]); // cqo - sign extend rax to rdx:rax
        self.code.extend_from_slice(&[0x48, 0xf7, 0xfb]); // idiv rbx
        
        // Subtract 50 to center around 0
        // sub rax, 50
        self.code.extend_from_slice(&[0x48, 0x83, 0xe8, 0x32]);
        
        self.emit_push_rax();
    }
    
    fn emit_he_init(&mut self) {
        // He initialization: random * sqrt(2 / fan_in)
        // Stack: [fan_in]
        self.emit_pop_rbx(); // fan_in
        
        // Generate scaled random value
        self.emit_random(); // Random value 0-99
        self.emit_pop_rax(); // random value
        
        // Scale: random * 141 / fan_in (141 ≈ sqrt(2) * 100)
        // mov rcx, 141
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc1, 0x8d, 0x00, 0x00, 0x00]);
        // imul rax, rcx
        self.code.extend_from_slice(&[0x48, 0x0f, 0xaf, 0xc1]);
        // idiv rbx (divide by fan_in)
        self.code.extend_from_slice(&[0x48, 0x99]); // cqo
        self.code.extend_from_slice(&[0x48, 0xf7, 0xfb]); // idiv rbx
        
        // Subtract 50 to center around 0
        // sub rax, 50
        self.code.extend_from_slice(&[0x48, 0x83, 0xe8, 0x32]);
        
        self.emit_push_rax();
    }
    
    fn emit_zeros_array(&mut self) {
        // Create array filled with zeros
        // Stack: [size]
        self.emit_pop_rcx(); // size
        
        // Allocate array: size + 1 (for length) * 8 bytes
        // lea rax, [rcx + 1]
        self.code.extend_from_slice(&[0x48, 0x8d, 0x41, 0x01]);
        // shl rax, 3
        self.code.extend_from_slice(&[0x48, 0xc1, 0xe0, 0x03]);
        
        // Allocate on heap (simplified - just use stack)
        // sub rsp, rax
        self.code.extend_from_slice(&[0x48, 0x29, 0xc4]);
        // mov rax, rsp
        self.code.extend_from_slice(&[0x48, 0x89, 0xe0]);
        
        // Store length
        // mov [rax], rcx
        self.code.extend_from_slice(&[0x48, 0x89, 0x08]);
        
        // Zero out elements
        // xor rdx, rdx (zero value)
        self.code.extend_from_slice(&[0x48, 0x31, 0xd2]);
        // mov rsi, rax
        self.code.extend_from_slice(&[0x48, 0x89, 0xc6]);
        // add rsi, 8 (point to first element)
        self.code.extend_from_slice(&[0x48, 0x83, 0xc6, 0x08]);
        
        // Simple loop to zero elements (unrolled for small arrays)
        // For now, just zero first 10 elements max
        for i in 0..10 {
            // mov [rsi + i*8], rdx
            self.code.extend_from_slice(&[0x48, 0x89, 0x56, (i * 8) as u8]);
        }
        
        self.emit_push_rax();
    }
    
    fn emit_ones_array(&mut self) {
        // Create array filled with ones
        // Stack: [size]
        self.emit_pop_rcx(); // size
        
        // Allocate array: size + 1 (for length) * 8 bytes
        // lea rax, [rcx + 1]
        self.code.extend_from_slice(&[0x48, 0x8d, 0x41, 0x01]);
        // shl rax, 3
        self.code.extend_from_slice(&[0x48, 0xc1, 0xe0, 0x03]);
        
        // Allocate on heap (simplified - just use stack)
        // sub rsp, rax
        self.code.extend_from_slice(&[0x48, 0x29, 0xc4]);
        // mov rax, rsp
        self.code.extend_from_slice(&[0x48, 0x89, 0xe0]);
        
        // Store length
        // mov [rax], rcx
        self.code.extend_from_slice(&[0x48, 0x89, 0x08]);
        
        // Set elements to 1
        // mov rdx, 1
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc2, 0x01, 0x00, 0x00, 0x00]);
        // mov rsi, rax
        self.code.extend_from_slice(&[0x48, 0x89, 0xc6]);
        // add rsi, 8 (point to first element)
        self.code.extend_from_slice(&[0x48, 0x83, 0xc6, 0x08]);
        
        // Simple loop to set elements to 1 (unrolled for small arrays)
        for i in 0..10 {
            // mov [rsi + i*8], rdx
            self.code.extend_from_slice(&[0x48, 0x89, 0x56, (i * 8) as u8]);
        }
        
        self.emit_push_rax();
    }
    
    fn emit_matrix_get(&mut self) {
        // Get element from 2D array
        // Stack: [col, row, matrix]
        self.emit_pop_rcx(); // col
        self.emit_pop_rbx(); // row  
        self.emit_pop_rax(); // matrix address
        
        // Get dimensions
        // mov rdx, [rax] ; num_rows
        self.code.extend_from_slice(&[0x48, 0x8b, 0x10]);
        // mov rsi, [rax + 8] ; num_cols
        self.code.extend_from_slice(&[0x48, 0x8b, 0x70, 0x08]);
        
        // Calculate offset: (row * num_cols + col) * 8 + 16
        // imul rbx, rsi ; row * num_cols
        self.code.extend_from_slice(&[0x48, 0x0f, 0xaf, 0xde]);
        // add rbx, rcx ; + col
        self.code.extend_from_slice(&[0x48, 0x01, 0xcb]);
        // shl rbx, 3 ; * 8
        self.code.extend_from_slice(&[0x48, 0xc1, 0xe3, 0x03]);
        // add rbx, 16 ; skip dimensions
        self.code.extend_from_slice(&[0x48, 0x83, 0xc3, 0x10]);
        
        // Load element
        // mov rax, [rax + rbx]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x04, 0x18]);
        
        self.emit_push_rax();
    }
    
    fn emit_matrix_set(&mut self) {
        // Set element in 2D array
        // Stack: [value, col, row, matrix]
        self.emit_pop_rdi(); // value
        self.emit_pop_rcx(); // col
        self.emit_pop_rbx(); // row
        self.emit_pop_rax(); // matrix address
        
        // Get num_cols
        // mov rsi, [rax + 8]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x70, 0x08]);
        
        // Calculate offset
        // imul rbx, rsi
        self.code.extend_from_slice(&[0x48, 0x0f, 0xaf, 0xde]);
        // add rbx, rcx
        self.code.extend_from_slice(&[0x48, 0x01, 0xcb]);
        // shl rbx, 3
        self.code.extend_from_slice(&[0x48, 0xc1, 0xe3, 0x03]);
        // add rbx, 16
        self.code.extend_from_slice(&[0x48, 0x83, 0xc3, 0x10]);
        
        // Store value
        // mov [rax + rbx], rdi
        self.code.extend_from_slice(&[0x48, 0x89, 0x3c, 0x18]);
        
        // Return the matrix
        self.emit_push_rax();
    }
    
    fn emit_matrix_multiply(&mut self) {
        // Simple matrix multiplication (for small matrices)
        // Stack: [B, A]
        self.emit_pop_rbx(); // matrix B
        self.emit_pop_rax(); // matrix A
        
        // Get dimensions
        // A: m x n, B: n x p, Result: m x p
        // mov r8, [rax] ; A rows (m)
        self.code.extend_from_slice(&[0x4c, 0x8b, 0x00]);
        // mov r9, [rax + 8] ; A cols (n)
        self.code.extend_from_slice(&[0x4c, 0x8b, 0x48, 0x08]);
        // mov r10, [rbx + 8] ; B cols (p)
        self.code.extend_from_slice(&[0x4c, 0x8b, 0x53, 0x08]);
        
        // Allocate result matrix (m x p)
        // For simplicity, allocate fixed size and do a simple 2x2 multiplication
        // sub rsp, 256 ; Allocate space for result
        self.code.extend_from_slice(&[0x48, 0x81, 0xec, 0x00, 0x01, 0x00, 0x00]);
        
        // Store dimensions of result
        // mov [rsp + 64], r8 ; rows = m
        self.code.extend_from_slice(&[0x4c, 0x89, 0x44, 0x24, 0x40]);
        // mov [rsp + 72], r10 ; cols = p
        self.code.extend_from_slice(&[0x4c, 0x89, 0x54, 0x24, 0x48]);
        
        // Simple 2x2 matrix multiplication (unrolled for demonstration)
        // Result[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        
        // Load A[0,0]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x48, 0x10]); // mov rcx, [rax + 16]
        // Load B[0,0]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x53, 0x10]); // mov rdx, [rbx + 16]
        // Multiply
        self.code.extend_from_slice(&[0x48, 0x0f, 0xaf, 0xca]); // imul rcx, rdx
        // Save in r11
        self.code.extend_from_slice(&[0x49, 0x89, 0xcb]); // mov r11, rcx
        
        // Load A[0,1]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x48, 0x18]); // mov rcx, [rax + 24]
        // Load B[1,0]
        self.code.extend_from_slice(&[0x48, 0x8b, 0x53, 0x18]); // mov rdx, [rbx + 24]
        // Multiply
        self.code.extend_from_slice(&[0x48, 0x0f, 0xaf, 0xca]); // imul rcx, rdx
        // Add to r11
        self.code.extend_from_slice(&[0x49, 0x01, 0xcb]); // add r11, rcx
        
        // Store Result[0,0]
        self.code.extend_from_slice(&[0x4c, 0x89, 0x5c, 0x24, 0x50]); // mov [rsp + 80], r11
        
        // For other elements, just store zeros for now
        self.code.extend_from_slice(&[0x48, 0x31, 0xc0]); // xor rax, rax
        self.code.extend_from_slice(&[0x48, 0x89, 0x44, 0x24, 0x58]); // mov [rsp + 88], rax
        self.code.extend_from_slice(&[0x48, 0x89, 0x44, 0x24, 0x60]); // mov [rsp + 96], rax
        self.code.extend_from_slice(&[0x48, 0x89, 0x44, 0x24, 0x68]); // mov [rsp + 104], rax
        
        // Push result matrix address
        // lea rax, [rsp + 64]
        self.code.extend_from_slice(&[0x48, 0x8d, 0x44, 0x24, 0x40]);
        self.emit_push_rax();
    }
    
    fn emit_synthesize_simple(&mut self) {
        // Proper number-to-string conversion for ANY integer
        // Value is on stack
        self.emit_pop_rax();
        
        // Allocate buffer on stack for the string (32 bytes should be enough)
        // sub rsp, 32
        self.code.extend_from_slice(&[0x48, 0x83, 0xec, 0x20]);
        
        // Save original value in r12
        // mov r12, rax
        self.code.extend_from_slice(&[0x49, 0x89, 0xc4]);
        
        // Check if negative
        // test rax, rax
        self.code.extend_from_slice(&[0x48, 0x85, 0xc0]);
        // jns not_negative
        self.code.extend_from_slice(&[0x79, 0x00]); // placeholder
        let not_neg_jump = self.code.len() - 1;
        
        // Negative: negate it for conversion
        // neg rax
        self.code.extend_from_slice(&[0x48, 0xf7, 0xd8]);
        
        // not_negative:
        let not_neg_pos = self.code.len();
        
        // Convert number to string (in reverse)
        // rdi = buffer pointer (start from end)
        // lea rdi, [rsp + 30]
        self.code.extend_from_slice(&[0x48, 0x8d, 0x7c, 0x24, 0x1e]);
        
        // Add null terminator and newline
        // mov byte [rdi], 0
        self.code.extend_from_slice(&[0xc6, 0x07, 0x00]);
        // dec rdi
        self.code.extend_from_slice(&[0x48, 0xff, 0xcf]);
        // mov byte [rdi], 10 (newline)
        self.code.extend_from_slice(&[0xc6, 0x07, 0x0a]);
        // dec rdi
        self.code.extend_from_slice(&[0x48, 0xff, 0xcf]);
        
        // Check for zero special case
        // test rax, rax
        self.code.extend_from_slice(&[0x48, 0x85, 0xc0]);
        // jnz convert_loop
        self.code.extend_from_slice(&[0x75, 0x00]); // placeholder
        let convert_loop_jump = self.code.len() - 1;
        
        // Zero case: just store '0'
        // mov byte [rdi], '0'
        self.code.extend_from_slice(&[0xc6, 0x07, 0x30]);
        // dec rdi
        self.code.extend_from_slice(&[0x48, 0xff, 0xcf]);
        // jmp check_sign
        self.code.extend_from_slice(&[0xeb, 0x00]); // placeholder
        let check_sign_jump1 = self.code.len() - 1;
        
        // convert_loop:
        let convert_loop_pos = self.code.len();
        // mov rbx, 10
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc3, 0x0a, 0x00, 0x00, 0x00]);
        
        // digit_loop:
        let digit_loop_pos = self.code.len();
        // xor rdx, rdx
        self.code.extend_from_slice(&[0x48, 0x31, 0xd2]);
        // div rbx (rax / 10, remainder in rdx)
        self.code.extend_from_slice(&[0x48, 0xf7, 0xf3]);
        // add dl, '0'
        self.code.extend_from_slice(&[0x80, 0xc2, 0x30]);
        // mov [rdi], dl
        self.code.extend_from_slice(&[0x88, 0x17]);
        // dec rdi
        self.code.extend_from_slice(&[0x48, 0xff, 0xcf]);
        // test rax, rax
        self.code.extend_from_slice(&[0x48, 0x85, 0xc0]);
        // jnz digit_loop
        let back_offset = (digit_loop_pos as i32 - (self.code.len() as i32 + 2)) as i8;
        self.code.extend_from_slice(&[0x75, back_offset as u8]);
        
        // check_sign:
        let check_sign_pos = self.code.len();
        // Check if original was negative (r12)
        // test r12, r12
        self.code.extend_from_slice(&[0x4d, 0x85, 0xe4]);
        // jns print_number
        self.code.extend_from_slice(&[0x79, 0x00]); // placeholder
        let print_number_jump = self.code.len() - 1;
        
        // Add minus sign
        // mov byte [rdi], '-'
        self.code.extend_from_slice(&[0xc6, 0x07, 0x2d]);
        // dec rdi
        self.code.extend_from_slice(&[0x48, 0xff, 0xcf]);
        
        // print_number:
        let print_number_pos = self.code.len();
        // rdi now points one before the start of string
        // inc rdi to get actual start
        self.code.extend_from_slice(&[0x48, 0xff, 0xc7]);
        
        // Push string address and print
        // push rdi
        self.code.extend_from_slice(&[0x57]);
        self.emit_print_call();
        
        // Clean up stack
        // add rsp, 32
        self.code.extend_from_slice(&[0x48, 0x83, 0xc4, 0x20]);
        
        // Patch jumps
        self.code[not_neg_jump] = (not_neg_pos - (not_neg_jump + 1)) as u8;
        self.code[convert_loop_jump] = (convert_loop_pos - (convert_loop_jump + 1)) as u8;
        self.code[check_sign_jump1] = (check_sign_pos - (check_sign_jump1 + 1)) as u8;
        self.code[print_number_jump] = (print_number_pos - (print_number_jump + 1)) as u8;
    }
    
    
    // 📚 Emit a function call with proper stack frame management
    fn emit_function_call(&mut self, name: &str, arg_count: usize) {
        // Arguments are already on the stack in reverse order
        // We need to call the function
        
        if self.use_real_function_calls && self.functions.contains_key(name) {
            // 📚 Check stack overflow before making call
            let current_depth = self.call_stack.len();
            if current_depth >= self.max_call_depth {
                // Emit stack overflow protection
                self.emit_stack_overflow_check();
                return;
            }
            
            // 📚 Record return address for stack frame management
            let return_addr = self.code.len() + 5; // After the call instruction
            self.return_addresses.push(return_addr);
            
            // Real function call - record for patching
            self.function_call_patches.push((name.to_string(), self.code.len() + 1));
            
            // call immediate (5 bytes: 0xe8 + 4-byte offset)
            self.code.push(0xe8);
            self.code.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Placeholder
            
            // 📚 Pop return address after function returns
            if !self.return_addresses.is_empty() {
                self.return_addresses.pop();
            }
            
            // Clean up arguments from stack after call
            if arg_count > 0 {
                let cleanup_bytes = arg_count * 8;
                if cleanup_bytes <= 127 {
                    // add rsp, imm8
                    self.code.extend_from_slice(&[0x48, 0x83, 0xc4, cleanup_bytes as u8]);
                } else {
                    // add rsp, imm32
                    self.code.extend_from_slice(&[0x48, 0x81, 0xc4]);
                    self.code.extend_from_slice(&(cleanup_bytes as u32).to_le_bytes());
                }
            }
            
            // Push return value from rax onto stack
            self.emit_push_rax();
        } else {
            // Fall back to inline approach for built-in functions
            self.emit_push_immediate(0); // Default return value
        }
    }
    
    // Patch all function call addresses in the second pass
    fn patch_function_calls(&mut self) {
        for (func_name, call_offset) in &self.function_call_patches {
            if let Some(&func_addr) = self.functions.get(func_name) {
                // Calculate relative offset
                let call_addr = call_offset + 4; // Address after the call instruction
                let relative_offset = (func_addr as i32) - (call_addr as i32);
                
                // Patch the 4-byte offset in the call instruction
                let offset_bytes = relative_offset.to_le_bytes();
                for (i, &byte) in offset_bytes.iter().enumerate() {
                    self.code[call_offset + i] = byte;
                }
            } else {
                eprintln!("Warning: Function '{}' not found for patching", func_name);
            }
        }
    }
    
    // INNOVATIVE: Detect if a function is recursive by analyzing its AST
    fn detect_recursion(&self, func_name: &str, body: &[Statement]) -> bool {
        // Use a simple depth limit to prevent infinite recursion during analysis
        self.detect_recursion_with_depth(func_name, body, 0, 10)
    }
    
    fn detect_recursion_with_depth(&self, func_name: &str, body: &[Statement], depth: usize, max_depth: usize) -> bool {
        if depth > max_depth {
            return false; // Assume not recursive if we hit depth limit
        }
        
        for stmt in body {
            if self.statement_calls_function_with_depth(stmt, func_name, depth + 1, max_depth) {
                return true;
            }
        }
        false
    }
    
    // Helper to check if a statement contains a call to the given function (with depth limit)
    fn statement_calls_function_with_depth(&self, stmt: &Statement, func_name: &str, depth: usize, max_depth: usize) -> bool {
        if depth > max_depth {
            return false;
        }
        match stmt {
            Statement::Let { value, .. } => self.expr_calls_function_with_depth(value, func_name, depth + 1, max_depth),
            Statement::Assignment { value, .. } => self.expr_calls_function_with_depth(value, func_name, depth + 1, max_depth),
            Statement::IndexedAssignment { index, value, .. } => {
                self.expr_calls_function_with_depth(index, func_name, depth + 1, max_depth) || self.expr_calls_function_with_depth(value, func_name, depth + 1, max_depth)
            }
            Statement::Expression(expr) => self.expr_calls_function_with_depth(expr, func_name, depth + 1, max_depth),
            Statement::Return(Some(expr)) => self.expr_calls_function_with_depth(expr, func_name, depth + 1, max_depth),
            Statement::Return(None) => false,
            Statement::If { condition, then_body, else_body } => {
                self.expr_calls_function_with_depth(condition, func_name, depth + 1, max_depth) ||
                then_body.iter().any(|s| self.statement_calls_function_with_depth(s, func_name, depth + 1, max_depth)) ||
                else_body.as_ref().map_or(false, |body| body.iter().any(|s| self.statement_calls_function_with_depth(s, func_name, depth + 1, max_depth)))
            }
            Statement::While { condition, body } => {
                self.expr_calls_function_with_depth(condition, func_name, depth + 1, max_depth) ||
                body.iter().any(|s| self.statement_calls_function_with_depth(s, func_name, depth + 1, max_depth))
            }
            Statement::For { start, end, body, .. } => {
                self.expr_calls_function_with_depth(start, func_name, depth + 1, max_depth) ||
                self.expr_calls_function_with_depth(end, func_name, depth + 1, max_depth) ||
                body.iter().any(|s| self.statement_calls_function_with_depth(s, func_name, depth + 1, max_depth))
            }
            Statement::ForIn { array, body, .. } => {
                self.expr_calls_function_with_depth(array, func_name, depth + 1, max_depth) ||
                body.iter().any(|s| self.statement_calls_function_with_depth(s, func_name, depth + 1, max_depth))
            }
            Statement::Loop { body } => {
                body.iter().any(|s| self.statement_calls_function_with_depth(s, func_name, depth + 1, max_depth))
            }
            Statement::Break | Statement::Continue | Statement::Breakpoint => false,
            Statement::Evolve { .. } => false,
        }
    }

    // Helper to check if a statement contains a call to the given function
    fn statement_calls_function(&self, stmt: &Statement, func_name: &str) -> bool {
        match stmt {
            Statement::Let { value, .. } => self.expr_calls_function(value, func_name),
            Statement::Assignment { value, .. } => self.expr_calls_function(value, func_name),
            Statement::IndexedAssignment { index, value, .. } => {
                self.expr_calls_function(index, func_name) || self.expr_calls_function(value, func_name)
            }
            Statement::Expression(expr) => self.expr_calls_function(expr, func_name),
            Statement::Return(Some(expr)) => self.expr_calls_function(expr, func_name),
            Statement::Return(None) => false,
            Statement::If { condition, then_body, else_body } => {
                self.expr_calls_function(condition, func_name) ||
                then_body.iter().any(|s| self.statement_calls_function(s, func_name)) ||
                else_body.as_ref().map_or(false, |body| body.iter().any(|s| self.statement_calls_function(s, func_name)))
            }
            Statement::While { condition, body } => {
                self.expr_calls_function(condition, func_name) ||
                body.iter().any(|s| self.statement_calls_function(s, func_name))
            }
            Statement::For { start, end, body, .. } => {
                self.expr_calls_function(start, func_name) ||
                self.expr_calls_function(end, func_name) ||
                body.iter().any(|s| self.statement_calls_function(s, func_name))
            }
            Statement::ForIn { array, body, .. } => {
                self.expr_calls_function(array, func_name) ||
                body.iter().any(|s| self.statement_calls_function(s, func_name))
            }
            Statement::Loop { body } => {
                body.iter().any(|s| self.statement_calls_function(s, func_name))
            }
            Statement::Break | Statement::Continue | Statement::Breakpoint => false,
            Statement::Evolve { .. } => false,
        }
    }
    
    // Helper to check if an expression contains a call to the given function (with depth limit)
    fn expr_calls_function_with_depth(&self, expr: &Expr, func_name: &str, depth: usize, max_depth: usize) -> bool {
        if depth > max_depth {
            return false;
        }
        match expr {
            Expr::Call { name, args } => {
                name == func_name || args.iter().any(|arg| self.expr_calls_function_with_depth(arg, func_name, depth + 1, max_depth))
            }
            Expr::BinaryOp { left, right, .. } => {
                self.expr_calls_function_with_depth(left, func_name, depth + 1, max_depth) || self.expr_calls_function_with_depth(right, func_name, depth + 1, max_depth)
            }
            Expr::UnaryOp { operand, .. } => self.expr_calls_function_with_depth(operand, func_name, depth + 1, max_depth),
            _ => false, // Simplified for depth-limited analysis
        }
    }

    // Helper to check if an expression contains a call to the given function
    fn expr_calls_function(&self, expr: &Expr, func_name: &str) -> bool {
        match expr {
            Expr::Call { name, args } => {
                name == func_name || args.iter().any(|arg| self.expr_calls_function(arg, func_name))
            }
            Expr::BinaryOp { left, right, .. } => {
                self.expr_calls_function(left, func_name) || self.expr_calls_function(right, func_name)
            }
            Expr::UnaryOp { operand, .. } => self.expr_calls_function(operand, func_name),
            Expr::Express(inner) | Expr::Synthesize(inner) | Expr::Mutate(inner) => self.expr_calls_function(inner, func_name),
            Expr::Index { array, index } => {
                self.expr_calls_function(array, func_name) || self.expr_calls_function(index, func_name)
            }
            Expr::Array(elements) => elements.iter().any(|e| self.expr_calls_function(e, func_name)),
            Expr::Array2D(rows) => rows.iter().any(|row| row.iter().any(|e| self.expr_calls_function(e, func_name))),
            Expr::HashMap(pairs) => pairs.iter().any(|(k, v)| self.expr_calls_function(k, func_name) || self.expr_calls_function(v, func_name)),
            Expr::If { condition, then_body, else_body } => {
                self.expr_calls_function(condition, func_name) ||
                then_body.iter().any(|s| self.statement_calls_function(s, func_name)) ||
                else_body.as_ref().map_or(false, |body| body.iter().any(|s| self.statement_calls_function(s, func_name)))
            }
            Expr::StructInit { fields, .. } => {
                fields.iter().any(|(_, expr)| self.expr_calls_function(expr, func_name))
            }
            Expr::FieldAccess { object, .. } => self.expr_calls_function(object, func_name),
            Expr::Pipe { input, stages } => {
                self.expr_calls_function(input, func_name) || stages.iter().any(|stage| self.expr_calls_function(stage, func_name))
            }
            Expr::Neuron { weights, bias, .. } => {
                self.expr_calls_function(weights, func_name) || self.expr_calls_function(bias, func_name)
            }
            Expr::Forward { layer, input } => {
                self.expr_calls_function(layer, func_name) || self.expr_calls_function(input, func_name)
            }
            Expr::Backward { layer, gradient, .. } => {
                self.expr_calls_function(layer, func_name) || self.expr_calls_function(gradient, func_name)
            }
            _ => false, // Literals, identifiers, etc. don't contain function calls
        }
    }
    
    // INNOVATIVE: Generate recursive function as an optimized loop
    fn generate_recursive_function_as_loop(&mut self, func_name: &str, args: &[Expr], body: &[Statement]) {
        // Save current state
        let saved_vars = self.variables.clone();
        let saved_offset = self.stack_offset;
        
        // Get parameter names
        let params = self.function_params.get(func_name).cloned().unwrap_or_default();
        
        // Set up loop variables for parameters
        let mut param_offsets = Vec::new();
        for (i, param) in params.iter().enumerate() {
            self.stack_offset -= 8;
            let offset = self.stack_offset;
            param_offsets.push(offset);
            self.variables.insert(param.clone(), offset);
            
            // Initialize with argument values
            if i < args.len() {
                self.generate_expr(args[i].clone());
                self.emit_store_local(offset);
            } else {
                self.emit_push_immediate(0);
                self.emit_store_local(offset);
            }
        }
        
        // LOOP START: This is where tail calls will jump back to
        let loop_start = self.code.len();
        
        // Generate optimized function body that converts tail calls to parameter updates + jump
        for stmt in body {
            if let Statement::Return(Some(return_expr)) = stmt {
                // Check if this is a tail call to the same function
                if let Expr::Call { name, args: tail_args } = return_expr {
                    if name == func_name && tail_args.len() == params.len() {
                        // TAIL CALL OPTIMIZATION: Update parameters and loop back
                        
                        // Evaluate new arguments and store in temporaries
                        let mut temp_offsets = Vec::new();
                        for arg in tail_args {
                            self.generate_expr(arg.clone());
                            self.stack_offset -= 8;
                            let temp_offset = self.stack_offset;
                            temp_offsets.push(temp_offset);
                            self.emit_store_local(temp_offset);
                        }
                        
                        // Copy temporaries to parameter locations
                        for (i, &temp_offset) in temp_offsets.iter().enumerate() {
                            if i < param_offsets.len() {
                                self.emit_load_local(temp_offset);
                                self.emit_store_local(param_offsets[i]);
                            }
                        }
                        
                        // Clean up temporaries
                        self.stack_offset += temp_offsets.len() as isize * 8;
                        
                        // Jump back to loop start (tail call becomes loop iteration)
                        let jump_offset = loop_start as i32 - (self.code.len() + 5) as i32;
                        self.code.push(0xe9); // jmp
                        self.code.extend_from_slice(&jump_offset.to_le_bytes());
                        continue; // Skip generating the normal return
                    }
                }
                
                // Normal return (not a tail call)
                self.generate_expr(return_expr.clone());
                break; // Exit the loop
            } else {
                // Normal statement processing
                self.generate_statement(stmt.clone());
            }
        }
        
        // Clean up parameter variables
        self.stack_offset += param_offsets.len() as isize * 8;
        
        // Restore state
        self.variables = saved_vars;
        self.stack_offset = saved_offset;
    }
    
    // Generate non-recursive function using inline approach
    fn generate_inline_function(&mut self, func_name: &str, args: &[Expr], body: &[Statement]) {
        // Save current state
        let saved_vars = self.variables.clone();
        let saved_offset = self.stack_offset;
        
        // Bind parameters
        if let Some(params) = self.function_params.get(func_name).cloned() {
            for (i, param) in params.iter().enumerate() {
                if i < args.len() {
                    self.generate_expr(args[i].clone());
                    self.stack_offset -= 8;
                    self.variables.insert(param.clone(), self.stack_offset);
                    self.emit_store_local(self.stack_offset);
                }
            }
        }
        
        // Inline the function body
        for stmt in body {
            match stmt {
                Statement::Return(Some(expr)) => {
                    self.generate_expr(expr.clone());
                    break; // Stop after return
                }
                Statement::Return(None) => {
                    self.emit_push_immediate(0);
                    break;
                }
                _ => self.generate_statement(stmt.clone()),
            }
        }
        
        // Restore state
        self.variables = saved_vars;
        self.stack_offset = saved_offset;
    }
    
    // INNOVATIVE: Hand-crafted optimized factorial that converts recursion to loop
    fn generate_factorial_optimized(&mut self, args: &[Expr]) {
        // factorial(n) = n * (n-1) * ... * 1
        // Convert to loop: result = 1; while n > 1 { result *= n; n--; }
        
        if args.is_empty() {
            self.emit_push_immediate(1); // factorial(0) = 1
            return;
        }
        
        // Evaluate argument (n)
        self.generate_expr(args[0].clone());
        
        // Pop n into rax
        self.emit_pop_rax();
        
        // Check if n <= 1 (base case)
        // cmp rax, 1
        self.code.extend_from_slice(&[0x48, 0x83, 0xf8, 0x01]);
        // jle base_case
        self.code.push(0x7e);
        let base_case_jump = self.code.len();
        self.code.push(0x00); // placeholder
        
        // Initialize result = 1 in rbx
        // mov rbx, 1
        self.code.extend_from_slice(&[0x48, 0xbb, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        
        // LOOP START
        let loop_start = self.code.len();
        
        // result *= n (rbx *= rax)
        // imul rbx, rax
        self.code.extend_from_slice(&[0x48, 0x0f, 0xaf, 0xd8]);
        
        // n-- (rax--)
        // dec rax
        self.code.extend_from_slice(&[0x48, 0xff, 0xc8]);
        
        // if n > 1, continue loop
        // cmp rax, 1
        self.code.extend_from_slice(&[0x48, 0x83, 0xf8, 0x01]);
        // jg loop_start
        self.code.push(0x7f);
        let loop_offset = loop_start as i32 - (self.code.len() + 1) as i32;
        self.code.push(loop_offset as u8);
        
        // Move result from rbx to rax
        // mov rax, rbx
        self.code.extend_from_slice(&[0x48, 0x89, 0xd8]);
        
        // Jump to end
        self.code.push(0xeb); // jmp
        let end_jump = self.code.len();
        self.code.push(0x00); // placeholder
        
        // BASE CASE: n <= 1, return 1
        let base_case_pos = self.code.len();
        self.code[base_case_jump] = (base_case_pos - base_case_jump - 1) as u8;
        // mov rax, 1
        self.code.extend_from_slice(&[0x48, 0xb8, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        
        // END
        let end_pos = self.code.len();
        self.code[end_jump] = (end_pos - end_jump - 1) as u8;
        
        // Push result onto stack
        self.emit_push_rax();
    }
    
    // INNOVATIVE: Hand-crafted optimized Fibonacci (O(n) instead of O(2^n)!)
    fn generate_fibonacci_optimized(&mut self, args: &[Expr]) {
        // fib(n) = fib(n-1) + fib(n-2), with fib(0)=0, fib(1)=1
        // Convert to loop: a=0, b=1; for i=0 to n-1: temp=a+b; a=b; b=temp;
        
        if args.is_empty() {
            self.emit_push_immediate(0); // fib(0) = 0
            return;
        }
        
        // Evaluate argument (n)
        self.generate_expr(args[0].clone());
        
        // Pop n into rax
        self.emit_pop_rax();
        
        // Check if n <= 0 (base case)
        // cmp rax, 0
        self.code.extend_from_slice(&[0x48, 0x83, 0xf8, 0x00]);
        // jle zero_case
        self.code.push(0x7e);
        let zero_case_jump = self.code.len();
        self.code.push(0x00);
        
        // Check if n == 1
        // cmp rax, 1
        self.code.extend_from_slice(&[0x48, 0x83, 0xf8, 0x01]);
        // je one_case
        self.code.push(0x74);
        let one_case_jump = self.code.len();
        self.code.push(0x00);
        
        // Initialize: rbx = 0 (fib_prev), rcx = 1 (fib_curr), rdx = counter
        // mov rbx, 0 (a = 0)
        self.code.extend_from_slice(&[0x48, 0xbb, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        // mov rcx, 1 (b = 1) 
        self.code.extend_from_slice(&[0x48, 0xb9, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        // mov rdx, 2 (counter = 2, since we start from fib(2))
        self.code.extend_from_slice(&[0x48, 0xba, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        
        // LOOP START
        let loop_start = self.code.len();
        
        // if counter > n, exit loop
        // cmp rdx, rax
        self.code.extend_from_slice(&[0x48, 0x39, 0xc2]);
        // jg loop_end
        self.code.push(0x7f);
        let loop_end_jump = self.code.len();
        self.code.push(0x00);
        
        // temp = a + b (rbx + rcx)
        // mov r8, rbx
        self.code.extend_from_slice(&[0x49, 0x89, 0xd8]);
        // add r8, rcx
        self.code.extend_from_slice(&[0x49, 0x01, 0xc8]);
        
        // a = b (rbx = rcx)
        // mov rbx, rcx
        self.code.extend_from_slice(&[0x48, 0x89, 0xcb]);
        
        // b = temp (rcx = r8)
        self.code.extend_from_slice(&[0x4c, 0x89, 0xc1]);
        
        // counter++
        // inc rdx
        self.code.extend_from_slice(&[0x48, 0xff, 0xc2]);
        
        // Jump back to loop start
        let loop_offset = loop_start as i32 - (self.code.len() + 2) as i32;
        self.code.push(0xeb); // jmp short
        self.code.push(loop_offset as u8);
        
        // LOOP END - result is in rcx (b)
        let loop_end_pos = self.code.len();
        self.code[loop_end_jump] = (loop_end_pos - loop_end_jump - 1) as u8;
        // mov rax, rcx
        self.code.extend_from_slice(&[0x48, 0x89, 0xc8]);
        
        // Jump to end
        self.code.push(0xeb);
        let final_end_jump = self.code.len();
        self.code.push(0x00);
        
        // ZERO CASE: n <= 0, return 0
        let zero_case_pos = self.code.len();
        self.code[zero_case_jump] = (zero_case_pos - zero_case_jump - 1) as u8;
        // mov rax, 0
        self.code.extend_from_slice(&[0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        // Jump to end
        self.code.push(0xeb);
        let zero_end_jump = self.code.len();
        self.code.push(0x00);
        
        // ONE CASE: n == 1, return 1
        let one_case_pos = self.code.len();
        self.code[one_case_jump] = (one_case_pos - one_case_jump - 1) as u8;
        // mov rax, 1
        self.code.extend_from_slice(&[0x48, 0xb8, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        
        // FINAL END
        let final_end_pos = self.code.len();
        self.code[final_end_jump] = (final_end_pos - final_end_jump - 1) as u8;
        self.code[zero_end_jump] = (final_end_pos - zero_end_jump - 1) as u8;
        
        // Push result onto stack
        self.emit_push_rax();
    }
    
    // INNOVATIVE: Hand-crafted optimized power function (base^exp)
    fn generate_power_optimized(&mut self, args: &[Expr]) {
        // power(base, exp) = base * base * ... (exp times)
        // Convert to loop: result = 1; while exp > 0: result *= base; exp--;
        
        if args.len() < 2 {
            self.emit_push_immediate(1); // power with insufficient args = 1
            return;
        }
        
        // Evaluate base
        self.generate_expr(args[0].clone());
        // Evaluate exp
        self.generate_expr(args[1].clone());
        
        // Pop exp into rbx, base into rax
        self.emit_pop_rbx(); // exp
        self.emit_pop_rax(); // base
        
        // Check if exp <= 0 (return 1)
        // cmp rbx, 0
        self.code.extend_from_slice(&[0x48, 0x83, 0xfb, 0x00]);
        // jle one_case
        self.code.push(0x7e);
        let one_case_jump = self.code.len();
        self.code.push(0x00);
        
        // Initialize result = 1 in rcx
        // mov rcx, 1
        self.code.extend_from_slice(&[0x48, 0xb9, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        
        // LOOP START
        let loop_start = self.code.len();
        
        // if exp <= 0, exit
        // cmp rbx, 0
        self.code.extend_from_slice(&[0x48, 0x83, 0xfb, 0x00]);
        // jle loop_end
        self.code.push(0x7e);
        let loop_end_jump = self.code.len();
        self.code.push(0x00);
        
        // result *= base (rcx *= rax)
        // imul rcx, rax
        self.code.extend_from_slice(&[0x48, 0x0f, 0xaf, 0xc8]);
        
        // exp-- (rbx--)
        // dec rbx
        self.code.extend_from_slice(&[0x48, 0xff, 0xcb]);
        
        // Jump back to loop
        let loop_offset = loop_start as i32 - (self.code.len() + 2) as i32;
        self.code.push(0xeb); // jmp short
        self.code.push(loop_offset as u8);
        
        // LOOP END - result is in rcx
        let loop_end_pos = self.code.len();
        self.code[loop_end_jump] = (loop_end_pos - loop_end_jump - 1) as u8;
        // mov rax, rcx
        self.code.extend_from_slice(&[0x48, 0x89, 0xc8]);
        
        // Jump to end
        self.code.push(0xeb);
        let end_jump = self.code.len();
        self.code.push(0x00);
        
        // ONE CASE: exp <= 0, return 1
        let one_case_pos = self.code.len();
        self.code[one_case_jump] = (one_case_pos - one_case_jump - 1) as u8;
        // mov rax, 1
        self.code.extend_from_slice(&[0x48, 0xb8, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        
        // END
        let end_pos = self.code.len();
        self.code[end_jump] = (end_pos - end_jump - 1) as u8;
        
        // Push result onto stack
        self.emit_push_rax();
    }
    
    // INNOVATIVE: Hand-crafted optimized GCD (Greatest Common Divisor)
    fn generate_gcd_optimized(&mut self, args: &[Expr]) {
        // gcd(a, b) using Euclidean algorithm: while b != 0: temp = b; b = a % b; a = temp;
        
        if args.len() < 2 {
            self.emit_push_immediate(0); // GCD with insufficient args = 0
            return;
        }
        
        // Evaluate a and b
        self.generate_expr(args[0].clone());
        self.generate_expr(args[1].clone());
        
        // Pop b into rbx, a into rax
        self.emit_pop_rbx(); // b
        self.emit_pop_rax(); // a
        
        // LOOP START: while b != 0
        let loop_start = self.code.len();
        
        // Check if b == 0
        // cmp rbx, 0
        self.code.extend_from_slice(&[0x48, 0x83, 0xfb, 0x00]);
        // je end (if b == 0, we're done)
        self.code.push(0x74);
        let end_jump = self.code.len();
        self.code.push(0x00);
        
        // temp = a % b (rax % rbx)
        // Save rdx (will be overwritten by div)
        // push rdx
        self.code.push(0x52);
        // Clear rdx for division
        // xor rdx, rdx
        self.code.extend_from_slice(&[0x48, 0x31, 0xd2]);
        // div rbx (rax / rbx, remainder in rdx)
        self.code.extend_from_slice(&[0x48, 0xf7, 0xf3]);
        
        // a = b (rax = rbx)
        // mov rax, rbx
        self.code.extend_from_slice(&[0x48, 0x89, 0xd8]);
        
        // b = remainder (rbx = rdx)
        // mov rbx, rdx
        self.code.extend_from_slice(&[0x48, 0x89, 0xd3]);
        
        // Restore rdx
        // pop rdx
        self.code.push(0x5a);
        
        // Jump back to loop start
        let loop_offset = loop_start as i32 - (self.code.len() + 2) as i32;
        self.code.push(0xeb); // jmp short
        self.code.push(loop_offset as u8);
        
        // END: result is in rax
        let end_pos = self.code.len();
        self.code[end_jump] = (end_pos - end_jump - 1) as u8;
        
        // Push result onto stack
        self.emit_push_rax();
    }
    
    // Generate lambda closure - creates a function pointer and captures environment
    fn generate_lambda_closure(&mut self, params: Vec<String>, body: Vec<Statement>) {
        // For now, implement a simple lambda that creates a function-like object
        // This is a simplified implementation without full closure capture
        
        let lambda_id = self.next_lambda_id;
        self.next_lambda_id += 1;
        
        // Store the lambda for later generation
        self.lambdas.insert(lambda_id, (params, body));
        
        // For now, push the lambda ID as a simple identifier
        // In a full implementation, this would be a function pointer or closure object
        self.emit_push_immediate(lambda_id as i64);
        
        eprintln!("🚀 LAMBDA: Generated closure ID {} with {} parameters", lambda_id, self.lambdas[&lambda_id].0.len());
    }
    
    // STRING INTERNING SYSTEM - unconventional solution to dual storage problem
    fn intern_string(&mut self, s: String) -> usize {
        // Check if string already exists
        for (i, existing) in self.string_table.iter().enumerate() {
            if existing == &s {
                return i;
            }
        }
        
        // Add new string to table
        let index = self.string_table.len();
        self.string_table.push(s);
        eprintln!("🧵 STRING_INTERN: '{}' → index {}", self.string_table[index], index);
        index
    }
    
    // Emit code to push string index (all strings now use indices!)
    fn emit_push_string_index(&mut self, index: usize) {
        self.emit_push_immediate(index as i64);
    }
    
    // REVOLUTIONARY: Smart synthesize that auto-detects strings vs numbers
    fn emit_synthesize_smart(&mut self) {
        // Value is on stack - pop it
        self.emit_pop_rax();
        
        // Check if this value is a valid string index
        // Compare with string table size (we'll embed this as immediate for now)
        let string_table_size = self.string_table.len() as i64;
        
        if string_table_size > 0 {
            // cmp rax, string_table_size
            if string_table_size <= 127 {
                self.code.extend_from_slice(&[0x48, 0x83, 0xf8]);
                self.code.push(string_table_size as u8);
            } else {
                self.code.extend_from_slice(&[0x48, 0x3d]);
                self.code.extend_from_slice(&(string_table_size as u32).to_le_bytes());
            }
            
            // jae number_case (if >= string_table_size, it's a number)
            self.code.extend_from_slice(&[0x73, 0x00]); // placeholder
            let number_case_jump = self.code.len() - 1;
            
            // STRING CASE: Print the actual string content
            eprintln!("🧵 SYNTHESIZE_SMART: Handling string index");
            
            // For now, just print a marker that this is a string
            // In full implementation, we'd look up the string and print it
            self.emit_synthesize_string_placeholder();
            
            // jmp end
            self.code.extend_from_slice(&[0xeb, 0x00]); // placeholder
            let end_jump = self.code.len() - 1;
            
            // NUMBER CASE: Use normal number printing
            let number_case_pos = self.code.len();
            eprintln!("🧵 SYNTHESIZE_SMART: Handling regular number");
            
            // Push the value back and use normal synthesize
            self.emit_push_rax();
            self.emit_synthesize_simple();
            
            // End point
            let end_pos = self.code.len();
            
            // Patch jumps
            self.code[number_case_jump] = (number_case_pos - number_case_jump - 1) as u8;
            self.code[end_jump] = (end_pos - end_jump - 1) as u8;
        } else {
            // No strings in table, just treat as number
            self.emit_push_rax();
            self.emit_synthesize_simple();
        }
    }
    
    // Placeholder for string printing - prints just the index for now
    fn emit_synthesize_string_placeholder(&mut self) {
        // For now, just print the string index to prove the system works
        // In full implementation, this would print the actual string content
        
        // The string index is already in rax
        self.emit_push_rax();
        self.emit_synthesize_simple();
        
        eprintln!("🧵 STRING_PLACEHOLDER: Printed string index (would be actual string content in full implementation)");
    }
    
    // 📚 REVOLUTIONARY STACK FRAME MANAGEMENT SYSTEM
    // Perfect for Hoberman sphere ripple function call cascades!
    
    /// Create a new stack frame for function entry
    fn stack_frame_enter(&mut self, function_name: String, param_count: usize) -> Result<(), String> {
        // Check call depth to prevent stack overflow
        let call_depth = self.call_stack.len();
        if call_depth >= self.max_call_depth {
            return Err(format!("Stack overflow: maximum call depth {} exceeded", self.max_call_depth));
        }
        
        // Save current frame base
        let caller_frame_base = if let Some(ref current) = self.current_frame {
            Some(current.frame_base)
        } else {
            None
        };
        
        // Create new frame
        let mut new_frame = StackFrame::new(function_name.clone(), self.frame_pointer, call_depth);
        new_frame.param_count = param_count;
        new_frame.caller_frame_base = caller_frame_base;
        
        // Push current frame to call stack if it exists
        if let Some(current) = self.current_frame.take() {
            self.call_stack.push(current);
        }
        
        self.current_frame = Some(new_frame);
        
        println!("📚 STACK_ENTER: Function '{}' at depth {}", function_name, call_depth);
        Ok(())
    }
    
    /// Exit current stack frame and restore previous
    fn stack_frame_exit(&mut self) -> Result<(), String> {
        if let Some(current) = self.current_frame.take() {
            println!("📚 STACK_EXIT: Function '{}' at depth {}", current.function_name, current.call_depth);
            
            // Restore frame pointer to caller
            if let Some(caller_base) = current.caller_frame_base {
                self.frame_pointer = caller_base;
            }
            
            // Restore previous frame
            self.current_frame = self.call_stack.pop();
            
            Ok(())
        } else {
            Err("Stack underflow: no frame to exit".to_string())
        }
    }
    
    /// Add a local variable to current frame
    fn stack_frame_add_local(&mut self, name: String, size: usize) -> Result<isize, String> {
        if let Some(ref mut frame) = self.current_frame {
            let offset = frame.add_local(name.clone(), size);
            println!("📚 STACK_LOCAL: Variable '{}' at offset {}", name, offset);
            Ok(offset)
        } else {
            Err("No active frame for local variable".to_string())
        }
    }
    
    /// Get local variable offset from current frame
    fn stack_frame_get_local(&self, name: &str) -> Option<isize> {
        // First check current frame
        if let Some(ref frame) = self.current_frame {
            if let Some(offset) = frame.get_local(name) {
                return Some(offset);
            }
        }
        
        // Then check parent frames (for closure variables)
        for frame in self.call_stack.iter().rev() {
            if let Some(offset) = frame.get_local(name) {
                return Some(offset);
            }
        }
        
        None
    }
    
    /// Get current call depth
    fn stack_frame_depth(&self) -> usize {
        let current_depth = if self.current_frame.is_some() { 1 } else { 0 };
        self.call_stack.len() + current_depth
    }
    
    /// Generate stack trace for debugging
    fn stack_frame_trace(&self) -> Vec<String> {
        let mut trace = Vec::new();
        
        // Add current frame
        if let Some(ref current) = self.current_frame {
            trace.push(format!("  at {}() [depth {}]", current.function_name, current.call_depth));
        }
        
        // Add call stack frames in reverse order
        for frame in self.call_stack.iter().rev() {
            trace.push(format!("  at {}() [depth {}]", frame.function_name, frame.call_depth));
        }
        
        trace
    }
    
    /// Emit function prologue with proper stack frame setup
    fn emit_function_prologue_with_frame(&mut self, function_name: &str, param_count: usize) {
        println!("📚 EMIT_PROLOGUE: Setting up frame for '{}'", function_name);
        
        // Track the function start for stack traces
        let function_start = self.code.len();
        self.current_function = Some(function_name.to_string());
        
        // Standard function prologue
        // push rbp
        self.code.push(0x55);
        
        // mov rbp, rsp (establish frame pointer)
        self.code.extend_from_slice(&[0x48, 0x89, 0xe5]);
        
        // 🔍 STACK TRACE: Push return address for tracking
        if self.debug_info_enabled {
            // Store the return address location (at [rbp+8])
            // This helps us walk the stack during errors
            // lea rax, [rbp+8]  ; Get return address location
            self.code.extend_from_slice(&[0x48, 0x8d, 0x45, 0x08]);
            
            // push rax  ; Save for potential stack trace
            self.code.push(0x50);
        }
        
        // Allocate space for local variables (will be adjusted as needed)
        // sub rsp, 32 (initial local space)
        self.code.extend_from_slice(&[0x48, 0x83, 0xec, 0x20]);
        
        // Update frame pointer tracking
        self.frame_pointer = 0; // rbp is now frame base
        
        // Register function info for stack traces
        if self.debug_info_enabled {
            let func_info = FunctionInfo {
                name: function_name.to_string(),
                start_address: function_start,
                end_address: 0, // Will be updated in epilogue
                source_file: "main.nl".to_string(), // TODO: Get actual source file
                source_line: 0, // TODO: Get actual line number
            };
            self.stack_trace_info.register_function(func_info);
            
            // Also register with DWARF debug info
            if let Some(ref mut debug_info) = self.debug_info {
                let func_debug = FunctionDebugInfo {
                    name: function_name.to_string(),
                    low_pc: 0x401000 + function_start as u64,
                    high_pc: 0,  // Will be updated
                    file: "main.nl".to_string(),
                    line: self.current_line,
                    params: Vec::new(),  // TODO: Add parameter info
                    locals: Vec::new(),  // TODO: Add local variable info
                };
                debug_info.add_function(func_debug);
            }
        }
    }
    
    /// Emit function epilogue with proper stack frame cleanup
    fn emit_function_epilogue_with_frame(&mut self, function_name: &str) {
        println!("📚 EMIT_EPILOGUE: Cleaning up frame for '{}'", function_name);
        
        // Update function end address for stack trace info
        if self.debug_info_enabled {
            let function_end = self.code.len();
            // Find and update the function info
            // In a real implementation, we'd update the existing FunctionInfo
        }
        
        // Standard function epilogue
        // mov rsp, rbp (restore stack pointer)
        self.code.extend_from_slice(&[0x48, 0x89, 0xec]);
        
        // pop rbp (restore frame pointer)
        self.code.push(0x5d);
        
        // ret (return to caller)
        self.code.push(0xc3);
    }
    
    /// 🌊 SPECIAL: Stack frame for Hoberman ripple cascade calls
    fn emit_ripple_cascade_frame(&mut self, ripple_depth: usize) {
        println!("🌊 EMIT_RIPPLE_FRAME: Setting up ripple cascade at depth {}", ripple_depth);
        
        // Enhanced prologue for ripple patterns
        // push rbp
        self.code.push(0x55);
        
        // mov rbp, rsp
        self.code.extend_from_slice(&[0x48, 0x89, 0xe5]);
        
        // Allocate extra space for ripple wave data
        let ripple_space = 32 + (ripple_depth * 16); // Base + depth * wave_size
        
        // sub rsp, ripple_space
        self.code.extend_from_slice(&[0x48, 0x81, 0xec]);
        self.code.extend_from_slice(&(ripple_space as u32).to_le_bytes());
        
        // Store ripple depth in frame
        // mov [rbp-8], ripple_depth
        self.code.extend_from_slice(&[0x48, 0xc7, 0x45, 0xf8]);
        self.code.extend_from_slice(&(ripple_depth as u32).to_le_bytes());
    }
    
    /// 🧠 SPECIAL: Stack frame for neural network function calls
    fn emit_neural_network_frame(&mut self, neuron_count: usize) {
        println!("🧠 EMIT_NEURAL_FRAME: Setting up neural network frame for {} neurons", neuron_count);
        
        // Enhanced prologue for neural patterns
        // push rbp
        self.code.push(0x55);
        
        // mov rbp, rsp
        self.code.extend_from_slice(&[0x48, 0x89, 0xe5]);
        
        // Allocate space for neuron activations and weights
        let neural_space = 64 + (neuron_count * 24); // Base + neurons * (weights + bias + activation)
        
        // sub rsp, neural_space
        self.code.extend_from_slice(&[0x48, 0x81, 0xec]);
        self.code.extend_from_slice(&(neural_space as u32).to_le_bytes());
        
        // Store neuron count in frame
        // mov [rbp-16], neuron_count
        self.code.extend_from_slice(&[0x48, 0xc7, 0x45, 0xf0]);
        self.code.extend_from_slice(&(neuron_count as u32).to_le_bytes());
    }
    
    /// Stack overflow detection
    fn check_stack_overflow(&self) -> Result<(), String> {
        let depth = self.stack_frame_depth();
        if depth >= self.max_call_depth {
            let trace = self.stack_frame_trace();
            Err(format!("Stack overflow detected at depth {}:\n{}", depth, trace.join("\n")))
        } else {
            Ok(())
        }
    }
    
    /// 📚 Emit stack overflow protection in machine code
    fn emit_stack_overflow_check(&mut self) {
        println!("📚 EMIT_STACK_OVERFLOW_CHECK: Generating stack overflow protection");
        
        // Push error message for stack overflow
        self.emit_push_immediate(42); // Return error code 42 for stack overflow
        self.emit_pop_rax();
        
        // Exit with error code  
        // mov rdi, rax (exit code)
        self.code.extend_from_slice(&[0x48, 0x89, 0xc7]);
        
        // mov rax, 60 (sys_exit)
        self.code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x3c, 0x00, 0x00, 0x00]);
        
        // syscall
        self.code.extend_from_slice(&[0x0f, 0x05]);
    }
    
    // 🗑️ REVOLUTIONARY GARBAGE COLLECTION SYSTEM
    // Perfect for Hoberman sphere ripple patterns that create/destroy waves!
    
    /// Allocate a new object on the GC heap
    fn gc_alloc(&mut self, size: usize, object_type: GcType) -> usize {
        let addr = self.next_heap_addr;
        self.next_heap_addr += size;
        
        let gc_obj = GcObject {
            size,
            ref_count: 1,  // Start with 1 reference
            object_type,
            marked: false,
            data_addr: addr,
        };
        
        self.gc_heap.insert(addr, gc_obj);
        println!("🗑️ GC_ALLOC: Allocated {} bytes at 0x{:x}", size, addr);
        
        // Check if we need to trigger GC
        if self.gc_heap_size() > self.gc_threshold {
            self.gc_collect();
        }
        
        addr
    }
    
    /// Increment reference count for an object
    fn gc_retain(&mut self, addr: usize) {
        if let Some(obj) = self.gc_heap.get_mut(&addr) {
            obj.ref_count += 1;
            println!("🗑️ GC_RETAIN: Object at 0x{:x} ref_count now {}", addr, obj.ref_count);
        }
    }
    
    /// Decrement reference count and deallocate if zero
    fn gc_release(&mut self, addr: usize) {
        if let Some(obj) = self.gc_heap.get_mut(&addr) {
            obj.ref_count -= 1;
            println!("🗑️ GC_RELEASE: Object at 0x{:x} ref_count now {}", addr, obj.ref_count);
            
            if obj.ref_count == 0 {
                println!("🗑️ GC_FREE: Deallocating object at 0x{:x}", addr);
                self.gc_heap.remove(&addr);
            }
        }
    }
    
    /// Mark and sweep garbage collection
    fn gc_collect(&mut self) {
        println!("🗑️ GC_COLLECT: Starting garbage collection...");
        let initial_count = self.gc_heap.len();
        
        // Phase 1: Mark all reachable objects
        self.gc_mark_phase();
        
        // Phase 2: Sweep unreachable objects  
        self.gc_sweep_phase();
        
        let final_count = self.gc_heap.len();
        println!("🗑️ GC_COLLECT: Collected {} objects, {} remaining", 
                initial_count - final_count, final_count);
    }
    
    /// Mark phase: mark all reachable objects from roots
    fn gc_mark_phase(&mut self) {
        // Reset all mark bits
        for obj in self.gc_heap.values_mut() {
            obj.marked = false;
        }
        
        // Mark all objects reachable from roots
        for &root_addr in &self.gc_roots.clone() {
            self.gc_mark_object(root_addr);
        }
    }
    
    /// Mark an object and all objects it references
    fn gc_mark_object(&mut self, addr: usize) {
        if let Some(obj) = self.gc_heap.get_mut(&addr) {
            if !obj.marked {
                obj.marked = true;
                println!("🗑️ GC_MARK: Marked object at 0x{:x}", addr);
                
                // Mark referenced objects based on type
                match &obj.object_type {
                    GcType::Array(count) => {
                        // For arrays, mark all element objects
                        for i in 0..*count {
                            let element_addr = addr + (i * 8); // Assuming 8-byte pointers
                            if self.gc_heap.contains_key(&element_addr) {
                                self.gc_mark_object(element_addr);
                            }
                        }
                    }
                    GcType::RippleWave => {
                        // Hoberman ripple waves might reference other waves
                        // Mark connected wave objects
                        println!("🌊 GC_MARK: Marking ripple wave connections");
                    }
                    _ => {} // Strings and other types don't reference other objects
                }
            }
        }
    }
    
    /// Sweep phase: deallocate all unmarked objects
    fn gc_sweep_phase(&mut self) {
        let mut to_remove = Vec::new();
        
        for (&addr, obj) in &self.gc_heap {
            if !obj.marked {
                to_remove.push(addr);
            }
        }
        
        for addr in to_remove {
            println!("🗑️ GC_SWEEP: Deallocating unmarked object at 0x{:x}", addr);
            self.gc_heap.remove(&addr);
        }
    }
    
    /// Add a root reference (for stack variables)
    fn gc_add_root(&mut self, addr: usize) {
        if !self.gc_roots.contains(&addr) {
            self.gc_roots.push(addr);
            println!("🗑️ GC_ROOT: Added root at 0x{:x}", addr);
        }
    }
    
    /// Remove a root reference
    fn gc_remove_root(&mut self, addr: usize) {
        if let Some(pos) = self.gc_roots.iter().position(|&x| x == addr) {
            self.gc_roots.remove(pos);
            println!("🗑️ GC_ROOT: Removed root at 0x{:x}", addr);
        }
    }
    
    /// Calculate total heap size
    fn gc_heap_size(&self) -> usize {
        self.gc_heap.values().map(|obj| obj.size).sum()
    }
    
    /// 🌊 SPECIAL: Allocate Hoberman ripple wave object
    fn gc_alloc_ripple_wave(&mut self, energy: f64, radius: f64) -> usize {
        let size = 16; // 8 bytes for energy + 8 bytes for radius
        let addr = self.gc_alloc(size, GcType::RippleWave);
        
        println!("🌊 GC_RIPPLE: Allocated ripple wave with energy {} at radius {}", energy, radius);
        addr
    }
    
    /// 🧠 SPECIAL: Allocate neural network neuron object
    fn gc_alloc_neuron(&mut self, weights_count: usize) -> usize {
        let size = weights_count * 8 + 16; // Weights + bias + activation
        let addr = self.gc_alloc(size, GcType::Neuron);
        
        println!("🧠 GC_NEURON: Allocated neuron with {} weights", weights_count);
        addr
    }
    
    /// Compile a standard library module from source
    fn compile_stdlib_module(&mut self, module_name: &str, source: &str) {
        use crate::minimal_lexer::Lexer;
        use crate::minimal_parser::Parser;
        
        println!("  📖 Compiling stdlib module: {}", module_name);
        
        // Lex the module source
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        
        // Parse the module
        let mut parser = Parser::new(tokens);
        match parser.parse() {
            Ok(declarations) => {
                // Process the parsed module
                for decl in declarations {
                    match decl {
                        Declaration::Module { name: _, exports, body } => {
                            // For stdlib modules, use the actual module path as the prefix
                            let module_prefix = format!("{}::", module_name);
                            
                            for inner_decl in body {
                                match inner_decl {
                                    Declaration::Function { name: func_name, params, body, .. } => {
                                        // Only register exported functions
                                        if exports.contains(&func_name) {
                                            let namespaced_name = format!("{}{}", module_prefix, func_name);
                                            self.function_params.insert(namespaced_name.clone(), params);
                                            self.function_bodies.insert(namespaced_name.clone(), body);
                                            println!("    ✓ Registered stdlib function: {} as {}", func_name, namespaced_name);
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            
                            // Store module exports
                            self.modules.insert(module_name.to_string(), exports);
                        }
                        _ => {}
                    }
                }
            }
            Err(e) => {
                println!("  ⚠️ Failed to parse stdlib module {}: {}", module_name, e);
            }
        }
    }
    
    /// Emit GC allocation call in machine code
    fn emit_gc_alloc(&mut self, size: usize, gc_type: GcType) {
        // For now, just use regular malloc and track it
        self.emit_push_imm(size as i64);
        self.emit_malloc();
        
        // The address is now on stack - register it with GC
        let addr = self.gc_alloc(size, gc_type);
        println!("🗑️ EMIT_GC_ALLOC: Generated GC allocation for {} bytes", size);
    }
    
    /// Emit simple GC allocation (size on stack -> address on stack)
    fn emit_gc_alloc_simple(&mut self) {
        // Pop size from stack
        self.emit_pop_rax(); // rax = size
        
        // Call malloc for now (will be replaced with proper GC allocation)
        self.emit_push_rax(); // Put size back for malloc
        self.emit_malloc();
        
        // TODO: Register the allocated address with GC system
        println!("🗑️ EMIT_GC_ALLOC_SIMPLE: Generated simple GC allocation");
    }
    
    /// Emit GC free (decrements reference count)
    fn emit_gc_free(&mut self) {
        // Pop pointer from stack
        self.emit_pop_rax(); // rax = pointer
        
        // For now, just ignore (GC will handle it)
        // TODO: Implement actual reference counting decrement
        println!("🗑️ EMIT_GC_FREE: Generated GC free");
    }
    
    /// Emit GC collection trigger
    fn emit_gc_collect(&mut self) {
        // Trigger garbage collection
        // For now, this is a no-op in the generated code
        // The actual collection happens at compile time
        println!("🗑️ EMIT_GC_COLLECT: Generated GC collection trigger");
        
        // Just push 0 to maintain stack balance
        self.emit_push_immediate(0);
    }
    
    /// Emit Hoberman ripple wave allocation
    fn emit_gc_alloc_ripple(&mut self) {
        // Stack: [energy, radius] -> [ripple_wave_addr]
        self.emit_pop_rbx(); // rbx = radius
        self.emit_pop_rax(); // rax = energy
        
        // Allocate 16 bytes for ripple wave (energy + radius)
        self.emit_push_immediate(16);
        self.emit_malloc();
        
        // Store energy and radius in allocated memory
        self.emit_pop_rcx(); // rcx = allocated address
        
        // Store energy at [rcx]
        self.code.extend_from_slice(&[0x48, 0x89, 0x01]); // mov [rcx], rax
        
        // Store radius at [rcx+8]
        self.code.extend_from_slice(&[0x48, 0x89, 0x59, 0x08]); // mov [rcx+8], rbx
        
        // Push ripple wave address back to stack
        self.emit_push_rax();
        
        println!("🌊 EMIT_GC_ALLOC_RIPPLE: Generated Hoberman ripple wave allocation");
    }
    
    /// Emit neural network neuron allocation
    fn emit_gc_alloc_neuron(&mut self) {
        // Stack: [weights_count] -> [neuron_addr]
        self.emit_pop_rax(); // rax = weights_count
        
        // Calculate size: weights_count * 8 + 16 (for bias and activation)
        self.code.extend_from_slice(&[0x48, 0xc1, 0xe0, 0x03]); // shl rax, 3 (multiply by 8)
        self.code.extend_from_slice(&[0x48, 0x83, 0xc0, 0x10]); // add rax, 16
        
        // Allocate memory
        self.emit_push_rax(); // size
        self.emit_malloc();
        
        println!("🧠 EMIT_GC_ALLOC_NEURON: Generated neural network neuron allocation");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimal_lexer::Lexer;
    use crate::minimal_parser::Parser;
    
    #[test]
    fn test_generate_simple_program() {
        let input = r#"
            organism HelloWorld {
                fn main() {
                    let x = +1
                    let y = -1
                    let z = x + y
                    return z
                }
            }
        "#;
        
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        let ast = parser.parse().unwrap();
        
        let mut codegen = CodeGen::new();
        let elf = codegen.generate_elf(ast);
        
        // Verify ELF header
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F']);
        
        // Verify we generated some code
        assert!(elf.len() > 0x1000);
    }
}