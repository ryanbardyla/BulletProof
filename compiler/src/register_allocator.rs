// 🎯 REGISTER ALLOCATION SYSTEM
// Optimizes register usage to minimize memory accesses

use std::collections::{HashMap, HashSet, VecDeque};

// x86_64 general-purpose registers available for allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Register {
    // Caller-saved registers (can be freely used)
    RAX,    // Return value, also used for arithmetic
    RCX,    // 4th argument
    RDX,    // 3rd argument, also used for division
    RSI,    // 2nd argument
    RDI,    // 1st argument
    R8,     // 5th argument
    R9,     // 6th argument
    R10,    // Temporary
    R11,    // Temporary
    
    // Callee-saved registers (must be preserved across calls)
    RBX,    // Base register
    R12,    // Temporary
    R13,    // Temporary
    R14,    // Temporary
    R15,    // Temporary
    
    // Special purpose (not allocated)
    RSP,    // Stack pointer (never allocated)
    RBP,    // Base pointer (used for frame)
}

impl Register {
    // Get the x86_64 encoding for the register
    pub fn encoding(&self) -> u8 {
        match self {
            Register::RAX => 0,
            Register::RCX => 1,
            Register::RDX => 2,
            Register::RBX => 3,
            Register::RSP => 4,
            Register::RBP => 5,
            Register::RSI => 6,
            Register::RDI => 7,
            Register::R8  => 8,
            Register::R9  => 9,
            Register::R10 => 10,
            Register::R11 => 11,
            Register::R12 => 12,
            Register::R13 => 13,
            Register::R14 => 14,
            Register::R15 => 15,
        }
    }
    
    // Check if register needs to be saved across function calls
    pub fn is_callee_saved(&self) -> bool {
        matches!(self, 
            Register::RBX | Register::R12 | Register::R13 | 
            Register::R14 | Register::R15
        )
    }
    
    // Get assembly name for the register
    pub fn name(&self) -> &'static str {
        match self {
            Register::RAX => "rax",
            Register::RCX => "rcx",
            Register::RDX => "rdx",
            Register::RBX => "rbx",
            Register::RSP => "rsp",
            Register::RBP => "rbp",
            Register::RSI => "rsi",
            Register::RDI => "rdi",
            Register::R8  => "r8",
            Register::R9  => "r9",
            Register::R10 => "r10",
            Register::R11 => "r11",
            Register::R12 => "r12",
            Register::R13 => "r13",
            Register::R14 => "r14",
            Register::R15 => "r15",
        }
    }
}

// Variable lifetime information
#[derive(Debug, Clone)]
pub struct LiveRange {
    pub var_name: String,
    pub start: usize,      // First use instruction index
    pub end: usize,        // Last use instruction index
    pub uses: Vec<usize>,  // All instruction indices where variable is used
    pub is_spilled: bool,  // Whether this variable is spilled to memory
}

// Register allocation result for a variable
#[derive(Debug, Clone)]
pub enum Allocation {
    Register(Register),         // Allocated to a register
    Spilled(isize),            // Spilled to stack at offset
    Constant(i64),             // Constant value (no allocation needed)
}

// Graph coloring register allocator
pub struct RegisterAllocator {
    // Available registers for allocation
    available_registers: Vec<Register>,
    
    // Live ranges for all variables
    live_ranges: HashMap<String, LiveRange>,
    
    // Interference graph: variables that are live at the same time
    interference_graph: HashMap<String, HashSet<String>>,
    
    // Final allocations
    allocations: HashMap<String, Allocation>,
    
    // Stack space needed for spilled variables
    spill_stack_size: usize,
    
    // Registers currently in use
    used_registers: HashSet<Register>,
}

impl RegisterAllocator {
    pub fn new() -> Self {
        // Initialize with all allocatable registers
        let available_registers = vec![
            // Caller-saved (preferred for short-lived variables)
            Register::RAX,
            Register::RCX,
            Register::RDX,
            Register::RSI,
            Register::RDI,
            Register::R8,
            Register::R9,
            Register::R10,
            Register::R11,
            
            // Callee-saved (preferred for long-lived variables)
            Register::RBX,
            Register::R12,
            Register::R13,
            Register::R14,
            Register::R15,
        ];
        
        RegisterAllocator {
            available_registers,
            live_ranges: HashMap::new(),
            interference_graph: HashMap::new(),
            allocations: HashMap::new(),
            spill_stack_size: 0,
            used_registers: HashSet::new(),
        }
    }
    
    // Add a variable's live range
    pub fn add_live_range(&mut self, var_name: String, start: usize, end: usize, uses: Vec<usize>) {
        self.live_ranges.insert(var_name.clone(), LiveRange {
            var_name,
            start,
            end,
            uses,
            is_spilled: false,
        });
    }
    
    // Build the interference graph
    pub fn build_interference_graph(&mut self) {
        let vars: Vec<String> = self.live_ranges.keys().cloned().collect();
        
        // Check each pair of variables
        for i in 0..vars.len() {
            for j in i+1..vars.len() {
                let var1 = &vars[i];
                let var2 = &vars[j];
                
                let range1 = &self.live_ranges[var1];
                let range2 = &self.live_ranges[var2];
                
                // Variables interfere if their live ranges overlap
                if ranges_overlap(range1, range2) {
                    self.interference_graph.entry(var1.clone())
                        .or_insert_with(HashSet::new)
                        .insert(var2.clone());
                    self.interference_graph.entry(var2.clone())
                        .or_insert_with(HashSet::new)
                        .insert(var1.clone());
                }
            }
        }
    }
    
    // Perform register allocation using graph coloring
    pub fn allocate(&mut self) -> Result<(), String> {
        // Build interference graph
        self.build_interference_graph();
        
        // Sort variables by spill cost (prefer to spill less-used variables)
        let mut vars: Vec<String> = self.live_ranges.keys().cloned().collect();
        vars.sort_by_key(|v| {
            let range = &self.live_ranges[v];
            // Spill cost: prefer to spill variables with fewer uses and shorter ranges
            (range.uses.len() * (range.end - range.start))
        });
        
        // Try to allocate each variable
        for var in vars {
            if !self.try_allocate_register(&var) {
                // Spill to memory if no register available
                self.spill_variable(&var);
            }
        }
        
        Ok(())
    }
    
    // Try to allocate a register for a variable
    fn try_allocate_register(&mut self, var: &str) -> bool {
        // Get neighbors in interference graph
        let neighbors = self.interference_graph.get(var)
            .map(|n| n.clone())
            .unwrap_or_else(HashSet::new);
        
        // Find registers used by neighbors
        let mut used_by_neighbors = HashSet::new();
        for neighbor in &neighbors {
            if let Some(Allocation::Register(reg)) = self.allocations.get(neighbor) {
                used_by_neighbors.insert(*reg);
            }
        }
        
        // Try to find an available register
        for reg in &self.available_registers {
            if !used_by_neighbors.contains(reg) {
                self.allocations.insert(var.to_string(), Allocation::Register(*reg));
                self.used_registers.insert(*reg);
                return true;
            }
        }
        
        false
    }
    
    // Spill a variable to memory
    fn spill_variable(&mut self, var: &str) {
        // Allocate stack space for spilled variable
        self.spill_stack_size += 8; // 64-bit values
        let offset = -(self.spill_stack_size as isize);
        
        self.allocations.insert(var.to_string(), Allocation::Spilled(offset));
        
        if let Some(range) = self.live_ranges.get_mut(var) {
            range.is_spilled = true;
        }
    }
    
    // Get allocation for a variable
    pub fn get_allocation(&self, var: &str) -> Option<&Allocation> {
        self.allocations.get(var)
    }
    
    // Get all callee-saved registers that were used
    pub fn get_used_callee_saved_registers(&self) -> Vec<Register> {
        self.used_registers.iter()
            .filter(|r| r.is_callee_saved())
            .copied()
            .collect()
    }
    
    // Get total stack space needed for spills
    pub fn get_spill_stack_size(&self) -> usize {
        self.spill_stack_size
    }
    
    // Linear scan allocation (alternative, simpler algorithm)
    pub fn linear_scan_allocate(&mut self) -> Result<(), String> {
        // Sort intervals by start point
        let mut intervals: Vec<(String, LiveRange)> = 
            self.live_ranges.iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
        intervals.sort_by_key(|(_, range)| range.start);
        
        // Track active intervals and their allocations
        let mut active: Vec<(String, LiveRange, Register)> = Vec::new();
        let mut free_registers: VecDeque<Register> = self.available_registers.iter().copied().collect();
        
        for (var, range) in intervals {
            // Expire old intervals
            active.retain(|(active_var, active_range, reg)| {
                if active_range.end < range.start {
                    // Return register to free pool
                    free_registers.push_back(*reg);
                    false
                } else {
                    true
                }
            });
            
            // Try to allocate register
            if let Some(reg) = free_registers.pop_front() {
                self.allocations.insert(var.clone(), Allocation::Register(reg));
                self.used_registers.insert(reg);
                active.push((var, range, reg));
            } else {
                // Need to spill - find variable with furthest end point
                if let Some(spill_idx) = active.iter()
                    .enumerate()
                    .max_by_key(|(_, (_, r, _))| r.end)
                    .map(|(i, _)| i) 
                {
                    if active[spill_idx].1.end > range.end {
                        // Spill the active variable and give its register to current
                        let (spilled_var, _, reg) = active.remove(spill_idx);
                        self.spill_variable(&spilled_var);
                        self.allocations.insert(var.clone(), Allocation::Register(reg));
                        active.push((var, range, reg));
                    } else {
                        // Spill current variable
                        self.spill_variable(&var);
                    }
                } else {
                    self.spill_variable(&var);
                }
            }
        }
        
        Ok(())
    }
    
    // Generate code to save callee-saved registers
    pub fn emit_save_registers(&self) -> Vec<u8> {
        let mut code = Vec::new();
        
        for reg in self.get_used_callee_saved_registers() {
            // push reg
            code.push(0x50 + reg.encoding());
        }
        
        code
    }
    
    // Generate code to restore callee-saved registers
    pub fn emit_restore_registers(&self) -> Vec<u8> {
        let mut code = Vec::new();
        
        // Restore in reverse order
        for reg in self.get_used_callee_saved_registers().iter().rev() {
            // pop reg
            code.push(0x58 + reg.encoding());
        }
        
        code
    }
    
    // Generate move instruction from source to destination
    pub fn emit_move(&self, dst: &Allocation, src: &Allocation) -> Vec<u8> {
        let mut code = Vec::new();
        
        match (dst, src) {
            (Allocation::Register(dst_reg), Allocation::Register(src_reg)) => {
                // mov dst_reg, src_reg
                if dst_reg.encoding() >= 8 || src_reg.encoding() >= 8 {
                    // Need REX prefix for R8-R15
                    code.push(0x49); // REX.WB
                }
                code.push(0x89); // MOV r/m64, r64
                code.push(0xC0 + (src_reg.encoding() << 3) + dst_reg.encoding());
            }
            (Allocation::Register(reg), Allocation::Spilled(offset)) => {
                // mov reg, [rbp + offset]
                if reg.encoding() >= 8 {
                    code.push(0x4C); // REX.WR
                } else {
                    code.push(0x48); // REX.W
                }
                code.push(0x8B); // MOV r64, r/m64
                code.push(0x85 + (reg.encoding() << 3)); // ModRM
                code.extend_from_slice(&(*offset as i32).to_le_bytes());
            }
            (Allocation::Spilled(offset), Allocation::Register(reg)) => {
                // mov [rbp + offset], reg
                if reg.encoding() >= 8 {
                    code.push(0x4C); // REX.WR
                } else {
                    code.push(0x48); // REX.W
                }
                code.push(0x89); // MOV r/m64, r64
                code.push(0x85 + (reg.encoding() << 3)); // ModRM
                code.extend_from_slice(&(*offset as i32).to_le_bytes());
            }
            (Allocation::Register(reg), Allocation::Constant(val)) => {
                // mov reg, imm64
                code.push(0x48 + (if reg.encoding() >= 8 { 1 } else { 0 })); // REX.W
                code.push(0xB8 + (reg.encoding() & 7)); // MOV r64, imm64
                code.extend_from_slice(&val.to_le_bytes());
            }
            _ => {
                // Other combinations may need temporary register
            }
        }
        
        code
    }
}

// Check if two live ranges overlap
fn ranges_overlap(range1: &LiveRange, range2: &LiveRange) -> bool {
    !(range1.end < range2.start || range2.end < range1.start)
}

// Liveness analysis to determine variable live ranges
pub struct LivenessAnalyzer {
    // Map from instruction index to variables used
    uses: HashMap<usize, HashSet<String>>,
    
    // Map from instruction index to variables defined
    defs: HashMap<usize, HashSet<String>>,
    
    // Live variables at each instruction
    live_in: HashMap<usize, HashSet<String>>,
    live_out: HashMap<usize, HashSet<String>>,
    
    // Control flow graph
    predecessors: HashMap<usize, Vec<usize>>,
    successors: HashMap<usize, Vec<usize>>,
}

impl LivenessAnalyzer {
    pub fn new() -> Self {
        LivenessAnalyzer {
            uses: HashMap::new(),
            defs: HashMap::new(),
            live_in: HashMap::new(),
            live_out: HashMap::new(),
            predecessors: HashMap::new(),
            successors: HashMap::new(),
        }
    }
    
    // Add use of a variable at instruction
    pub fn add_use(&mut self, inst: usize, var: String) {
        self.uses.entry(inst).or_insert_with(HashSet::new).insert(var);
    }
    
    // Add definition of a variable at instruction
    pub fn add_def(&mut self, inst: usize, var: String) {
        self.defs.entry(inst).or_insert_with(HashSet::new).insert(var);
    }
    
    // Add control flow edge
    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.successors.entry(from).or_insert_with(Vec::new).push(to);
        self.predecessors.entry(to).or_insert_with(Vec::new).push(from);
    }
    
    // Perform liveness analysis using dataflow equations
    pub fn analyze(&mut self, num_instructions: usize) {
        // Initialize live sets
        for i in 0..num_instructions {
            self.live_in.insert(i, HashSet::new());
            self.live_out.insert(i, HashSet::new());
        }
        
        // Fixed-point iteration
        let mut changed = true;
        while changed {
            changed = false;
            
            // Process instructions in reverse order
            for i in (0..num_instructions).rev() {
                // live_in[i] = use[i] ∪ (live_out[i] - def[i])
                let mut new_live_in = self.uses.get(&i)
                    .cloned()
                    .unwrap_or_else(HashSet::new);
                
                if let Some(live_out) = self.live_out.get(&i) {
                    let defs = self.defs.get(&i)
                        .cloned()
                        .unwrap_or_else(HashSet::new);
                    
                    for var in live_out {
                        if !defs.contains(var) {
                            new_live_in.insert(var.clone());
                        }
                    }
                }
                
                if new_live_in != *self.live_in.get(&i).unwrap() {
                    self.live_in.insert(i, new_live_in);
                    changed = true;
                }
                
                // live_out[i] = ∪ live_in[succ] for all successors
                let mut new_live_out = HashSet::new();
                if let Some(succs) = self.successors.get(&i) {
                    for succ in succs {
                        if let Some(succ_live_in) = self.live_in.get(succ) {
                            new_live_out.extend(succ_live_in.clone());
                        }
                    }
                }
                
                if new_live_out != *self.live_out.get(&i).unwrap() {
                    self.live_out.insert(i, new_live_out);
                    changed = true;
                }
            }
        }
    }
    
    // Extract live ranges from liveness information
    pub fn extract_live_ranges(&self) -> HashMap<String, LiveRange> {
        let mut ranges = HashMap::new();
        let mut var_uses: HashMap<String, Vec<usize>> = HashMap::new();
        
        // Collect all uses and defs
        for (inst, vars) in &self.uses {
            for var in vars {
                var_uses.entry(var.clone()).or_insert_with(Vec::new).push(*inst);
            }
        }
        
        for (inst, vars) in &self.defs {
            for var in vars {
                var_uses.entry(var.clone()).or_insert_with(Vec::new).push(*inst);
            }
        }
        
        // Create live ranges
        for (var, uses) in var_uses {
            if !uses.is_empty() {
                let start = *uses.iter().min().unwrap();
                let end = *uses.iter().max().unwrap();
                
                ranges.insert(var.clone(), LiveRange {
                    var_name: var,
                    start,
                    end,
                    uses,
                    is_spilled: false,
                });
            }
        }
        
        ranges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_register_allocation_simple() {
        let mut allocator = RegisterAllocator::new();
        
        // Add some live ranges
        allocator.add_live_range("x".to_string(), 0, 5, vec![0, 2, 5]);
        allocator.add_live_range("y".to_string(), 3, 8, vec![3, 6, 8]);
        allocator.add_live_range("z".to_string(), 7, 10, vec![7, 9, 10]);
        
        // Perform allocation
        allocator.allocate().unwrap();
        
        // Check that variables got allocated
        assert!(allocator.get_allocation("x").is_some());
        assert!(allocator.get_allocation("y").is_some());
        assert!(allocator.get_allocation("z").is_some());
        
        // x and y overlap, so should get different registers
        if let (Some(Allocation::Register(rx)), Some(Allocation::Register(ry))) = 
            (allocator.get_allocation("x"), allocator.get_allocation("y")) {
            assert_ne!(rx, ry);
        }
    }
    
    #[test]
    fn test_liveness_analysis() {
        let mut analyzer = LivenessAnalyzer::new();
        
        // Example: x = 1; y = x + 2; z = y * 3;
        analyzer.add_def(0, "x".to_string());
        analyzer.add_use(1, "x".to_string());
        analyzer.add_def(1, "y".to_string());
        analyzer.add_use(2, "y".to_string());
        analyzer.add_def(2, "z".to_string());
        
        // Linear control flow
        analyzer.add_edge(0, 1);
        analyzer.add_edge(1, 2);
        
        // Analyze
        analyzer.analyze(3);
        
        // Extract ranges
        let ranges = analyzer.extract_live_ranges();
        
        assert_eq!(ranges["x"].start, 0);
        assert_eq!(ranges["x"].end, 1);
        assert_eq!(ranges["y"].start, 1);
        assert_eq!(ranges["y"].end, 2);
    }
}