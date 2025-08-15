// 🔴 BREAKPOINT SUPPORT
// Implements software breakpoints for debugging NeuronLang programs

use std::collections::HashMap;

/// Breakpoint types
#[derive(Debug, Clone, PartialEq)]
pub enum BreakpointType {
    /// Standard INT3 software breakpoint
    Software,
    /// Conditional breakpoint (evaluates expression)
    Conditional(String),
    /// Temporary breakpoint (removed after hit)
    Temporary,
    /// Watchpoint (triggers on memory access)
    Watchpoint { address: usize, size: usize },
}

/// Breakpoint information
#[derive(Debug, Clone)]
pub struct Breakpoint {
    pub id: u32,
    pub address: usize,
    pub original_byte: u8,
    pub breakpoint_type: BreakpointType,
    pub hit_count: u32,
    pub enabled: bool,
    pub source_location: Option<SourceLocation>,
}

#[derive(Debug, Clone)]
pub struct SourceLocation {
    pub file: String,
    pub line: u32,
    pub function: Option<String>,
}

/// Manages breakpoints in the compiled code
pub struct BreakpointManager {
    breakpoints: HashMap<usize, Breakpoint>,
    next_id: u32,
    source_to_address: HashMap<(String, u32), usize>, // (file, line) -> address
    address_to_source: HashMap<usize, SourceLocation>,
}

impl BreakpointManager {
    pub fn new() -> Self {
        BreakpointManager {
            breakpoints: HashMap::new(),
            next_id: 1,
            source_to_address: HashMap::new(),
            address_to_source: HashMap::new(),
        }
    }
    
    /// Register a source line to address mapping
    pub fn register_source_mapping(&mut self, file: String, line: u32, address: usize, function: Option<String>) {
        self.source_to_address.insert((file.clone(), line), address);
        self.address_to_source.insert(address, SourceLocation {
            file,
            line,
            function,
        });
    }
    
    /// Set a breakpoint at a specific address
    pub fn set_breakpoint_at_address(&mut self, address: usize, breakpoint_type: BreakpointType) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        
        let source_location = self.address_to_source.get(&address).cloned();
        
        let breakpoint = Breakpoint {
            id,
            address,
            original_byte: 0, // Will be filled when breakpoint is inserted
            breakpoint_type,
            hit_count: 0,
            enabled: true,
            source_location,
        };
        
        self.breakpoints.insert(address, breakpoint);
        id
    }
    
    /// Set a breakpoint at a source line
    pub fn set_breakpoint_at_line(&mut self, file: &str, line: u32) -> Option<u32> {
        if let Some(&address) = self.source_to_address.get(&(file.to_string(), line)) {
            Some(self.set_breakpoint_at_address(address, BreakpointType::Software))
        } else {
            None
        }
    }
    
    /// Set a breakpoint at a function entry
    pub fn set_breakpoint_at_function(&mut self, function_name: &str) -> Option<u32> {
        // Find the first address associated with this function
        for (addr, loc) in &self.address_to_source {
            if let Some(ref func) = loc.function {
                if func == function_name {
                    return Some(self.set_breakpoint_at_address(*addr, BreakpointType::Software));
                }
            }
        }
        None
    }
    
    /// Remove a breakpoint
    pub fn remove_breakpoint(&mut self, id: u32) -> bool {
        self.breakpoints.retain(|_, bp| bp.id != id);
        true
    }
    
    /// Enable/disable a breakpoint
    pub fn enable_breakpoint(&mut self, id: u32, enabled: bool) -> bool {
        for bp in self.breakpoints.values_mut() {
            if bp.id == id {
                bp.enabled = enabled;
                return true;
            }
        }
        false
    }
    
    /// Get all breakpoints
    pub fn get_breakpoints(&self) -> Vec<&Breakpoint> {
        self.breakpoints.values().collect()
    }
    
    /// Check if an address has a breakpoint
    pub fn has_breakpoint(&self, address: usize) -> bool {
        self.breakpoints.contains_key(&address)
    }
    
    /// Generate breakpoint insertion code
    pub fn generate_int3_instruction() -> Vec<u8> {
        vec![0xCC] // INT3 instruction
    }
    
    /// Generate NOP instruction (for removing breakpoints)
    pub fn generate_nop_instruction() -> Vec<u8> {
        vec![0x90] // NOP instruction
    }
}

/// Code generation helpers for breakpoint support
pub struct BreakpointCodeGen;

impl BreakpointCodeGen {
    /// Generate a breakpoint handler routine
    pub fn generate_breakpoint_handler() -> Vec<u8> {
        let mut code = Vec::new();
        
        // This is the INT3 handler that will be called when a breakpoint is hit
        // Save all registers
        code.push(0x60); // pushad (save all general purpose registers)
        
        // Save flags
        code.push(0x9C); // pushf
        
        // Call our breakpoint handling logic
        // In a real implementation, this would:
        // 1. Identify which breakpoint was hit
        // 2. Check if it's enabled
        // 3. Evaluate conditions if it's conditional
        // 4. Log or notify the debugger
        // 5. Wait for user input or continue
        
        // For now, just print a message
        // mov rax, 1 (write syscall)
        code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00]);
        // mov rdi, 1 (stdout)
        code.extend_from_slice(&[0x48, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00]);
        // lea rsi, [breakpoint_msg]
        code.extend_from_slice(&[0x48, 0x8d, 0x35, 0x00, 0x00, 0x00, 0x00]); // Will be patched
        // mov rdx, msg_length
        code.extend_from_slice(&[0x48, 0xc7, 0xc2, 0x14, 0x00, 0x00, 0x00]); // 20 bytes
        // syscall
        code.extend_from_slice(&[0x0f, 0x05]);
        
        // Restore flags
        code.push(0x9D); // popf
        
        // Restore registers
        code.push(0x61); // popad
        
        // Return from interrupt
        code.push(0xCF); // iret
        
        code
    }
    
    /// Generate code to check for breakpoint conditions
    pub fn generate_conditional_check(condition: &str) -> Vec<u8> {
        let mut code = Vec::new();
        
        // This would evaluate the condition and set flags
        // For now, just a placeholder that always succeeds
        
        // cmp rax, rax (always sets ZF)
        code.extend_from_slice(&[0x48, 0x39, 0xC0]);
        
        code
    }
    
    /// Generate debug trap for single-stepping
    pub fn generate_single_step_trap() -> Vec<u8> {
        let mut code = Vec::new();
        
        // Set trap flag (TF) in RFLAGS for single-stepping
        // pushfq
        code.push(0x9C);
        // pop rax
        code.push(0x58);
        // or rax, 0x100 (set TF bit)
        code.extend_from_slice(&[0x48, 0x0D, 0x00, 0x01, 0x00, 0x00]);
        // push rax
        code.push(0x50);
        // popfq
        code.push(0x9D);
        
        code
    }
    
    /// Generate code to clear single-step mode
    pub fn generate_clear_single_step() -> Vec<u8> {
        let mut code = Vec::new();
        
        // Clear trap flag (TF) in RFLAGS
        // pushfq
        code.push(0x9C);
        // pop rax
        code.push(0x58);
        // and rax, ~0x100 (clear TF bit)
        code.extend_from_slice(&[0x48, 0x25, 0xFF, 0xFE, 0xFF, 0xFF]);
        // push rax
        code.push(0x50);
        // popfq
        code.push(0x9D);
        
        code
    }
}

/// Debugger commands that can be handled
#[derive(Debug, Clone)]
pub enum DebugCommand {
    /// Set breakpoint at line
    Break { file: String, line: u32 },
    /// Set breakpoint at function
    BreakFunction { name: String },
    /// Continue execution
    Continue,
    /// Step over (execute one line)
    Step,
    /// Step into (enter function calls)
    StepInto,
    /// Step out (finish current function)
    StepOut,
    /// List breakpoints
    ListBreakpoints,
    /// Delete breakpoint
    Delete { id: u32 },
    /// Print variable value
    Print { variable: String },
    /// Print stack trace
    Backtrace,
    /// Quit debugger
    Quit,
}

/// Simple debugger interface
pub struct Debugger {
    breakpoint_manager: BreakpointManager,
    single_step_mode: bool,
    current_address: usize,
    breakpoint_hit: bool,
}

impl Debugger {
    pub fn new() -> Self {
        Debugger {
            breakpoint_manager: BreakpointManager::new(),
            single_step_mode: false,
            current_address: 0,
            breakpoint_hit: false,
        }
    }
    
    /// Handle a debug command
    pub fn handle_command(&mut self, command: DebugCommand) -> Result<(), String> {
        match command {
            DebugCommand::Break { file, line } => {
                if let Some(id) = self.breakpoint_manager.set_breakpoint_at_line(&file, line) {
                    println!("Breakpoint {} set at {}:{}", id, file, line);
                    Ok(())
                } else {
                    Err(format!("No code at {}:{}", file, line))
                }
            }
            DebugCommand::BreakFunction { name } => {
                if let Some(id) = self.breakpoint_manager.set_breakpoint_at_function(&name) {
                    println!("Breakpoint {} set at function {}", id, name);
                    Ok(())
                } else {
                    Err(format!("Function {} not found", name))
                }
            }
            DebugCommand::Continue => {
                self.single_step_mode = false;
                self.breakpoint_hit = false;
                println!("Continuing...");
                Ok(())
            }
            DebugCommand::Step => {
                self.single_step_mode = true;
                println!("Stepping...");
                Ok(())
            }
            DebugCommand::ListBreakpoints => {
                let breakpoints = self.breakpoint_manager.get_breakpoints();
                if breakpoints.is_empty() {
                    println!("No breakpoints set");
                } else {
                    for bp in breakpoints {
                        println!("Breakpoint {}: address 0x{:x}, hits: {}, enabled: {}", 
                            bp.id, bp.address, bp.hit_count, bp.enabled);
                        if let Some(ref loc) = bp.source_location {
                            println!("  at {}:{}", loc.file, loc.line);
                        }
                    }
                }
                Ok(())
            }
            DebugCommand::Delete { id } => {
                if self.breakpoint_manager.remove_breakpoint(id) {
                    println!("Breakpoint {} deleted", id);
                    Ok(())
                } else {
                    Err(format!("Breakpoint {} not found", id))
                }
            }
            _ => {
                println!("Command not yet implemented");
                Ok(())
            }
        }
    }
}

/// Generate breakpoint data section
pub fn generate_breakpoint_data() -> Vec<u8> {
    let mut data = Vec::new();
    
    // Breakpoint hit message
    data.extend_from_slice(b"[BREAKPOINT HIT]\n");
    
    // Single-step message
    data.extend_from_slice(b"[SINGLE STEP]\n");
    
    // Continue message
    data.extend_from_slice(b"[CONTINUING]\n");
    
    data
}