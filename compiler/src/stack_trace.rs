// 🔍 STACK TRACE SUPPORT
// Runtime stack trace generation for better debugging

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct StackFrame {
    pub function_name: String,
    pub return_address: usize,
    pub frame_pointer: usize,
    pub source_location: Option<SourceLocation>,
}

#[derive(Debug, Clone)]
pub struct SourceLocation {
    pub file: String,
    pub line: usize,
    pub column: usize,
}

pub struct StackTraceInfo {
    // Map from code address to function info
    pub function_map: HashMap<usize, FunctionInfo>,
    // Map from code address to source location
    pub source_map: HashMap<usize, SourceLocation>,
    // Current stack frames
    pub frames: Vec<StackFrame>,
}

#[derive(Debug, Clone)]
pub struct FunctionInfo {
    pub name: String,
    pub start_address: usize,
    pub end_address: usize,
    pub source_file: String,
    pub source_line: usize,
}

impl StackTraceInfo {
    pub fn new() -> Self {
        StackTraceInfo {
            function_map: HashMap::new(),
            source_map: HashMap::new(),
            frames: Vec::new(),
        }
    }
    
    pub fn register_function(&mut self, info: FunctionInfo) {
        for addr in info.start_address..=info.end_address {
            self.function_map.insert(addr, info.clone());
        }
    }
    
    pub fn register_source_mapping(&mut self, address: usize, location: SourceLocation) {
        self.source_map.insert(address, location);
    }
    
    pub fn push_frame(&mut self, function_name: String, return_address: usize, frame_pointer: usize) {
        let source_location = self.source_map.get(&return_address).cloned();
        self.frames.push(StackFrame {
            function_name,
            return_address,
            frame_pointer,
            source_location,
        });
    }
    
    pub fn pop_frame(&mut self) -> Option<StackFrame> {
        self.frames.pop()
    }
    
    pub fn format_stack_trace(&self) -> String {
        let mut output = String::new();
        output.push_str("Stack trace (most recent call first):\n");
        
        for (i, frame) in self.frames.iter().rev().enumerate() {
            output.push_str(&format!("  #{} {} at 0x{:x}", 
                i, 
                frame.function_name,
                frame.return_address
            ));
            
            if let Some(ref loc) = frame.source_location {
                output.push_str(&format!(" ({}:{}:{})", 
                    loc.file, 
                    loc.line, 
                    loc.column
                ));
            }
            
            output.push('\n');
        }
        
        output
    }
    
    pub fn unwind_stack(&mut self, rbp: usize, rsp: usize, max_frames: usize) -> Vec<StackFrame> {
        let mut frames = Vec::new();
        let mut current_rbp = rbp;
        let mut frame_count = 0;
        
        // Walk the stack using frame pointers
        while current_rbp != 0 && frame_count < max_frames {
            // The return address is typically at [rbp + 8]
            let return_addr_ptr = current_rbp + 8;
            
            // Look up function info for this return address
            if let Some(func_info) = self.function_map.get(&return_addr_ptr) {
                let source_location = self.source_map.get(&return_addr_ptr).cloned();
                
                frames.push(StackFrame {
                    function_name: func_info.name.clone(),
                    return_address: return_addr_ptr,
                    frame_pointer: current_rbp,
                    source_location,
                });
            }
            
            // Move to the previous frame
            // The previous rbp is stored at [rbp]
            // In a real implementation, we'd read memory here
            // For now, we'll break after one iteration
            break;
            
            frame_count += 1;
        }
        
        frames
    }
}

// Assembly code snippets for stack trace support
pub struct StackTraceCode {
    pub push_frame_code: Vec<u8>,
    pub pop_frame_code: Vec<u8>,
    pub capture_trace_code: Vec<u8>,
}

impl StackTraceCode {
    pub fn new() -> Self {
        StackTraceCode {
            push_frame_code: Self::generate_push_frame(),
            pop_frame_code: Self::generate_pop_frame(),
            capture_trace_code: Self::generate_capture_trace(),
        }
    }
    
    // Generate code to save stack frame info
    fn generate_push_frame() -> Vec<u8> {
        let mut code = Vec::new();
        
        // Save current RBP and return address
        // push rbp
        code.push(0x55);
        
        // mov rbp, rsp
        code.extend_from_slice(&[0x48, 0x89, 0xe5]);
        
        // Push return address location for tracking
        // This would be used by the runtime to track calls
        
        code
    }
    
    // Generate code to restore stack frame
    fn generate_pop_frame() -> Vec<u8> {
        let mut code = Vec::new();
        
        // mov rsp, rbp
        code.extend_from_slice(&[0x48, 0x89, 0xec]);
        
        // pop rbp
        code.push(0x5d);
        
        code
    }
    
    // Generate code to capture current stack trace
    fn generate_capture_trace() -> Vec<u8> {
        let mut code = Vec::new();
        
        // Save all registers we'll use
        // push rax
        code.push(0x50);
        // push rbx  
        code.push(0x53);
        // push rcx
        code.push(0x51);
        // push rdx
        code.push(0x52);
        
        // Get current RBP for stack walking
        // mov rax, rbp
        code.extend_from_slice(&[0x48, 0x89, 0xe8]);
        
        // Call our stack trace handler
        // This would be a call to a runtime function
        
        // Restore registers
        // pop rdx
        code.push(0x5a);
        // pop rcx
        code.push(0x59);
        // pop rbx
        code.push(0x5b);
        // pop rax
        code.push(0x58);
        
        code
    }
}

// Panic handler that prints stack trace
pub fn panic_with_trace(message: &str, trace_info: &StackTraceInfo) {
    eprintln!("PANIC: {}", message);
    eprintln!("{}", trace_info.format_stack_trace());
    std::process::exit(1);
}

// Signal handler for segfaults and other errors
pub fn install_signal_handlers() {
    // In a real implementation, we'd use signal handlers
    // to catch SIGSEGV, SIGBUS, etc. and print stack traces
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stack_trace_formatting() {
        let mut trace = StackTraceInfo::new();
        
        trace.push_frame("main".to_string(), 0x1000, 0x7fff0000);
        trace.push_frame("calculate".to_string(), 0x1100, 0x7fff0100);
        trace.push_frame("helper".to_string(), 0x1200, 0x7fff0200);
        
        let formatted = trace.format_stack_trace();
        assert!(formatted.contains("main"));
        assert!(formatted.contains("calculate"));
        assert!(formatted.contains("helper"));
    }
    
    #[test]
    fn test_function_registration() {
        let mut trace = StackTraceInfo::new();
        
        let func_info = FunctionInfo {
            name: "test_func".to_string(),
            start_address: 0x1000,
            end_address: 0x1100,
            source_file: "test.nl".to_string(),
            source_line: 42,
        };
        
        trace.register_function(func_info);
        
        assert!(trace.function_map.contains_key(&0x1000));
        assert!(trace.function_map.contains_key(&0x1050));
        assert!(trace.function_map.contains_key(&0x1100));
    }
}