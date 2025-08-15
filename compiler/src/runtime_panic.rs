// 🚨 RUNTIME PANIC HANDLER
// Handles runtime errors with stack trace support

use std::io::Write;

/// Emit assembly code for panic handler
pub fn emit_panic_handler() -> Vec<u8> {
    let mut code = Vec::new();
    
    // This is our panic handler that will be called on errors
    // It prints an error message and stack trace
    
    // Save all registers
    // pushfq (save flags)
    code.push(0x9c);
    
    // push rax
    code.push(0x50);
    // push rbx
    code.push(0x53);
    // push rcx
    code.push(0x51);
    // push rdx
    code.push(0x52);
    // push rsi
    code.push(0x56);
    // push rdi
    code.push(0x57);
    // push r8
    code.extend_from_slice(&[0x41, 0x50]);
    // push r9
    code.extend_from_slice(&[0x41, 0x51]);
    
    // Print panic message
    // mov rax, 1 (write syscall)
    code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00]);
    // mov rdi, 2 (stderr)
    code.extend_from_slice(&[0x48, 0xc7, 0xc7, 0x02, 0x00, 0x00, 0x00]);
    // lea rsi, [panic_msg] (message to print)
    // We'll patch this address later
    code.extend_from_slice(&[0x48, 0x8d, 0x35, 0x00, 0x00, 0x00, 0x00]);
    // mov rdx, msg_len
    code.extend_from_slice(&[0x48, 0xc7, 0xc2, 0x20, 0x00, 0x00, 0x00]); // 32 bytes
    // syscall
    code.extend_from_slice(&[0x0f, 0x05]);
    
    // Walk the stack to print trace
    // mov rbx, rbp (start with current frame)
    code.extend_from_slice(&[0x48, 0x89, 0xeb]);
    
    // Stack walking loop (simplified)
    // We'll walk up to 10 frames
    // mov rcx, 10
    code.extend_from_slice(&[0x48, 0xc7, 0xc1, 0x0a, 0x00, 0x00, 0x00]);
    
    // Loop start
    let loop_start = code.len();
    
    // Check if rbx is 0 (end of stack)
    // test rbx, rbx
    code.extend_from_slice(&[0x48, 0x85, 0xdb]);
    // jz end_loop
    code.extend_from_slice(&[0x74, 0x20]); // Jump forward if zero
    
    // Get return address from frame
    // mov rax, [rbx + 8]
    code.extend_from_slice(&[0x48, 0x8b, 0x43, 0x08]);
    
    // Print the address (simplified - just store it)
    // In a real implementation, we'd format and print it
    
    // Move to previous frame
    // mov rbx, [rbx] (previous rbp)
    code.extend_from_slice(&[0x48, 0x8b, 0x1b]);
    
    // Decrement counter
    // dec rcx
    code.extend_from_slice(&[0x48, 0xff, 0xc9]);
    // jnz loop_start
    code.push(0x75);
    code.push((loop_start as i8).wrapping_sub(code.len() as i8 + 1) as u8);
    
    // End loop - restore registers and exit
    // pop r9
    code.extend_from_slice(&[0x41, 0x59]);
    // pop r8
    code.extend_from_slice(&[0x41, 0x58]);
    // pop rdi
    code.push(0x5f);
    // pop rsi
    code.push(0x5e);
    // pop rdx
    code.push(0x5a);
    // pop rcx
    code.push(0x59);
    // pop rbx
    code.push(0x5b);
    // pop rax
    code.push(0x58);
    // popfq
    code.push(0x9d);
    
    // Exit with error code
    // mov rax, 60 (exit syscall)
    code.extend_from_slice(&[0x48, 0xc7, 0xc0, 0x3c, 0x00, 0x00, 0x00]);
    // mov rdi, 1 (error code)
    code.extend_from_slice(&[0x48, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00]);
    // syscall
    code.extend_from_slice(&[0x0f, 0x05]);
    
    code
}

/// Emit code to call panic handler
pub fn emit_panic_call() -> Vec<u8> {
    let mut code = Vec::new();
    
    // Call our panic handler
    // We'll patch this address later with the actual handler location
    // call panic_handler
    code.push(0xe8);
    code.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Relative offset to be patched
    
    code
}

/// Emit bounds check with panic on failure
pub fn emit_bounds_check(index_reg: u8, array_size: i64) -> Vec<u8> {
    let mut code = Vec::new();
    
    // Assume index is in the given register
    // Compare with array size
    // cmp reg, array_size
    code.push(0x48);
    code.push(0x81);
    code.push(0xf8 | index_reg);
    code.extend_from_slice(&(array_size as u32).to_le_bytes());
    
    // jae panic (jump if above or equal - out of bounds)
    code.extend_from_slice(&[0x73, 0x05]); // Skip next instruction if in bounds
    
    // Call panic handler
    code.extend_from_slice(&emit_panic_call());
    
    code
}

/// Emit null pointer check
pub fn emit_null_check(ptr_reg: u8) -> Vec<u8> {
    let mut code = Vec::new();
    
    // test reg, reg
    code.push(0x48);
    code.push(0x85);
    code.push(0xc0 | (ptr_reg << 3) | ptr_reg);
    
    // jz panic
    code.extend_from_slice(&[0x74, 0x05]); // Skip if not null
    
    // Call panic handler
    code.extend_from_slice(&emit_panic_call());
    
    code
}

/// Generate panic message data
pub fn generate_panic_messages() -> Vec<u8> {
    let mut data = Vec::new();
    
    // Panic messages
    data.extend_from_slice(b"PANIC: Runtime error!\n");
    data.extend_from_slice(b"Stack trace:\n");
    data.extend_from_slice(b"  at function ");
    data.extend_from_slice(b"  at address 0x");
    data.extend_from_slice(b"\n");
    
    data
}