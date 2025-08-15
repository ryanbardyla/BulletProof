// 🔍 PEEPHOLE OPTIMIZER
// Optimizes small sequences of x86-64 machine code instructions

use std::collections::VecDeque;

// Instruction patterns for x86-64
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    // Stack operations
    Push(Operand),
    Pop(Operand),
    
    // Move operations
    Mov(Operand, Operand),
    
    // Arithmetic
    Add(Operand, Operand),
    Sub(Operand, Operand),
    Mul(Operand),
    Div(Operand),
    Neg(Operand),
    Inc(Operand),
    Dec(Operand),
    
    // Logical
    And(Operand, Operand),
    Or(Operand, Operand),
    Xor(Operand, Operand),
    Not(Operand),
    
    // Comparison
    Cmp(Operand, Operand),
    Test(Operand, Operand),
    
    // Jumps
    Jmp(i32),
    Je(i32),
    Jne(i32),
    Jg(i32),
    Jl(i32),
    Jge(i32),
    Jle(i32),
    
    // Other
    Call(String),
    Ret,
    Nop,
    Raw(Vec<u8>), // Raw bytes that we don't optimize
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    Register(Register),
    Immediate(i64),
    Memory(Register, i32), // [reg + offset]
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Register {
    RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI,
    R8, R9, R10, R11, R12, R13, R14, R15,
}

pub struct PeepholeOptimizer {
    window_size: usize,
    optimization_passes: u32,
}

impl PeepholeOptimizer {
    pub fn new() -> Self {
        PeepholeOptimizer {
            window_size: 5, // Look at 5 instructions at a time
            optimization_passes: 3, // Run optimization passes multiple times
        }
    }
    
    // Main optimization function for raw bytes
    pub fn optimize(&self, code: Vec<u8>) -> Vec<u8> {
        // First decode the machine code into instructions
        let mut instructions = self.decode_instructions(&code);
        
        // Run multiple optimization passes
        for _ in 0..self.optimization_passes {
            let old_len = instructions.len();
            instructions = self.optimize_instructions(instructions);
            
            // Stop if no more optimizations found
            if instructions.len() == old_len {
                break;
            }
        }
        
        // Encode back to machine code
        self.encode_instructions(&instructions)
    }
    
    // Optimize instruction sequence
    fn optimize_instructions(&self, instructions: Vec<Instruction>) -> Vec<Instruction> {
        let mut result = Vec::new();
        let mut i = 0;
        
        while i < instructions.len() {
            let mut optimized = false;
            
            // Try different peephole patterns
            if !optimized && i + 1 < instructions.len() {
                optimized = self.try_optimize_pair(&instructions[i], &instructions[i + 1], &mut result);
                if optimized {
                    i += 2;
                    continue;
                }
            }
            
            if !optimized && i + 2 < instructions.len() {
                optimized = self.try_optimize_triple(
                    &instructions[i], 
                    &instructions[i + 1], 
                    &instructions[i + 2], 
                    &mut result
                );
                if optimized {
                    i += 3;
                    continue;
                }
            }
            
            // No optimization found, keep the instruction
            if !optimized {
                result.push(instructions[i].clone());
                i += 1;
            }
        }
        
        result
    }
    
    // Try to optimize a pair of instructions
    fn try_optimize_pair(&self, inst1: &Instruction, inst2: &Instruction, result: &mut Vec<Instruction>) -> bool {
        match (inst1, inst2) {
            // OPTIMIZATION 1: Push followed by Pop to same register
            // push rax; pop rax => (remove both)
            (Instruction::Push(Operand::Register(r1)), Instruction::Pop(Operand::Register(r2))) 
                if r1 == r2 => {
                // Eliminate redundant push/pop
                true
            }
            
            // OPTIMIZATION 2: Push followed by Pop to different register
            // push rax; pop rbx => mov rbx, rax
            (Instruction::Push(Operand::Register(r1)), Instruction::Pop(Operand::Register(r2))) => {
                result.push(Instruction::Mov(
                    Operand::Register(*r2),
                    Operand::Register(*r1)
                ));
                true
            }
            
            // OPTIMIZATION 3: Two moves that cancel
            // mov rax, rbx; mov rbx, rax => mov rax, rbx
            (Instruction::Mov(Operand::Register(r1), Operand::Register(r2)),
             Instruction::Mov(Operand::Register(r3), Operand::Register(r4)))
                if r1 == r4 && r2 == r3 => {
                result.push(inst1.clone());
                true
            }
            
            // OPTIMIZATION 4: Move to itself
            // mov rax, rax => (remove)
            (Instruction::Mov(Operand::Register(r1), Operand::Register(r2)), _)
                if r1 == r2 => {
                result.push(inst2.clone());
                true
            }
            
            // OPTIMIZATION 5: Add/Sub with 0
            // add rax, 0 => (remove)
            (Instruction::Add(_, Operand::Immediate(0)), _) |
            (Instruction::Sub(_, Operand::Immediate(0)), _) => {
                result.push(inst2.clone());
                true
            }
            
            // OPTIMIZATION 6: Multiply by 1
            // mul 1 => (remove)
            (Instruction::Mul(Operand::Immediate(1)), _) => {
                result.push(inst2.clone());
                true
            }
            
            // OPTIMIZATION 7: XOR with itself (common idiom for zeroing)
            // Already optimal, but we can recognize it
            (Instruction::Xor(Operand::Register(r1), Operand::Register(r2)), _)
                if r1 == r2 => {
                // This is actually good - it's the fastest way to zero a register
                result.push(inst1.clone());
                result.push(inst2.clone());
                false // Don't optimize further
            }
            
            // OPTIMIZATION 8: Consecutive increments
            // inc rax; inc rax => add rax, 2
            (Instruction::Inc(Operand::Register(r1)), Instruction::Inc(Operand::Register(r2)))
                if r1 == r2 => {
                result.push(Instruction::Add(
                    Operand::Register(*r1),
                    Operand::Immediate(2)
                ));
                true
            }
            
            // OPTIMIZATION 9: Consecutive decrements
            // dec rax; dec rax => sub rax, 2
            (Instruction::Dec(Operand::Register(r1)), Instruction::Dec(Operand::Register(r2)))
                if r1 == r2 => {
                result.push(Instruction::Sub(
                    Operand::Register(*r1),
                    Operand::Immediate(2)
                ));
                true
            }
            
            _ => false
        }
    }
    
    // Try to optimize a triple of instructions
    fn try_optimize_triple(&self, inst1: &Instruction, inst2: &Instruction, inst3: &Instruction, 
                           result: &mut Vec<Instruction>) -> bool {
        match (inst1, inst2, inst3) {
            // OPTIMIZATION 10: Push, operation, pop pattern
            // push rax; add [rsp], 5; pop rax => add rax, 5
            (Instruction::Push(Operand::Register(r1)),
             Instruction::Add(Operand::Memory(Register::RSP, 0), imm),
             Instruction::Pop(Operand::Register(r2)))
                if r1 == r2 => {
                result.push(Instruction::Add(
                    Operand::Register(*r1),
                    imm.clone()
                ));
                true
            }
            
            // OPTIMIZATION 11: Load, operation, store pattern
            // mov rax, [rbp-8]; add rax, 1; mov [rbp-8], rax => add [rbp-8], 1
            (Instruction::Mov(Operand::Register(r1), Operand::Memory(base1, off1)),
             Instruction::Add(Operand::Register(r2), imm),
             Instruction::Mov(Operand::Memory(base2, off2), Operand::Register(r3)))
                if r1 == r2 && r2 == r3 && base1 == base2 && off1 == off2 => {
                result.push(Instruction::Add(
                    Operand::Memory(*base1, *off1),
                    imm.clone()
                ));
                true
            }
            
            // OPTIMIZATION 12: Comparison and conditional jump
            // cmp rax, 0; je label => test rax, rax; je label (more efficient)
            (Instruction::Cmp(Operand::Register(r), Operand::Immediate(0)),
             Instruction::Je(label),
             next) => {
                result.push(Instruction::Test(
                    Operand::Register(*r),
                    Operand::Register(*r)
                ));
                result.push(Instruction::Je(*label));
                result.push(next.clone());
                true
            }
            
            _ => false
        }
    }
    
    // Decode x86-64 machine code into instructions (simplified)
    fn decode_instructions(&self, code: &[u8]) -> Vec<Instruction> {
        let mut instructions = Vec::new();
        let mut i = 0;
        
        while i < code.len() {
            // Try to decode common patterns
            if i + 1 < code.len() {
                match code[i] {
                    // Push register
                    0x50..=0x57 => {
                        let reg = self.decode_register(code[i] - 0x50);
                        instructions.push(Instruction::Push(Operand::Register(reg)));
                        i += 1;
                        continue;
                    }
                    // Pop register
                    0x58..=0x5F => {
                        let reg = self.decode_register(code[i] - 0x58);
                        instructions.push(Instruction::Pop(Operand::Register(reg)));
                        i += 1;
                        continue;
                    }
                    // REX prefix
                    0x48 | 0x49 | 0x4C | 0x4D => {
                        if i + 2 < code.len() {
                            // Handle REX-prefixed instructions
                            let (inst, consumed) = self.decode_rex_instruction(&code[i..]);
                            instructions.push(inst);
                            i += consumed;
                            continue;
                        }
                    }
                    _ => {}
                }
            }
            
            // If we can't decode, store as raw bytes
            instructions.push(Instruction::Raw(vec![code[i]]));
            i += 1;
        }
        
        instructions
    }
    
    // Decode register from encoding
    fn decode_register(&self, encoding: u8) -> Register {
        match encoding & 0x7 {
            0 => Register::RAX,
            1 => Register::RCX,
            2 => Register::RDX,
            3 => Register::RBX,
            4 => Register::RSP,
            5 => Register::RBP,
            6 => Register::RSI,
            7 => Register::RDI,
            _ => Register::RAX,
        }
    }
    
    // Decode REX-prefixed instruction
    fn decode_rex_instruction(&self, code: &[u8]) -> (Instruction, usize) {
        // Simplified decoding for common REX-prefixed instructions
        if code.len() >= 3 && code[0] == 0x48 {
            match code[1] {
                0x89 => {
                    // MOV r/m64, r64
                    let modrm = code[2];
                    let reg = self.decode_register((modrm >> 3) & 0x7);
                    let rm = self.decode_register(modrm & 0x7);
                    return (Instruction::Mov(
                        Operand::Register(rm),
                        Operand::Register(reg)
                    ), 3);
                }
                0x01 => {
                    // ADD r/m64, r64
                    let modrm = code[2];
                    let reg = self.decode_register((modrm >> 3) & 0x7);
                    let rm = self.decode_register(modrm & 0x7);
                    return (Instruction::Add(
                        Operand::Register(rm),
                        Operand::Register(reg)
                    ), 3);
                }
                _ => {}
            }
        }
        
        // Default: treat as raw
        (Instruction::Raw(vec![code[0]]), 1)
    }
    
    // Encode instructions back to machine code
    fn encode_instructions(&self, instructions: &[Instruction]) -> Vec<u8> {
        let mut code = Vec::new();
        
        for inst in instructions {
            match inst {
                Instruction::Push(Operand::Register(reg)) => {
                    code.push(0x50 + self.encode_register(*reg));
                }
                Instruction::Pop(Operand::Register(reg)) => {
                    code.push(0x58 + self.encode_register(*reg));
                }
                Instruction::Mov(Operand::Register(dst), Operand::Register(src)) => {
                    code.push(0x48); // REX.W
                    code.push(0x89); // MOV r/m64, r64
                    code.push(0xC0 + (self.encode_register(*src) << 3) + self.encode_register(*dst));
                }
                Instruction::Add(Operand::Register(dst), Operand::Immediate(imm)) => {
                    if *imm == 1 {
                        // Use INC for add 1
                        code.push(0x48); // REX.W
                        code.push(0xFF); // INC r/m64
                        code.push(0xC0 + self.encode_register(*dst));
                    } else if *imm <= 127 && *imm >= -128 {
                        // Use short immediate
                        code.push(0x48); // REX.W
                        code.push(0x83); // ADD r/m64, imm8
                        code.push(0xC0 + self.encode_register(*dst));
                        code.push(*imm as u8);
                    } else {
                        // Use full immediate
                        code.push(0x48); // REX.W
                        code.push(0x81); // ADD r/m64, imm32
                        code.push(0xC0 + self.encode_register(*dst));
                        code.extend_from_slice(&(*imm as i32).to_le_bytes());
                    }
                }
                Instruction::Sub(Operand::Register(dst), Operand::Immediate(imm)) => {
                    if *imm == 1 {
                        // Use DEC for sub 1
                        code.push(0x48); // REX.W
                        code.push(0xFF); // DEC r/m64
                        code.push(0xC8 + self.encode_register(*dst));
                    } else if *imm <= 127 && *imm >= -128 {
                        // Use short immediate
                        code.push(0x48); // REX.W
                        code.push(0x83); // SUB r/m64, imm8
                        code.push(0xE8 + self.encode_register(*dst));
                        code.push(*imm as u8);
                    } else {
                        // Use full immediate
                        code.push(0x48); // REX.W
                        code.push(0x81); // SUB r/m64, imm32
                        code.push(0xE8 + self.encode_register(*dst));
                        code.extend_from_slice(&(*imm as i32).to_le_bytes());
                    }
                }
                Instruction::Test(Operand::Register(r1), Operand::Register(r2)) => {
                    code.push(0x48); // REX.W
                    code.push(0x85); // TEST r/m64, r64
                    code.push(0xC0 + (self.encode_register(*r2) << 3) + self.encode_register(*r1));
                }
                Instruction::Nop => {
                    code.push(0x90); // NOP
                }
                Instruction::Raw(bytes) => {
                    code.extend_from_slice(bytes);
                }
                _ => {
                    // For unhandled instructions, we could emit a placeholder or panic
                    // For now, emit a NOP
                    code.push(0x90);
                }
            }
        }
        
        code
    }
    
    // Encode register to its x86-64 encoding
    fn encode_register(&self, reg: Register) -> u8 {
        match reg {
            Register::RAX => 0,
            Register::RCX => 1,
            Register::RDX => 2,
            Register::RBX => 3,
            Register::RSP => 4,
            Register::RBP => 5,
            Register::RSI => 6,
            Register::RDI => 7,
            Register::R8 => 0,  // Need REX prefix
            Register::R9 => 1,  // Need REX prefix
            Register::R10 => 2, // Need REX prefix
            Register::R11 => 3, // Need REX prefix
            Register::R12 => 4, // Need REX prefix
            Register::R13 => 5, // Need REX prefix
            Register::R14 => 6, // Need REX prefix
            Register::R15 => 7, // Need REX prefix
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_redundant_push_pop() {
        let optimizer = PeepholeOptimizer::new();
        
        // push rax; pop rax => (nothing)
        let code = vec![0x50, 0x58]; // push rax, pop rax
        let optimized = optimizer.optimize(code);
        assert_eq!(optimized.len(), 0);
    }
    
    #[test]
    fn test_push_pop_different_registers() {
        let optimizer = PeepholeOptimizer::new();
        
        // push rax; pop rcx => mov rcx, rax
        let code = vec![0x50, 0x59]; // push rax, pop rcx
        let optimized = optimizer.optimize(code);
        
        // Should be: mov rcx, rax (48 89 C1)
        assert_eq!(optimized, vec![0x48, 0x89, 0xC1]);
    }
    
    #[test]
    fn test_consecutive_increments() {
        let optimizer = PeepholeOptimizer::new();
        
        let mut instructions = vec![
            Instruction::Inc(Operand::Register(Register::RAX)),
            Instruction::Inc(Operand::Register(Register::RAX)),
        ];
        
        instructions = optimizer.optimize_instructions(instructions);
        
        // Should become: add rax, 2
        assert_eq!(instructions.len(), 1);
        assert!(matches!(
            instructions[0],
            Instruction::Add(Operand::Register(Register::RAX), Operand::Immediate(2))
        ));
    }
}