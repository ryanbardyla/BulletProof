// 🐛 DEBUG SYMBOLS GENERATION
// Generates DWARF debug information for ELF executables

use std::collections::HashMap;

// DWARF format constants
const DW_TAG_COMPILE_UNIT: u8 = 0x11;
const DW_TAG_SUBPROGRAM: u8 = 0x2e;
const DW_TAG_VARIABLE: u8 = 0x34;
const DW_TAG_BASE_TYPE: u8 = 0x24;
const DW_TAG_FORMAL_PARAMETER: u8 = 0x05;

const DW_AT_NAME: u8 = 0x03;
const DW_AT_LOW_PC: u8 = 0x11;
const DW_AT_HIGH_PC: u8 = 0x12;
const DW_AT_LANGUAGE: u8 = 0x13;
const DW_AT_PRODUCER: u8 = 0x25;
const DW_AT_STMT_LIST: u8 = 0x10;
const DW_AT_COMP_DIR: u8 = 0x1b;
const DW_AT_TYPE: u8 = 0x49;
const DW_AT_LOCATION: u8 = 0x02;
const DW_AT_DECL_FILE: u8 = 0x3a;
const DW_AT_DECL_LINE: u8 = 0x3b;
const DW_AT_BYTE_SIZE: u8 = 0x0b;
const DW_AT_ENCODING: u8 = 0x3e;

const DW_FORM_ADDR: u8 = 0x01;
const DW_FORM_DATA2: u8 = 0x05;
const DW_FORM_DATA4: u8 = 0x06;
const DW_FORM_STRING: u8 = 0x08;
const DW_FORM_DATA1: u8 = 0x0b;
const DW_FORM_STRP: u8 = 0x0e;
const DW_FORM_REF4: u8 = 0x13;

const DW_LANG_C99: u16 = 0x0c;  // We'll pretend to be C for compatibility

const DW_ATE_SIGNED: u8 = 0x05;
const DW_ATE_FLOAT: u8 = 0x04;

#[derive(Debug, Clone)]
pub struct DebugInfo {
    pub compilation_unit: CompilationUnit,
    pub functions: Vec<FunctionDebugInfo>,
    pub variables: Vec<VariableDebugInfo>,
    pub line_program: LineProgram,
    pub string_table: Vec<String>,
    pub abbrev_table: Vec<AbbrevEntry>,
}

#[derive(Debug, Clone)]
pub struct CompilationUnit {
    pub name: String,
    pub producer: String,
    pub language: u16,
    pub comp_dir: String,
    pub low_pc: u64,
    pub high_pc: u64,
}

#[derive(Debug, Clone)]
pub struct FunctionDebugInfo {
    pub name: String,
    pub low_pc: u64,
    pub high_pc: u64,
    pub file: String,
    pub line: u32,
    pub params: Vec<ParameterDebugInfo>,
    pub locals: Vec<VariableDebugInfo>,
}

#[derive(Debug, Clone)]
pub struct ParameterDebugInfo {
    pub name: String,
    pub type_name: String,
    pub location: i32,  // Offset from frame pointer
}

#[derive(Debug, Clone)]
pub struct VariableDebugInfo {
    pub name: String,
    pub type_name: String,
    pub location: i32,  // Offset from frame pointer
    pub file: String,
    pub line: u32,
}

#[derive(Debug, Clone)]
pub struct LineProgram {
    pub minimum_instruction_length: u8,
    pub default_is_stmt: u8,
    pub line_base: i8,
    pub line_range: u8,
    pub opcode_base: u8,
    pub file_names: Vec<String>,
    pub line_entries: Vec<LineEntry>,
}

#[derive(Debug, Clone)]
pub struct LineEntry {
    pub address: u64,
    pub file: u32,
    pub line: u32,
    pub column: u32,
}

#[derive(Debug, Clone)]
pub struct AbbrevEntry {
    pub code: u32,
    pub tag: u8,
    pub has_children: bool,
    pub attributes: Vec<(u8, u8)>,  // (attribute, form) pairs
}

impl DebugInfo {
    pub fn new(source_file: String, output_file: String) -> Self {
        let mut debug_info = DebugInfo {
            compilation_unit: CompilationUnit {
                name: source_file.clone(),
                producer: "NeuronLang Compiler v0.1.0".to_string(),
                language: DW_LANG_C99,
                comp_dir: std::env::current_dir()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
                low_pc: 0x401000,  // Default code start
                high_pc: 0x402000,  // Will be updated
            },
            functions: Vec::new(),
            variables: Vec::new(),
            line_program: LineProgram {
                minimum_instruction_length: 1,
                default_is_stmt: 1,
                line_base: -5,
                line_range: 14,
                opcode_base: 13,
                file_names: vec![source_file],
                line_entries: Vec::new(),
            },
            string_table: Vec::new(),
            abbrev_table: Self::create_abbrev_table(),
        };
        
        // Add standard strings to string table
        debug_info.add_string("NeuronLang Compiler v0.1.0");
        debug_info.add_string("int");
        debug_info.add_string("float");
        debug_info.add_string("string");
        
        debug_info
    }
    
    fn create_abbrev_table() -> Vec<AbbrevEntry> {
        vec![
            // Compilation unit
            AbbrevEntry {
                code: 1,
                tag: DW_TAG_COMPILE_UNIT,
                has_children: true,
                attributes: vec![
                    (DW_AT_NAME, DW_FORM_STRP),
                    (DW_AT_PRODUCER, DW_FORM_STRP),
                    (DW_AT_LANGUAGE, DW_FORM_DATA2),
                    (DW_AT_LOW_PC, DW_FORM_ADDR),
                    (DW_AT_HIGH_PC, DW_FORM_ADDR),
                    (DW_AT_STMT_LIST, DW_FORM_DATA4),
                    (DW_AT_COMP_DIR, DW_FORM_STRP),
                ],
            },
            // Subprogram (function)
            AbbrevEntry {
                code: 2,
                tag: DW_TAG_SUBPROGRAM,
                has_children: true,
                attributes: vec![
                    (DW_AT_NAME, DW_FORM_STRP),
                    (DW_AT_LOW_PC, DW_FORM_ADDR),
                    (DW_AT_HIGH_PC, DW_FORM_ADDR),
                    (DW_AT_DECL_FILE, DW_FORM_DATA1),
                    (DW_AT_DECL_LINE, DW_FORM_DATA2),
                ],
            },
            // Variable
            AbbrevEntry {
                code: 3,
                tag: DW_TAG_VARIABLE,
                has_children: false,
                attributes: vec![
                    (DW_AT_NAME, DW_FORM_STRP),
                    (DW_AT_TYPE, DW_FORM_REF4),
                    (DW_AT_LOCATION, DW_FORM_DATA4),
                    (DW_AT_DECL_FILE, DW_FORM_DATA1),
                    (DW_AT_DECL_LINE, DW_FORM_DATA2),
                ],
            },
            // Base type
            AbbrevEntry {
                code: 4,
                tag: DW_TAG_BASE_TYPE,
                has_children: false,
                attributes: vec![
                    (DW_AT_NAME, DW_FORM_STRP),
                    (DW_AT_BYTE_SIZE, DW_FORM_DATA1),
                    (DW_AT_ENCODING, DW_FORM_DATA1),
                ],
            },
            // Parameter
            AbbrevEntry {
                code: 5,
                tag: DW_TAG_FORMAL_PARAMETER,
                has_children: false,
                attributes: vec![
                    (DW_AT_NAME, DW_FORM_STRP),
                    (DW_AT_TYPE, DW_FORM_REF4),
                    (DW_AT_LOCATION, DW_FORM_DATA4),
                ],
            },
        ]
    }
    
    pub fn add_string(&mut self, s: &str) -> u32 {
        let offset = self.string_table.len() as u32;
        self.string_table.push(s.to_string());
        offset
    }
    
    pub fn add_function(&mut self, func: FunctionDebugInfo) {
        self.functions.push(func);
    }
    
    pub fn add_line_entry(&mut self, address: u64, file: u32, line: u32, column: u32) {
        self.line_program.line_entries.push(LineEntry {
            address,
            file,
            line,
            column,
        });
    }
    
    pub fn generate_debug_sections(&self) -> DebugSections {
        DebugSections {
            debug_info: self.generate_debug_info_section(),
            debug_abbrev: self.generate_debug_abbrev_section(),
            debug_str: self.generate_debug_str_section(),
            debug_line: self.generate_debug_line_section(),
        }
    }
    
    fn generate_debug_info_section(&self) -> Vec<u8> {
        let mut data = Vec::new();
        
        // Compilation unit header
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Length (will patch)
        data.extend_from_slice(&[0x04, 0x00]); // DWARF version 4
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Abbrev offset
        data.push(8); // Address size
        
        // Compilation unit DIE
        self.emit_uleb128(&mut data, 1); // Abbrev code for compile unit
        
        // Attributes
        data.extend_from_slice(&self.get_string_offset(&self.compilation_unit.name).to_le_bytes());
        data.extend_from_slice(&self.get_string_offset(&self.compilation_unit.producer).to_le_bytes());
        data.extend_from_slice(&self.compilation_unit.language.to_le_bytes());
        data.extend_from_slice(&self.compilation_unit.low_pc.to_le_bytes());
        data.extend_from_slice(&self.compilation_unit.high_pc.to_le_bytes());
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Stmt list offset
        data.extend_from_slice(&self.get_string_offset(&self.compilation_unit.comp_dir).to_le_bytes());
        
        // Functions
        for func in &self.functions {
            self.emit_uleb128(&mut data, 2); // Abbrev code for subprogram
            data.extend_from_slice(&self.get_string_offset(&func.name).to_le_bytes());
            data.extend_from_slice(&func.low_pc.to_le_bytes());
            data.extend_from_slice(&func.high_pc.to_le_bytes());
            data.push(1); // File index
            data.extend_from_slice(&(func.line as u16).to_le_bytes());
            
            // Parameters and locals
            for param in &func.params {
                self.emit_uleb128(&mut data, 5); // Abbrev code for parameter
                data.extend_from_slice(&self.get_string_offset(&param.name).to_le_bytes());
                data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Type ref
                data.extend_from_slice(&param.location.to_le_bytes());
            }
            
            for var in &func.locals {
                self.emit_uleb128(&mut data, 3); // Abbrev code for variable
                data.extend_from_slice(&self.get_string_offset(&var.name).to_le_bytes());
                data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Type ref
                data.extend_from_slice(&var.location.to_le_bytes());
                data.push(1); // File index
                data.extend_from_slice(&(var.line as u16).to_le_bytes());
            }
            
            data.push(0); // End of children
        }
        
        data.push(0); // End of compilation unit children
        
        // Patch length
        let length = (data.len() - 4) as u32;
        data[0..4].copy_from_slice(&length.to_le_bytes());
        
        data
    }
    
    fn generate_debug_abbrev_section(&self) -> Vec<u8> {
        let mut data = Vec::new();
        
        for entry in &self.abbrev_table {
            self.emit_uleb128(&mut data, entry.code);
            self.emit_uleb128(&mut data, entry.tag as u32);
            data.push(if entry.has_children { 1 } else { 0 });
            
            for (attr, form) in &entry.attributes {
                self.emit_uleb128(&mut data, *attr as u32);
                self.emit_uleb128(&mut data, *form as u32);
            }
            
            data.push(0); // End of attributes
            data.push(0);
        }
        
        data.push(0); // End of abbrev table
        
        data
    }
    
    fn generate_debug_str_section(&self) -> Vec<u8> {
        let mut data = Vec::new();
        
        for s in &self.string_table {
            data.extend_from_slice(s.as_bytes());
            data.push(0); // Null terminator
        }
        
        data
    }
    
    fn generate_debug_line_section(&self) -> Vec<u8> {
        let mut data = Vec::new();
        
        // Line program header
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Length (will patch)
        data.extend_from_slice(&[0x04, 0x00]); // DWARF version 4
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Header length (will patch)
        
        data.push(self.line_program.minimum_instruction_length);
        data.push(1); // Maximum operations per instruction
        data.push(self.line_program.default_is_stmt);
        data.push(self.line_program.line_base as u8);
        data.push(self.line_program.line_range);
        data.push(self.line_program.opcode_base);
        
        // Standard opcode lengths
        for _ in 1..self.line_program.opcode_base {
            data.push(0);
        }
        
        // Include directories (none for now)
        data.push(0);
        
        // File names
        for file in &self.line_program.file_names {
            data.extend_from_slice(file.as_bytes());
            data.push(0); // Null terminator
            self.emit_uleb128(&mut data, 0); // Directory index
            self.emit_uleb128(&mut data, 0); // Modification time
            self.emit_uleb128(&mut data, 0); // File length
        }
        data.push(0); // End of file names
        
        // Line number program
        for entry in &self.line_program.line_entries {
            // Set address
            data.push(0x00); // Extended opcode
            self.emit_uleb128(&mut data, 9); // Length
            data.push(0x02); // DW_LNE_set_address
            data.extend_from_slice(&entry.address.to_le_bytes());
            
            // Set file
            data.push(0x04); // DW_LNS_set_file
            self.emit_uleb128(&mut data, entry.file);
            
            // Advance line
            data.push(0x03); // DW_LNS_advance_line
            self.emit_sleb128(&mut data, entry.line as i32);
            
            // Copy (append row to matrix)
            data.push(0x01); // DW_LNS_copy
        }
        
        // End sequence
        data.push(0x00); // Extended opcode
        self.emit_uleb128(&mut data, 1);
        data.push(0x01); // DW_LNE_end_sequence
        
        // Patch lengths
        let total_length = (data.len() - 4) as u32;
        data[0..4].copy_from_slice(&total_length.to_le_bytes());
        
        let header_length = (data.len() - 10) as u32;
        data[6..10].copy_from_slice(&header_length.to_le_bytes());
        
        data
    }
    
    fn get_string_offset(&self, s: &str) -> u32 {
        let mut offset = 0u32;
        for string in &self.string_table {
            if string == s {
                return offset;
            }
            offset += (string.len() + 1) as u32; // +1 for null terminator
        }
        0 // Not found
    }
    
    fn emit_uleb128(&self, data: &mut Vec<u8>, mut value: u32) {
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            data.push(byte);
            if value == 0 {
                break;
            }
        }
    }
    
    fn emit_sleb128(&self, data: &mut Vec<u8>, mut value: i32) {
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            let sign = (byte & 0x40) != 0;
            if (value == 0 && !sign) || (value == -1 && sign) {
                data.push(byte);
                break;
            } else {
                byte |= 0x80;
                data.push(byte);
            }
        }
    }
}

pub struct DebugSections {
    pub debug_info: Vec<u8>,
    pub debug_abbrev: Vec<u8>,
    pub debug_str: Vec<u8>,
    pub debug_line: Vec<u8>,
}

// Helper to add debug sections to ELF
pub fn add_debug_sections_to_elf(elf: &mut Vec<u8>, debug_sections: DebugSections) {
    // This would modify the ELF to include .debug_info, .debug_abbrev, .debug_str, .debug_line sections
    // For now, we'll just append them as a placeholder
    elf.extend_from_slice(&debug_sections.debug_info);
    elf.extend_from_slice(&debug_sections.debug_abbrev);
    elf.extend_from_slice(&debug_sections.debug_str);
    elf.extend_from_slice(&debug_sections.debug_line);
}