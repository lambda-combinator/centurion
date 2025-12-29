# === Architecture Parameters ===

XLEN        = 32 
NUM_REGS    = 32
REG_WIDTH   = 32
INST_WIDTH  = 32

# === Register Names ===

REG_NAMES = {
    0: "zero",  1: "ra",    2: "sp",    3: "gp",
    4: "tp",    5: "t0",    6: "t1",    7: "t2", 
    8: "s0",    9: "s1",    10: "a0",   11: "a1", 
    12: "a2",   13: "a3",   14: "a4",   15: "a5",
    16: "a6",   17: "a7",   18: "s2",   19: "s3",
    20: "s4",   21: "s5",   22: "s6",   23: "s7",
    24: "s8",   25: "s9",   26: "s10",  27: "s11",
    28: "t3",   29: "t4",   30: "t5",   31: "t6"
}

# Common indices
REG_ZERO = 0
REG_RA = 1
REG_SP = 2
REG_GP = 3
REG_TP = 4
REG_A0 = 10
REG_A1 = 11

# === Instruction Opcodes ===

# R-type
OPCODE_R_TYPE = 0b0110011

# I-type
OPCODE_ADDI  = 0b0010011
OPCODE_ANDI  = 0b0010011
OPCODE_ORI   = 0b0010011
OPCODE_XORI  = 0b0010011
OPCODE_SLTI  = 0b0010011
OPCODE_SLTIU = 0b0010011
OPCODE_SLLI  = 0b0010011
OPCODE_SRLI  = 0b0010011
OPCODE_SRAI  = 0b0010011

# Load
OPCODE_LOAD = 0b0000011

# Store 
OPCODE_STORE = 0b0100011

# Branch
OPCODE_BRANCH = 0b1100011

# Jump
OPCODE_JAL  = 0b1101111
OPCODE_JALR = 0b1100111

# Upper immediate
OPCODE_LUI   = 0b0110111
OPCODE_AUIPC = 0b0010111

# === Instruction Masks and Shifts ===

# Opcode [6:0]
OPCODE_MASK = 0x7F
OPCODE_SHIFT = 0

# RD [11:7]
RD_MASK = 0x1F
RD_SHIFT = 7

# Funct3 [14:12]
FUNCT3_MASK = 0x7
FUNCT3_SHIFT = 12

# RS1 [19:15]
RS1_MASK = 0x1F
RS1_SHIFT = 15

# RS2 [24:20]
RS2_MASK = 0x1F
RS2_SHIFT = 20

# R-type Funct7 [31:25] 
FUNCT7_MASK = 0x7F
FUNCT7_SHIFT = 25

# I-type Immediate [31:20]
IMM_I_MASK = 0xFFF
IMM_I_SHIFT = 20

# S-type Immediate [31:25|11:7]
IMM_S_HI_MASK = 0x7F
IMM_S_HI_SHIFT = 25
IMM_S_LO_MASK = 0x1F
IMM_S_LO_SHIFT = 7

# B-type Immediate [31|7|30:25|11:8]
IMM_B_BIT11_MASK = 0x1
IMM_B_BIT11_SHIFT = 7
IMM_B_BITS10_5_MASK = 0x3F
IMM_B_BITS10_5_SHIFT = 25
IMM_B_BITS4_1_MASK = 0xF
IMM_B_BITS4_1_SHIFT = 8
IMM_B_BIT12_MASK = 0x1
IMM_B_BIT12_SHIFT = 31

# U-type Immediate [31:12]
IMM_U_MASK = 0xFFFFF
IMM_U_SHIFT = 12

# J-type Immediate [31|19:12|20|30:21]
IMM_J_BIT11_MASK = 0x1
IMM_J_BIT11_SHIFT = 20
IMM_J_BITS19_12_MASK = 0xFF
IMM_J_BITS19_12_SHIFT = 12
IMM_J_BITS10_1_MASK = 0x3FF
IMM_J_BITS10_1_SHIFT = 21
IMM_J_BIT20_MASK = 0x1
IMM_J_BIT20_SHIFT = 31

# === Funct3 Codes ===

# R-type and I-type
FUNCT3_ADD_SUB = 0b000
FUNCT3_SLL     = 0b001
FUNCT3_SLT     = 0b010
FUNCT3_SLTU    = 0b011
FUNCT3_XOR     = 0b100
FUNCT3_SRL_SRA = 0b101
FUNCT3_OR      = 0b110
FUNCT3_AND     = 0b111

# Load
FUNCT3_LB  = 0b000
FUNCT3_LH  = 0b001
FUNCT3_LW  = 0b010
FUNCT3_LBU = 0b100
FUNCT3_LHU = 0b101

# Store
FUNCT3_SB = 0b000
FUNCT3_SH = 0b001
FUNCT3_SW = 0b010

# Branch
FUNCT3_BEQ  = 0b000
FUNCT3_BNE  = 0b001
FUNCT3_BLT  = 0b100
FUNCT3_BGE  = 0b101
FUNCT3_BLTU = 0b110
FUNCT3_BGEU = 0b111

# === Funct7 Codes ===

FUNCT7_NORMAL = 0b0000000
FUNCT7_ALT    = 0b0100000

# === ALU Codes ===

ALU_OP_ADD  = 0b0000
ALU_OP_SUB  = 0b0001
ALU_OP_AND  = 0b0010
ALU_OP_OR   = 0b0011
ALU_OP_XOR  = 0b0100
ALU_OP_SLL  = 0b0101
ALU_OP_SRL  = 0b0110
ALU_OP_SRA  = 0b0111
ALU_OP_SLT  = 0b1000
ALU_OP_SLTU = 0b1001
ALU_OP_MUL  = 0b1010
ALU_OP_DIV  = 0b1011

ALU_OP_PASS_B = 0b1100
ALU_OP_PASS_A = 0b1101


# === Control Signals ===

CTRL_REG_WRITE_YES = 1
CTRL_REG_WRITE_NO  = 0

CTRL_MEM_READ_YES  = 1
CTRL_MEM_READ_NO   = 0

CTRL_MEM_WRITE_YES = 1
CTRL_MEM_WRITE_NO  = 0

CTRL_ALU_SRC_REG = 1
CTRL_ALU_SRC_IMM = 0

CTRL_MEM_TO_REG_ALU = 0
CTRL_MEM_TO_REG_MEM = 1

CTRL_BRANCH_YES = 1
CTRL_BRANCH_NO  = 0

CTRL_BRANCH_TYPE_NONE = 0b00
CTRL_BRANCH_TYPE_BEQ  = 0b00
CTRL_BRANCH_TYPE_BNE  = 0b01
CTRL_BRANCH_TYPE_BLT  = 0b10
CTRL_BRANCH_TYPE_BGE  = 0b11

CTRL_JUMP_YES = 1
CTRL_JUMP_NO  = 0

# === Cache Parameters ===

ICACHE_SIZE          = 1024 # 1 KB
ICACHE_LINE_SIZE     = 32
ICACHE_NUM_LINES     = ICACHE_SIZE // ICACHE_LINE_SIZE
ICACHE_ASSOCIATIVITY = 1

DCACHE_SIZE          = 1024 # 1 KB
DCACHE_LINE_SIZE     = 32
DCACHE_NUM_LINES     = DCACHE_SIZE // DCACHE_LINE_SIZE
DCACHE_ASSOCIATIVITY = 1

MAIN_MEMORY_SIZE    = 65536  # 64 KB
MAIN_MEMORY_LATENCY = 10

# === Pipeline Parameters ===

NUM_PIPELINE_STAGES      = 5
ENABLE_BRANCH_PREDICTION = True
ENABLE_FORWARDING        = True

# === Instruction Helpers ===

def encode_r_type(rd, rs1, rs2, funct3, funct7, opcode):
    instr = opcode
    instr |= (rd & RD_MASK) << RD_SHIFT
    instr |= (funct3 & FUNCT3_MASK) << FUNCT3_SHIFT
    instr |= (rs1 & RS1_MASK) << RS1_SHIFT
    instr |= (rs2 & RS2_MASK) << RS2_SHIFT
    instr |= (funct7 & FUNCT7_MASK) << FUNCT7_SHIFT
    return instr

def encode_i_type(rd, rs1, imm, funct3, opcode):
    instr = opcode
    instr |= (rd & RD_MASK) << RD_SHIFT
    instr |= (funct3 & FUNCT3_MASK) << FUNCT3_SHIFT
    instr |= (rs1 & RS1_MASK) << RS1_SHIFT
    instr |= (imm & IMM_I_MASK) << IMM_I_SHIFT
    return instr

def encode_s_type(rs1, rs2, imm, funct3, opcode):
    instr = opcode
    instr |= ((imm & IMM_S_LO_MASK)) << IMM_S_LO_SHIFT
    instr |= (funct3 & FUNCT3_MASK) << FUNCT3_SHIFT
    instr |= (rs1 & RS1_MASK) << RS1_SHIFT
    instr |= (rs2 & RS2_MASK) << RS2_SHIFT
    instr |= ((imm >> 5) & IMM_S_HI_MASK) << IMM_S_HI_SHIFT
    return instr

def encode_b_type(rs1, rs2, imm, funct3, opcode):
    instr = opcode
    instr |= ((imm & IMM_B_BIT11_MASK) << IMM_B_BIT11_SHIFT)
    instr |= ((imm >> 1) & IMM_B_BITS4_1_MASK) << IMM_B_BITS4_1_SHIFT
    instr |= (funct3 & FUNCT3_MASK) << FUNCT3_SHIFT
    instr |= (rs1 & RS1_MASK) << RS1_SHIFT
    instr |= (rs2 & RS2_MASK) << RS2_SHIFT
    instr |= ((imm >> 5) & IMM_B_BITS10_5_MASK) << IMM_B_BITS10_5_SHIFT
    instr |= ((imm >> 11) & IMM_B_BIT12_MASK) << IMM_B_BIT12_SHIFT
    return instr

# === Decoding Helpers ===

def extract_opcode(instr):
    return (instr >> OPCODE_SHIFT) & OPCODE_MASK

def extract_rd(instr):
    return (instr >> RD_SHIFT) & RD_MASK

def extract_rs1(instr):
    return (instr >> RS1_SHIFT) & RS1_MASK

def extract_rs2(instr):
    return (instr >> RS2_SHIFT) & RS2_MASK

def extract_funct3(instr):
    return (instr >> FUNCT3_SHIFT) & FUNCT3_MASK

def extract_funct7(instr):
    return (instr >> FUNCT7_SHIFT) & FUNCT7_MASK

def extract_imm_i(instr):
    imm = (instr >> IMM_I_SHIFT) & IMM_I_MASK
    if imm & 0x800:
        imm |= ~0xFFF
    return imm

def sign_extend_12(value):
    if value & 0x800:
        return value | ~0xFFF
    return value

def sign_extend_20(value):
    if value & 0x80000:
        return value | ~0xFFFFF
    return value

# === Instruction Names ===

INSTRUCTION_NAMES = {
    0x00: 'NOP',
    0x13: 'ADDI', 0x33: 'ADD', 0x33 | (0x20 << 25): 'SUB',
    0x33 | (0x07 << 12): 'AND', 0x13 | (0x07 << 12): 'ANDI',
    0x33 | (0x06 << 12): 'OR',  0x13 | (0x06 << 12): 'ORI',
    0x33 | (0x04 << 12): 'XOR', 0x13 | (0x04 << 12): 'XORI',
    0x33 | (0x01 << 12): 'SLL', 0x13 | (0x01 << 12): 'SLLI',
    0x33 | (0x05 << 12): 'SRL', 0x13 | (0x05 << 12): 'SRLI',
    0x33 | (0x05 << 12) | (0x20 << 25): 'SRA',
    0x13 | (0x05 << 12) | (0x20 << 25): 'SRAI',
    0x03 | (0x02 << 12): 'LW',
    0x23 | (0x02 << 12): 'SW',
    0x63 | (0x00 << 12): 'BEQ',
    0x63 | (0x01 << 12): 'BNE',
    0x63 | (0x04 << 12): 'BLT',
    0x63 | (0x05 << 12): 'BGE',
    0x6F: 'JAL',
    0x67: 'JALR',
    0x37: 'LUI',
    0x17: 'AUIPC',
}