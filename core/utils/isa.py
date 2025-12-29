from .constants import *

# === Instruction Classes ===

class Instr:
    pass

class RType(Instr):
    def __init__(self, rd, rs1, rs2, funct3, funct7, opcode):
        self.rd = rd
        self.rs1 = rs1
        self.rs2 = rs2
        self.funct3 = funct3
        self.funct7 = funct7
        self.opcode = opcode

    def encode(self): 
        instr = self.opcode
        instr |= (self.rd & RD_MASK) << RD_SHIFT
        instr |= (self.funct3 & FUNCT3_MASK) << FUNCT3_SHIFT
        instr |= (self.rs1 & RS1_MASK) << RS1_SHIFT
        instr |= (self.rs2 & RS2_MASK) << RS2_SHIFT
        instr |= (self.funct7 & FUNCT7_MASK) << FUNCT7_SHIFT
        return instr & 0xFFFFFFFF
    
class IType(Instr):
    def __init__(self, rd, rs1, imm, funct3, opcode):
        self.rd = rd
        self.rs1 = rs1
        self.imm = imm
        self.funct3 = funct3
        self.opcode = opcode

    def encode(self):
        instr = self.opcode
        instr |= (self.rd & RD_MASK) << RD_SHIFT
        instr |= (self.funct3 & FUNCT3_MASK) << FUNCT3_SHIFT
        instr |= (self.rs1 & RS1_MASK) << RS1_SHIFT
        instr |= (self.imm & IMM_I_MASK) << IMM_I_SHIFT
        return instr & 0xFFFFFFFF

class SType(Instr):
    def __init__(self, rs1, rs2, imm, funct3, opcode):
        self.rs1 = rs1
        self.rs2 = rs2
        self.imm = imm
        self.funct3 = funct3
        self.opcode = opcode

    def encode(self):
        instr = self.opcode
        instr |= ((self.imm & IMM_S_LO_MASK)) << IMM_S_LO_SHIFT
        instr |= (self.funct3 & FUNCT3_MASK) << FUNCT3_SHIFT
        instr |= (self.rs1 & RS1_MASK) << RS1_SHIFT
        instr |= (self.rs2 & RS2_MASK) << RS2_SHIFT
        instr |= ((self.imm >> 5) & IMM_S_HI_MASK) << IMM_S_HI_SHIFT
        return instr & 0xFFFFFFFF

class BType(Instr):
    def __init__(self, rs1, rs2, imm, funct3, opcode):
        self.rs1 = rs1
        self.rs2 = rs2
        self.imm = imm
        self.funct3 = funct3
        self.opcode = opcode

    def encode(self):
        instr = self.opcode
        instr |= ((self.imm & IMM_B_BIT11_MASK) << IMM_B_BIT11_SHIFT)
        instr |= ((self.imm >> 1) & IMM_B_BITS4_1_MASK) << IMM_B_BITS4_1_SHIFT
        instr |= (self.funct3 & FUNCT3_MASK) << FUNCT3_SHIFT
        instr |= (self.rs1 & RS1_MASK) << RS1_SHIFT
        instr |= (self.rs2 & RS2_MASK) << RS2_SHIFT
        instr |= ((self.imm >> 5) & IMM_B_BITS10_5_MASK) << IMM_B_BITS10_5_SHIFT
        instr |= ((self.imm >> 11) & IMM_B_BIT12_MASK) << IMM_B_BIT12_SHIFT
        return instr & 0xFFFFFFFF
    
class UType(Instr):
    def __init__(self, rd, imm, opcode):
        self.rd = rd
        self.imm = imm
        self.opcode = opcode

    def encode(self):
        instr = self.opcode
        instr |= (self.rd & RD_MASK) << RD_SHIFT
        instr |= (self.imm & IMM_U_MASK) << IMM_U_SHIFT
        return instr & 0xFFFFFFFF
    
class JType(Instr):
    def __init__(self, rd, imm, opcode):
        self.rd = rd
        self.imm = imm
        self.opcode = opcode

    def encode(self):
        instr = self.opcode
        instr |= (self.rd & RD_MASK) << RD_SHIFT
        instr |= ((self.imm & IMM_J_BITS19_12_MASK) << IMM_J_BITS19_12_SHIFT)
        instr |= ((self.imm >> 12) & IMM_J_BIT11_MASK) << IMM_J_BIT11_SHIFT
        instr |= ((self.imm >> 11) & IMM_J_BITS10_1_MASK) << IMM_J_BITS10_1_SHIFT
        instr |= ((self.imm >> 21) & IMM_J_BIT20_MASK) << IMM_J_BIT20_SHIFT
        return instr & 0xFFFFFFFF

# === Decoding Helpers ===

def extract_opcode(instr):
    return (instr >> OPCODE_SHIFT) & OPCODE_MASK