# RISC-V Core with Caches - Complete Design Specification

## Overview

This document describes a complete 5-stage pipelined RISC-V core (RV32I) with separate instruction and data caches, designed for simulation and synthesis in PyRTL. The core is optimized for educational purposes and research on cache behavior, featuring straightforward hazard detection, forwarding logic, and direct-mapped caches.

**Key Specifications:**
- **Architecture**: 5-stage pipeline (Fetch → Decode → Execute → Memory → Write-back)
- **ISA**: RISC-V RV32I (32-bit, base integer instruction set)
- **Registers**: 32 × 32-bit general-purpose registers (x0-x31)
- **Instruction Cache**: 1 KB, direct-mapped, 32-byte lines
- **Data Cache**: 1 KB, direct-mapped, 32-byte lines
- **Main Memory**: 64 KB, with configurable access latency

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RISC-V Core with Caches                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                         FETCH STAGE                              │   │
│  │  ┌─────────────┐              ┌──────────┐                       │   │
│  │  │   Program   │◄─────┬──────►│ I-Cache  │                       │   │
│  │  │  Counter    │      │       │ (1KB)    │                       │   │
│  │  │   (PC)      │      │       └──────────┘                       │   │
│  │  └─────────────┘      │                                          │   │
│  │                       │       ┌──────────┐                       │   │
│  │                       └──────►│  Memory  │                       │   │
│  │                               │Controller│                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│           │ [Instruction]                                               │
│           ▼                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      DECODE STAGE                                │   │
│  │  ┌───────────────────────────────────────────────────────────┐   │   │
│  │  │      Instruction Decoder                                  │   │   │
│  │  │  ┌───────┬────────────┬──────────┬─────────────┐          │   │   │
│  │  │  │Opcode │ Register   │ Immediate│  Control    │          │   │   │
│  │  │  │Parser │ Fields     │ Extract  │  Signals    │          │   │   │
│  │  │  └───────┴────────────┴──────────┴─────────────┘          │   │   │
│  │  └───────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│           │ [Opcode, RS1/RS2, RD, Immediate, Ctrl]                      │
│           ▼                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    REGISTER FILE                                 │   │
│  │  ┌──────────────────────────────────────────────────────────┐    │   │
│  │  │                    32 × 32-bit Registers                 │    │   │
│  │  │  x0=0  x1  x2  x3  ... x31                               │    │   │
│  │  │  Read Port 1  ◄─ RS1  (from Decode)                      │    │   │
│  │  │  Read Port 2  ◄─ RS2  (from Decode)                      │    │   │
│  │  │  Write Port   ◄─ WD   (from Write-back)                  │    │   │
│  │  └──────────────────────────────────────────────────────────┘    │   │
│  └───────────────────┬──────────────┬───────────────────────────────┘   │
│                      │ [RD1]   [RD2]│                                   │
│           ┌──────────┘              └──────────┐                        │
│           ▼                                     ▼                       │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    EXECUTE STAGE                                 │   │
│  │  ┌──────────────────────────────────────────────────────────┐    │   │
│  │  │                        ALU                               │    │   │
│  │  │  ┌─────────┬────────┬─────────┬──────┬────────┐          │    │   │
│  │  │  │ Adder   │ Shift  │ Logic   │ Comp │ Mux    │          │    │   │
│  │  │  │ (Add/   │ (SLL,  │ (AND,   │ (LT, │ (Op    │          │    │   │
│  │  │  │  Sub)   │  SRL)  │ OR, XOR)│ SLT) | Select)│          │    │   │
│  │  │  └─────────┴────────┴─────────┴──────┴────────┘          │    │   │
│  │  └──────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│           │ [ALU Result, Branch Taken, Condition Codes]                 │
│           ▼                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    MEMORY STAGE                                  │   │
│  │  ┌────────────────────────────────────────────────────────────┐  │   │
│  │  │              Data Cache (1KB)                              │  │   │
│  │  │  ┌──────────────────────────────────────────────────────┐  │  │   │
│  │  │  │  Cache Controller                                    │  │  │   │
│  │  │  │  ┌────────┬─────────────┬──────────┐                 │  │  │   │
│  │  │  │  │ Lookup │ Hit/Miss    │ Replace  │                 │  │  │   │
│  │  │  │  │ Engine │ Logic       │ Policy   │                 │  │  │   │
│  │  │  │  └────────┴─────────────┴──────────┘                 │  │  │   │
│  │  │  └──────────────────────────────────────────────────────┘  │  │   │
│  │  │                      │ Cache Miss                          │  │   │
│  │  │                      ▼                                     │  │   │
│  │  │          ┌──────────────────────┐                          │  │   │
│  │  │          │  Memory Controller   │                          │  │   │
│  │  │          │  Main Memory (64KB)  │                          │  │   │
│  │  │          └──────────────────────┘                          │  │   │
│  │  └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│           │ [Memory Data or Forward from ALU]                           │
│           ▼                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   WRITE-BACK STAGE                               │   │
│  │  ┌──────────────────────────────────────────────────────────┐    │   │
│  │  │  Multiplexer: Choose result source                       │    │   │
│  │  │  ┌─────────┬──────────┬─────────────┐                    │    │   │
│  │  │  │  ALU    │ Memory   │ Immediate   │                    │    │   │
│  │  │  │ Result  │ Data     │ (for ADDI)  │                    │    │   │
│  │  │  └─────────┴──────────┴─────────────┘                    │    │   │
│  │  └──────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │            FORWARDING LOGIC (Data Hazard Resolution)             │   │
│  │  If (RS1 == RD_previous) forward ALU result to Execute           │   │
│  │  If (RS2 == RD_previous) forward ALU result to Execute           │   │
│  │  Load-use hazard: Stall for 1 cycle                              │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Summary

```
   PC ──► I-Cache ──► Fetch ──► [Instr]
                       │
                       ▼
   Decode ◄────────────┴────────────────┐
   │                                    │
   ├─► Reg File Read ────┐              │
   │        │            │              │
   │        └─► Execute ─┼──────────────┘
   │             │       │
   │             ├────────┤ Forwarding
   │             │       │
   │             ▼       │
   │          Memory ◄───┤ D-Cache ◄─► Memory Controller ◄─► Main Memory
   │             │       │
   │             ▼       │
   │          WriteBack ─┴──► Reg File Write
   │
   └──────────────────────────────────────►  (Next Cycle)
```

---

## Recommended Directory Layout

```
riscv-core/
├── README.md
├── requirements.txt
├── core/
│   ├── __init__.py
│   ├── riscv_core.py              # Main top-level module
│   ├── fetch.py                   # Fetch stage
│   ├── decode.py                  # Decode stage & instruction decoder
│   ├── execute.py                 # Execute stage (ALU, comparisons)
│   ├── memory.py                  # Memory stage (load/store)
│   ├── writeback.py               # Write-back stage
│   └── regfile.py                 # Register file (32 registers for RV32I)
├── cache/
│   ├── __init__.py
│   ├── base.py              # Base cache class
│   ├── icache.py                  # Instruction cache
│   ├── dcache.py                  # Data cache
│   └── controller.py       # Main memory & cache coherence
├── utils/
│   ├── __init__.py
│   ├── constants.py               # RISC-V opcodes, register definitions
│   ├── helpers.py                 # Common utility functions
│   ├── isa.py                     # ISA definitions & instruction formats
│   └── instrumentation.py         # Probes and debugging helpers
├── sim/
│   ├── __init__.py
│   ├── simulator.py               # High-level simulation wrapper
│   ├── testbench.py               # Generic testbench utilities
│   └── program_loader.py          # Load RISC-V programs into memory
├── tests/
│   ├── __init__.py
│   ├── test_alu.py                # Test arithmetic/logic operations
│   ├── test_regfile.py            # Test register file
│   ├── test_decoder.py            # Test instruction decoder
│   ├── test_cache.py              # Test cache behavior
│   ├── test_core.py               # Integration tests
│   └── test_programs/
│       ├── add_test.s             # Simple assembly test programs
│       ├── memory_test.s
│       ├── loop_test.s
│       └── cache_test.s
├── examples/
│   ├── simple_adder.py            # Example: Simple add program
│   ├── fibonacci.py               # Example: Fibonacci sequence
│   └── memory_ops.py              # Example: Memory operations
├── docs/
│   ├── architecture.md            # Detailed architecture documentation
│   ├── instruction_set.md         # Supported RISC-V instructions
│   ├── cache_design.md            # Cache specifications
│   └── pipeline.md                # Pipeline diagram and explanation
└── output/
    ├── design.v                   # Generated Verilog
    └── simulation_trace.vcd       # Waveform file
```

## Core Module Dependencies

```
riscv_core.py (top-level)
    ├── fetch.py
    │   └── icache.py
    ├── decode.py
    ├── execute.py
    ├── memory.py
    │   └── dcache.py
    ├── writeback.py
    ├── regfile.py
    └── control.py (or embedded in decode)

cache/
    ├── icache.py → cache_base.py
    ├── dcache.py → cache_base.py
    └── memory_controller.py (main memory)

utils/
    ├── constants.py (shared)
    ├── helpers.py (shared)
    └── isa.py (shared)
```

## Suggested Implementation Order

1. **Utilities & Constants** (`utils/constants.py`, `utils/isa.py`)
   - Define RISC-V opcodes, registers, instruction formats
   - Keep reusable across all modules

2. **Register File** (`core/regfile.py`)
   - Simple, isolated component
   - Easy to test independently

3. **ALU & Execute** (`core/execute.py`)
   - Core computation unit
   - Test with random values

4. **Instruction Decoder** (`core/decode.py`)
   - Parse RISC-V instruction format
   - Test with known instruction encodings

5. **Caches** (`cache/icache.py`, `cache/dcache.py`, `cache/cache_base.py`)
   - Implement basic cache logic (direct-mapped or 2-way)
   - Test cache hit/miss behavior

6. **Memory Controller** (`cache/memory_controller.py`)
   - Main memory simulation
   - Integrate with caches

7. **Pipeline Stages** (`core/fetch.py`, `core/memory.py`, etc.)
   - Build stages that use previous components
   - Connect with forwarding logic

8. **Top-Level Core** (`core/riscv_core.py`)
   - Integrate all stages
   - Add pipeline control hazard detection

9. **Simulator & Tests** (`sim/`, `tests/`)
   - High-level simulation interface
   - Comprehensive test suite

## Configuration File (Optional)

Create `config.py` in the root:

```python
# Core Configuration
XLEN = 32                    # 32-bit RISC-V
NUM_REGISTERS = 32           # x0-x31

# Pipeline Configuration
NUM_PIPELINE_STAGES = 5      # Fetch, Decode, Execute, Memory, Write-back
ENABLE_BRANCH_PREDICTION = False

# Cache Configuration
ICACHE_SIZE = 1024           # 1 KB
ICACHE_LINE_SIZE = 32        # 32 bytes per line
ICACHE_ASSOCIATIVITY = 1     # Direct-mapped

DCACHE_SIZE = 1024           # 1 KB
DCACHE_LINE_SIZE = 32        # 32 bytes per line
DCACHE_ASSOCIATIVITY = 1     # Direct-mapped

# Memory Configuration
MAIN_MEMORY_SIZE = 65536     # 64 KB
MEMORY_LATENCY = 10          # Cycles to main memory
```

## File Responsibilities

| File | Responsibility |
|------|-----------------|
| `constants.py` | RISC-V opcodes, register names, instruction masks |
| `isa.py` | Instruction format definitions, decoding helpers |
| `regfile.py` | 32 registers, read/write logic |
| `execute.py` | ALU operations (add, sub, and, or, xor, slt, etc.) |
| `decode.py` | Parse instruction bits to opcode/operands |
| `fetch.py` | Instruction address generation, PC management |
| `memory.py` | Load/store logic, address translation (if any) |
| `writeback.py` | Results back to register file |
| `base.py` | Common cache logic (hit/miss, replacement) |
| `icache.py` | Instruction cache with base |
| `dcache.py` | Data cache with base |
| `controller.py` | Main memory, cache coherence |
| `riscv_core.py` | Wire everything together, pipeline control |
| `simulator.py` | User-friendly simulation wrapper |

---

## Cache Architecture Details

### Instruction Cache (I-Cache)

**Specifications:**
- **Size**: 1 KB (1024 bytes)
- **Line Size**: 32 bytes
- **Number of Lines**: 32 lines (1024 / 32)
- **Associativity**: 1-way (direct-mapped)
- **Replacement Policy**: N/A (direct-mapped)

**Cache Entry Structure:**
```
┌─────────────────────────────────┐
│  I-Cache (1 KB, Direct-Mapped)  │
├─────────────────────────────────┤
│  Tag   │ Valid │ Instruction    │
├────────┼───────┼────────────────┤
│  0x00  │   1   │ 0x00000013     │
│  0x01  │   1   │ 0x00110113     │
│  0x02  │   0   │ xxxxxx         │
│  ...   │  ...  │    ...         │
└─────────────────────────────────┘
```

**Address Mapping:**
```
┌────────────────────┬───────────────┐
│      Tag           │    Index      │
│   (31:11) [21b]    │  (10:5) [6b]  │
└────────────────────┴───────────────┘
```

**Access Logic:**
- Index (bits 10:5) selects the line
- Valid bit checked for cache hit
- Tag compared with address tag for verification
- Single-cycle hit latency

**Miss Handling:**
- Cache miss goes to Memory Controller
- Memory Controller fetches 32-byte line from main memory
- Returns instruction to cache and to Execute stage
- Typically 10+ cycles latency

---

### Data Cache (D-Cache)

**Specifications:**
- **Size**: 1 KB (1024 bytes)
- **Line Size**: 32 bytes
- **Number of Lines**: 32 lines
- **Associativity**: 1-way (direct-mapped)
- **Write Policy**: Write-back (can be changed to write-through)
- **Dirty Bit**: Tracks modified lines for write-back

**Cache Entry Structure:**
```
┌───────────────────────────────────────┐
│  D-Cache (1 KB, Direct-Mapped)        │
├───────────────────────────────────────┤
│ Tag  │ Valid │ Dirty │ Data           │
├──────┼───────┼───────┼────────────────┤
│ 0x00 │   1   │   0   │ 0xDEADBEEF     │
│ 0x01 │   1   │   1   │ 0x12345678     │
│ 0x02 │   0   │   0   │ xxxxxxxxxx     │
│ ...  │  ...  │  ...  │    ...         │
└──────┴───────┴───────┴────────────────┘
```

**Address Mapping:**
```
┌─────────────────┬──────────────┐
│      Tag        │   Index      │
│   (31:11)       │ (10:2) [9b]  │
└─────────────────┴──────────────┘
```

**Access Logic:**
- Index (bits 10:2) selects the line
- Valid and Dirty bits checked
- Tag comparison for hit/miss detection
- Write operations update data and set Dirty bit
- Read operations check cache and forward from pipeline if needed

**Write Behavior:**
- **Write-hit**: Update cache line, set Dirty bit
- **Write-miss**: Load line from memory, then write (allocate on write)
- **Flush**: On cache eviction, write-back only if Dirty bit set

---

## Pipeline Stages - Detailed Specification

### Stage 1: Fetch (F)
**Inputs:**
- `branch_target` (32-bit): Jump/branch destination from Execute
- `branch_taken` (1-bit): Control signal for PC update

**Outputs:**
- `instr` (32-bit): Instruction from I-Cache
- `pc` (32-bit): Current program counter

**Logic:**
```
if branch_taken:
    PC ← branch_target
else:
    PC ← PC + 4

instr ← I-Cache[PC]
```

**Register:** `PC` (32-bit register, reset to 0)

---

### Stage 2: Decode (D)
**Inputs:**
- `instr` (32-bit): Instruction from Fetch

**Outputs:**
- `opcode` (7 bits): Instruction opcode
- `rs1` (5 bits): First source register
- `rs2` (5 bits): Second source register
- `rd` (5 bits): Destination register
- `imm` (32 bits): Immediate value (sign-extended)
- `control_signals`: RegWrite, MemRead, MemWrite, ALUOp, etc.

**Instruction Formats Supported:**
```
R-type (Register-Register):
┌─────────┬───────┬───────┬───────┬──────┬───────┐
│ funct7  │ rs2   │ rs1   │funct3 │ rd   │opcode │
│(31:25)  │(24:20)│(19:15)│(14:12)│(11:7)│(6:0)  │
└─────────┴───────┴───────┴───────┴──────┴───────┘

I-type (Immediate):
┌────────────────┬───────┬───────┬──────┬───────┐
│   imm[11:0]    │ rs1   │funct3 │ rd   │opcode │
│(31:20)         │(19:15)│(14:12)│(11:7)│(6:0)  │
└────────────────┴───────┴───────┴──────┴───────┘

S-type (Store):
┌─────────┬───────┬───────┬───────┬──────────┬───────┐
│imm[11:5]│ rs2   │ rs1   │funct3 │imm[4:0]  │opcode │
│(31:25)  │(24:20)│(19:15)│(14:12)│(11:7)    │(6:0)  │
└─────────┴───────┴───────┴───────┴──────────┴───────┘

B-type (Branch):
┌───────┬─────────┬───────┬───────┬───────┬────────┬───────┬──────┐
│imm[12]│imm[10:5]│ rs2   │ rs1   │funct3 │imm[4:1]│imm[11]│opcode│
│(31)   │(30:25)  │(24:20)│(19:15)│(14:12)│(11:8)  │(7)    │(6:0) │
└───────┴─────────┴───────┴───────┴───────┴────────┴───────┴──────┘
```

---

### Stage 3: Execute (E)
**Inputs:**
- `rs1_data` (32-bit): Register 1 value (possibly forwarded)
- `rs2_data` (32-bit): Register 2 value (possibly forwarded)
- `imm` (32-bit): Immediate from Decode
- `opcode` (7 bits): Instruction opcode
- `funct3` (3 bits): ALU operation specifier

**Outputs:**
- `alu_result` (32-bit): ALU computation result
- `branch_target` (32-bit): PC + offset (for branches)
- `branch_taken` (1-bit): Whether branch should be taken
- `rs2_data_for_store` (32-bit): Data to store (for S-type)

**ALU Operations Supported:**
| OpCode | Funct3 | Operation | Example |
|--------|--------|-----------|---------|
| 0010011 | 000 | ADD (I-type) | ADDI x1, x2, 100 |
| 0110011 | 000 | ADD/SUB | ADD x1, x2, x3 |
| 0110011 | 001 | Shift Left Logical | SLL x1, x2, x3 |
| 0110011 | 101 | Shift Right | SRL/SRA x1, x2, x3 |
| 0110011 | 111 | AND | AND x1, x2, x3 |
| 0110011 | 110 | OR | OR x1, x2, x3 |
| 0110011 | 100 | XOR | XOR x1, x2, x3 |
| 0110011 | 010 | Set Less Than | SLT x1, x2, x3 |
| 1100011 | 000 | BEQ (branch) | BEQ x1, x2, offset |

---

### Stage 4: Memory (M)
**Inputs:**
- `alu_result` (32-bit): Address and/or data from Execute
- `mem_write_data` (32-bit): Data to write (from Execute)
- `control_signals`: MemRead, MemWrite

**Outputs:**
- `mem_read_data` (32-bit): Data from D-Cache
- `ready` (1-bit): Cache hit or miss resolution

**Memory Access Logic:**
```
if MemRead:
    if D-Cache hit:
        mem_read_data ← D-Cache[address]
        ready ← 1
    else:
        ready ← 0  (stall pipeline)
        (Memory Controller fetches line)

if MemWrite:
    D-Cache[address] ← mem_write_data
    Set Dirty bit
```

---

### Stage 5: Write-Back (W)
**Inputs:**
- `alu_result` (32-bit): From Execute stage (via Memory stage)
- `mem_read_data` (32-bit): From Memory stage
- `rd` (5 bits): Destination register
- `MemToReg` (1 bit): Select ALU result or memory data
- `RegWrite` (1 bit): Enable register write

**Logic:**
```
if RegWrite and rd != 0:
    if MemToReg:
        RegFile[rd] ← mem_read_data
    else:
        RegFile[rd] ← alu_result
```

---

## Control Signals

### From Decode Stage

| Signal | Width | Purpose | Values |
|--------|-------|---------|--------|
| `RegWrite` | 1 | Enable register file write | 0=no write, 1=write |
| `MemRead` | 1 | Initiate memory read | 0=no read, 1=read |
| `MemWrite` | 1 | Initiate memory write | 0=no write, 1=write |
| `ALUOp[3:0]` | 4 | ALU operation selector | 0000-1111 (see table) |
| `ALUSrc` | 1 | ALU B input source | 0=register, 1=immediate |
| `MemToReg` | 1 | Write-back source | 0=ALU, 1=memory |
| `Branch` | 1 | Branch instruction flag | 0=normal, 1=branch |
| `BranchType` | 2 | Type of branch | 00=BEQ, 01=BNE, 10=BLT, 11=BGE |
| `Jump` | 1 | Jump instruction flag | 0=normal, 1=jump |

**Control Logic Implementation:**

The control ROM (or combinational logic) decodes the opcode and funct3:

```python
def decode_control(opcode, funct3):
    if opcode == 0x13:  # ADDI
        return {
            'RegWrite': 1, 'MemRead': 0, 'MemWrite': 0,
            'ALUOp': 0, 'ALUSrc': 1, 'MemToReg': 0,
            'Branch': 0, 'Jump': 0
        }
    elif opcode == 0x33:  # R-type (ADD, SUB, etc.)
        return {
            'RegWrite': 1, 'MemRead': 0, 'MemWrite': 0,
            'ALUOp': funct3, 'ALUSrc': 0, 'MemToReg': 0,
            'Branch': 0, 'Jump': 0
        }
    elif opcode == 0x03:  # Load
        return {
            'RegWrite': 1, 'MemRead': 1, 'MemWrite': 0,
            'ALUOp': 0, 'ALUSrc': 1, 'MemToReg': 1,
            'Branch': 0, 'Jump': 0
        }
    # ... more cases
```

---

## Hazard Detection and Handling

### Control Hazard (Branches and Jumps)

**Problem:** Branch target is not known until Execute stage, but Fetch stage runs every cycle.

**Solution:** 
- **Flush pipeline**: When branch is taken in Execute stage, flush Fetch and Decode stages
- **PC update**: Set PC to branch target on next cycle
- **Penalty**: 1-cycle stall for taken branches

**Implementation:**
```
In Execute stage:
if Branch and (rs1 == rs2):  // For BEQ
    branch_taken ← 1
    branch_target ← PC + imm
    flush_signal ← 1

In Fetch stage (next cycle):
if flush_signal:
    Discard fetched instruction
    PC ← branch_target
```

---

### Data Hazard - Register Dependencies

**Problem:** A register may be written in one stage while being read in another.

**Example:**
```
ADD x1, x2, x3    (Write-back: x1 ← ALU result)
ADD x4, x1, x5    (Execute: Read x1)
```

Without mitigation, x4 gets wrong value.

**Solution 1: Forwarding (Operand Forwarding)**
- Route Execute result directly back to Execute input
- Applicable when ALU result needed immediately

```
if (Execute.RS1 == WriteBack.RD) and (WriteBack.RegWrite):
    Execute.input_a ← WriteBack.result  // Forward
else:
    Execute.input_a ← RegFile[RS1]      // Normal read
```

**Solution 2: Load-Use Stall**
- Loads return data in Memory stage, but needed in Execute
- One cycle after load, must use loaded value → Stall

```
if (Memory.opcode == LOAD) and (Decode.RS1 == Memory.RD or Decode.RS2 == Memory.RD):
    Stall the Decode stage (hold PC and Fetch/Decode latches)
    Memory stage continues to Memory.next stage
```

**Timeline Example:**
```
Cycle  │ F  │ D  │ E  │ M  │ W
───────┼────┼────┼────┼────┼────
1      │LW  │    │    │    │      (load x1 from memory)
2      │ADD │LW  │    │    │
3      │SUB │ADD │LW  │    │      (stall needed)
3      │    │SUB │(s) │    │      (SUB stalls - x1 not ready)
4      │AND │    │SUB │LW  │      (x1 ready, SUB continues)
5      │    │AND │    │SUB │LW
```

---

### Structural Hazard

**Problem:** Two stages need the same resource simultaneously.

**Mitigation:**
- Separate I-Cache and D-Cache (no conflict)
- Register file has multiple read ports (typically 2) and 1 write port
- Bypassing and staging ensure no direct conflicts

**No significant structural hazards in this design.**

---

## Execution Timeline and Pipeline Behavior

### Ideal Case (No Hazards)

```
Cycle  │ F    │ D    │ E    │ M    │ W
───────┼──────┼──────┼──────┼──────┼──────
1      │ADDI  │      │      │      │
2      │ADD   │ADDI  │      │      │
3      │SUB   │ADD   │ADDI  │      │
4      │OR    │SUB   │ADD   │ADDI  │
5      │XOR   │OR    │SUB   │ADD   │ADDI
6      │      │XOR   │OR    │SUB   │ADD
7      │      │      │XOR   │OR    │SUB
8      │      │      │      │XOR   │OR
9      │      │      │      │      │XOR

Throughput: 1 instruction per cycle (IPC = 1.0)
```

### With Load-Use Hazard

```
Cycle  │ F    │ D    │ E    │ M    │ W
───────┼──────┼──────┼──────┼──────┼──────
1      │LW    │      │      │      │      (Load x1)
2      │ADD   │LW    │      │      │
3      │SUB   │ADD   │LW    │      │      (Hazard: x1 not ready)
4      │(s)   │SUB   │(s)   │      │      (Stall - x1 arrives in M)
4      │      │SUB   │LW    │      │
5      │OR    │      │SUB   │LW    │      (x1 forwarded from Memory)
6      │      │OR    │      │SUB   │LW
7      │      │      │OR    │      │SUB

Penalty: 1 cycle stall
```

### With Branch Misprediction

```
Cycle  │ F    │ D    │ E    │ M    │ W
───────┼──────┼──────┼──────┼──────┼──────
1      │ADDI  │      │      │      │
2      │BEQ   │ADDI  │      │      │
3      │ADD   │BEQ   │ADDI  │      │      (Wrong path)
3      │(x)   │(x)   │(x)   │      │      (Flush on branch)
4      │JUMP_ │      │      │      │      (Correct branch target fetched)
5      │INSTR │JUMP_ │      │      │
6      │      │INSTR │JUMP_ │      │
7      │      │      │INSTR │JUMP_ │

Penalty: 1 cycle flush on branch execution
```

---

## Design Decisions and Rationale

### Why 5 Stages?

**Benefits:**
- Simple hazard detection logic
- Reasonable pipeline depth for teaching
- Easy to implement forwarding logic

**Alternatives:**
- 3 stages: Simpler, but poor performance
- 7+ stages: Better potential performance, but more complex hazard logic and more stalls

---

### Why Direct-Mapped Caches?

**Benefits:**
- Simple index calculation (only lower bits of address)
- Single-cycle hit latency
- Minimal logic for hit/miss detection
- Easy to understand and debug

**Trade-off:**
- More cache misses than associative caches
- 1 KB size acceptable for educational purposes

**Alternative:** Could upgrade to 2-way or 4-way set-associative with additional complexity.

---

### Why Write-Back D-Cache?

**Benefits:**
- Fewer writes to main memory
- Better performance for repeated writes to same address
- Dirty bit tracks which lines need writing back

**Trade-off:**
- More complex logic (dirty bit, write-back on eviction)
- Potential data consistency issues in multi-core (not applicable here)

**Alternative:** Write-through (write to both cache and memory) is simpler but slower.

---

### Why Forwarding Logic?

**Benefits:**
- Resolves most data hazards without stalling
- Keep pipeline full and improving performance
- Most dependencies are resolvable by forwarding

**Limitation:**
- Load-use hazards still require 1-cycle stall (data not available until Memory stage)

---

### Instruction Set Support

**Full RV32I Base Integer ISA includes:**
- Arithmetic: ADD, ADDI, SUB, SLI, SLIU
- Logic: AND, ANDI, OR, ORI, XOR, XORI
- Shift: SLL, SLLI, SRL, SRLI, SRA, SRAI
- Comparison: SLT, SLTI, SLTU, SLTIU
- Memory: LW, LH, LB, SW, SH, SB
- Branch: BEQ, BNE, BLT, BGE, BLTU, BGEU
- Jump: JAL, JALR
- Special: LUI, AUIPC, FENCE, ECALL, EBREAK

**Simplified Implementation** (for this project):
- Core: ADD, ADDI, SUB, AND, ANDI, OR, ORI, XOR, XORI
- Memory: LW, SW
- Branch: BEQ, BNE, BLT, BGE
- Jump: JAL, JALR (optional)

---

## Implementation Checklist

### Phase 1: Utilities & Constants

- [x] Implement utils/constants.py (opcodes, register names)
  - [x] Define RISC-V opcodes for all supported instructions
  - [x] Define register names (x0-x31) and ABI names
  - [x] Define instruction field masks (opcode, funct3, funct7)
  - [x] Define control signal constants

- [ ] Implement utils/isa.py (instruction format helpers)
  - [ ] Create instruction format classes (R-type, I-type, S-type, B-type, U-type, J-type)
  - [ ] Implement sign extension helpers
  - [ ] Create instruction encoder/decoder functions
  - [ ] Add instruction validation and verification helpers

### Phase 2: Core Components

- [ ] Implement core/regfile.py (32 registers, read/write)
  - [ ] Create 32 × 32-bit register array
  - [ ] Implement dual read ports with RS1/RS2 inputs
  - [ ] Implement write port with RD/write_data/write_enable
  - [ ] Ensure x0 (zero register) always reads 0 and ignores writes
  - [ ] Write unit tests for all read/write scenarios

- [ ] Implement core/execute.py (ALU operations)
  - [ ] Implement arithmetic operations (ADD, ADDI, SUB)
  - [ ] Implement logic operations (AND, ANDI, OR, ORI, XOR, XORI)
  - [ ] Implement shift operations (SLL, SRL, SRA)
  - [ ] Implement comparison operations (SLT, SLTU)
  - [ ] Create ALU operation multiplexer based on ALUOp signal
  - [ ] Write unit tests with known input/output pairs

- [ ] Implement core/decode.py (instruction decoder)
  - [ ] Implement opcode extraction and parsing
  - [ ] Implement register field extraction (RS1, RS2, RD)
  - [ ] Implement immediate extraction and sign extension (I/S/B/U/J types)
  - [ ] Create control signal ROM mapping opcodes to control signals
  - [ ] Write tests for each instruction format and opcode

### Phase 3: Cache Subsystem

- [ ] Implement cache/base.py (cache base class)
  - [ ] Define cache entry structure with tag, valid, data
  - [ ] Implement index calculation from address
  - [ ] Implement tag comparison logic
  - [ ] Create hit/miss detection logic
  - [ ] Implement cache initialization and reset

- [ ] Implement cache/icache.py (instruction cache)
  - [ ] Inherit from base cache class
  - [ ] Configure for 1 KB, direct-mapped, 32-byte lines
  - [ ] Implement instruction fetch logic
  - [ ] Add cache statistics (hits/misses/accesses)
  - [ ] Write tests for hit/miss scenarios

- [ ] Implement cache/dcache.py (data cache)
  - [ ] Inherit from base cache class
  - [ ] Configure for 1 KB, direct-mapped, 32-byte lines
  - [ ] Add dirty bit tracking for write-back policy
  - [ ] Implement read/write port logic
  - [ ] Handle write-hit and write-miss behavior
  - [ ] Write tests for read/write operations and dirty bit handling

- [ ] Implement cache/controller.py (memory controller)
  - [ ] Create main memory storage (64 KB)
  - [ ] Implement I-Cache miss handling
  - [ ] Implement D-Cache miss handling with line filling
  - [ ] Implement write-back handling for dirty lines
  - [ ] Configure memory access latency (10 cycles)
  - [ ] Add memory initialization from external programs

### Phase 4: Pipeline Stages

- [ ] Implement core/fetch.py (PC management)
  - [ ] Create program counter register (32-bit, reset to 0)
  - [ ] Implement I-Cache interface
  - [ ] Implement branch target multiplexer (sequential vs branch)
  - [ ] Implement stall signal handling
  - [ ] Write tests for PC increment and branch behavior

- [ ] Implement core/memory.py (memory stage)
  - [ ] Implement D-Cache interface
  - [ ] Create address calculation from ALU result
  - [ ] Implement read/write control logic
  - [ ] Handle cache hit/miss stalls
  - [ ] Write tests for load/store operations and stalls

- [ ] Implement core/writeback.py (write-back stage)
  - [ ] Create multiplexer for result selection (ALU vs memory)
  - [ ] Implement register file write control
  - [ ] Ensure x0 cannot be written
  - [ ] Write tests for both ALU and memory write-back paths

### Phase 5: Pipeline Integration

- [ ] Implement core/riscv_core.py (top-level integration)
  - [ ] Wire all pipeline stages together
  - [ ] Implement pipeline registers (F/D, D/E, E/M, M/W latches)
  - [ ] Implement forwarding logic (ALU result to Execute inputs)
  - [ ] Implement load-use hazard detection and stalling
  - [ ] Implement branch flush logic
  - [ ] Add pipeline visualization/debugging outputs

### Phase 6: Simulation & Testing

- [ ] Implement sim/simulator.py (simulation wrapper)
  - [ ] Create high-level simulation interface
  - [ ] Implement program loading into memory
  - [ ] Implement step/run functionality
  - [ ] Add register/memory inspection methods
  - [ ] Implement trace collection and waveform export

- [ ] Write comprehensive tests in tests/
  - [ ] test_alu.py: Test all ALU operations with edge cases
  - [ ] test_regfile.py: Test register read/write, zero register
  - [ ] test_decoder.py: Test all instruction formats and opcodes
  - [ ] test_cache.py: Test cache hits/misses/write-back
  - [ ] test_core.py: Integration tests with simple programs
  - [ ] Create test programs for common patterns

- [ ] Create example programs in examples/
  - [ ] simple_adder.py: Add two numbers and store result
  - [ ] fibonacci.py: Compute Fibonacci sequence with loops
  - [ ] memory_ops.py: Demonstrate load/store patterns
  - [ ] branch_test.py: Test branch prediction and flushing
  - [ ] cache_patterns.py: Programs to exercise cache behavior

### Phase 7: Validation & Export

- [ ] Export to Verilog
  - [ ] Export complete design with pyrtl.output_to_verilog()
  - [ ] Verify timing analysis with critical path
  - [ ] Generate area estimation

- [ ] Documentation & Final Testing
  - [ ] Create README with setup and usage instructions
  - [ ] Document all supported instructions with examples
  - [ ] Add performance analysis tools
  - [ ] Run full integration test suite



