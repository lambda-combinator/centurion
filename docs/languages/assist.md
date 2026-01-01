# AI Assistant Guide for Centurion Project

## Purpose of This Document

This guide helps AI coding assistants provide consistent, high-quality support throughout the Centurion project. When assisting with this lab, you should read:
1. **This file (assist.md)** - For assistance methodology and patterns
2. **[summary.md](../summary.md)** - For project overview and architecture
3. **Specific section doc** (e.g., 2. bringup.md) - For detailed requirements

Together, these documents provide everything needed to guide the user through building a complete computer system from RTL to browser.

---

## Core Principles for AI Assistants

### 1. Educational Focus
- **Explain the "why"**: Don't just give code; explain the underlying concepts
- **Build incrementally**: Start simple, add complexity gradually
- **Show connections**: Relate current work to previous sections and future sections
- **Encourage experimentation**: Suggest variations or extensions

### 2. Code Skeleton Philosophy
- **Structure, not implementation**: Provide the framework; let the user fill in logic
- **Comprehensive comments**: Explain what each section should do
- **Clear TODOs**: Mark exactly what the user needs to implement
- **Working by default**: Skeletons should compile/run (even if not fully functional)

### 3. Consistency Standards
- Follow the coding conventions outlined below religiously
- Use the same patterns across similar components
- Reference existing code when creating new components
- Maintain the established project structure

---

## Project Structure Deep Dive

```
centurion/
├── rtl/                          # Hardware (SystemVerilog)
│   ├── core/                     # CPU, MMU, caches
│   ├── peripherals/              # UART, Ethernet, SD, etc.
│   └── soc/                      # Top-level integration
├── tb/                           # Testbenches (cocotb/Python)
├── sim/                          # Simulation scripts
├── synth/                        # Synthesis scripts
├── sw/                           # Software
│   ├── bootrom/                  # Initial boot code (assembly)
│   ├── bootloader/               # Loads kernel (C)
│   ├── libc/                     # Standard library (C)
│   ├── kernel/                   # OS kernel (C)
│   └── user/                     # User programs (C)
├── tools/                        # Development tools
│   ├── assembler/                # Python RISC-V assembler
│   ├── compiler/                 # Haskell C compiler
│   └── linker/                   # Python ELF linker
└── docs/                         # Documentation
    ├── languages/                # Language-specific guides
    └── [section docs]            # Per-section instructions
```

### File Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| SystemVerilog module | `snake_case.sv` | `uart_tx.sv`, `register_file.sv` |
| Python tool | `snake_case.py` | `riscv_asm.py`, `elf_parser.py` |
| Haskell module | `PascalCase.hs` | `CodeGen.hs`, `Parser.hs` |
| C source | `snake_case.c/h` | `syscall.c`, `tcp.c` |
| Assembly | `snake_case.s` | `bootrom.s`, `crt0.s` |
| Test | `test_*.py` | `test_uart.py`, `test_alu.py` |

---

## SystemVerilog Code Skeletons

### Module Header Template

Every SystemVerilog module should start with:

```systemverilog
/**
 * Module: <module_name>
 * 
 * Description:
 *   <Brief description of what this module does>
 *
 * Parameters:
 *   <PARAM_NAME> - <description>
 *
 * Inputs:
 *   clk   - Clock signal
 *   rst_n - Active-low asynchronous reset
 *   <other inputs>
 *
 * Outputs:
 *   <outputs and their descriptions>
 *
 * Notes:
 *   - <Any important implementation details>
 *   - <Timing requirements, constraints, etc.>
 */

module <module_name> #(
    parameter PARAM_NAME = default_value
) (
    input  logic        clk,
    input  logic        rst_n,
    // ... other ports
    output logic [7:0]  data_out
);
```

### Common Patterns

#### 1. Simple State Machine Template

```systemverilog
typedef enum logic [1:0] {
    STATE_IDLE   = 2'b00,
    STATE_ACTIVE = 2'b01,
    STATE_DONE   = 2'b10
} state_t;

state_t state, state_next;

// State register
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= STATE_IDLE;
    end else begin
        state <= state_next;
    end
end

// Next state logic
always_comb begin
    state_next = state;  // Default: stay in current state
    
    case (state)
        STATE_IDLE: begin
            if (condition) begin
                state_next = STATE_ACTIVE;
            end
        end
        
        STATE_ACTIVE: begin
            // TODO: Implement state logic
            if (done_condition) begin
                state_next = STATE_DONE;
            end
        end
        
        STATE_DONE: begin
            state_next = STATE_IDLE;
        end
        
        default: state_next = STATE_IDLE;
    endcase
end
```

#### 2. Counter Pattern

```systemverilog
logic [$clog2(MAX_COUNT)-1:0] counter;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        counter <= '0;
    end else if (enable) begin
        if (counter == MAX_COUNT - 1) begin
            counter <= '0;
        end else begin
            counter <= counter + 1;
        end
    end
end
```

#### 3. Shift Register Pattern

```systemverilog
logic [WIDTH-1:0] shift_reg;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        shift_reg <= '0;
    end else if (shift_enable) begin
        shift_reg <= {shift_reg[WIDTH-2:0], shift_in};
    end
end
```

#### 4. MMIO Register Interface Pattern

```systemverilog
// Memory-mapped registers
logic [31:0] control_reg;
logic [31:0] status_reg;
logic [31:0] data_reg;

// Write logic
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        control_reg <= '0;
        data_reg <= '0;
    end else if (write_enable) begin
        case (write_addr[3:0])
            4'h0: control_reg <= write_data;
            4'h4: data_reg <= write_data;
            // Add more registers as needed
        endcase
    end
end

// Read logic (combinational)
always_comb begin
    read_data = 32'h0;
    if (read_enable) begin
        case (read_addr[3:0])
            4'h0: read_data = control_reg;
            4'h4: read_data = data_reg;
            4'h8: read_data = status_reg;
            default: read_data = 32'hDEADBEEF;
        endcase
    end
end
```

### Creating a SystemVerilog Skeleton

When the user asks for a SystemVerilog module skeleton:

1. **Start with the header**: Include module description, parameters, ports
2. **Declare internal signals**: Group by purpose (state, counters, registers, etc.)
3. **Add sequential blocks**: `always_ff` for registers, `always_comb` for combinational
4. **Include reset logic**: Always use active-low async reset (`rst_n`)
5. **Add TODOs**: Mark where user needs to implement logic
6. **Provide assertions**: Add `assert` statements for invariants (commented out initially)

**Example request**: "Create a skeleton for a UART transmitter"

**Example response**:

```systemverilog
/**
 * Module: uart_tx
 * 
 * Description:
 *   UART transmitter with configurable baud rate.
 *   Transmits 8-bit data with 1 start bit, 8 data bits, 1 stop bit.
 *
 * Parameters:
 *   CLK_FREQ   - Input clock frequency in Hz
 *   BAUD_RATE  - Target baud rate (e.g., 115200)
 *
 * Inputs:
 *   clk        - Clock signal
 *   rst_n      - Active-low asynchronous reset
 *   tx_valid   - Assert high with valid data to transmit
 *   tx_data    - 8-bit data to transmit
 *
 * Outputs:
 *   tx_ready   - High when ready to accept new data
 *   tx_pin     - Serial output (connects to external RX)
 */

module uart_tx #(
    parameter CLK_FREQ  = 50_000_000,
    parameter BAUD_RATE = 115200
) (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       tx_valid,
    input  logic [7:0] tx_data,
    output logic       tx_ready,
    output logic       tx_pin
);

    // Calculate clocks per bit
    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    localparam COUNTER_WIDTH = $clog2(CLKS_PER_BIT);
    
    // State machine
    typedef enum logic [1:0] {
        IDLE  = 2'b00,
        START = 2'b01,
        DATA  = 2'b10,
        STOP  = 2'b11
    } state_t;
    
    state_t state, state_next;
    
    // Internal registers
    logic [COUNTER_WIDTH-1:0] clk_counter;
    logic [2:0]               bit_counter;
    logic [7:0]               tx_shift_reg;
    
    // State register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= state_next;
        end
    end
    
    // Clock counter (for baud rate timing)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            clk_counter <= '0;
        end else begin
            // TODO: Implement counter logic
            //   - Reset when reaching CLKS_PER_BIT-1
            //   - Increment when in active transmission
        end
    end
    
    // Bit counter (tracks which data bit we're sending)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bit_counter <= '0;
        end else begin
            // TODO: Implement bit counter
            //   - Reset in IDLE/START states
            //   - Increment when completing each bit
        end
    end
    
    // Shift register (holds data being transmitted)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_shift_reg <= '0;
        end else begin
            // TODO: Load tx_data when starting transmission
            // TODO: Shift right as bits are sent
        end
    end
    
    // Next state logic
    always_comb begin
        state_next = state;
        
        case (state)
            IDLE: begin
                if (tx_valid) begin
                    state_next = START;
                end
            end
            
            START: begin
                // TODO: Transition to DATA after one bit period
            end
            
            DATA: begin
                // TODO: Transition to STOP after 8 bits sent
            end
            
            STOP: begin
                // TODO: Transition to IDLE after stop bit
            end
            
            default: state_next = IDLE;
        endcase
    end
    
    // Output logic
    always_comb begin
        // TODO: Implement tx_pin output
        //   IDLE/STOP: high (idle state)
        //   START: low
        //   DATA: current bit from shift register
        
        tx_pin = 1'b1;  // Default to idle
    end
    
    assign tx_ready = (state == IDLE);
    
    // Assertions for debugging (uncomment after implementation)
    // assert property (@(posedge clk) disable iff (!rst_n)
    //     (state == DATA) |-> (bit_counter < 8)
    // );

endmodule
```

---

## Python Code Skeletons

### Tool Structure Pattern

Python tools (assembler, linker) should follow this structure:

```python
#!/usr/bin/env python3
"""
Module: <tool_name>

Description:
    <What this tool does>

Usage:
    python <tool_name>.py <args>

Example:
    python riscv_asm.py input.s -o output.bin
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Constants
CONSTANT_NAME = value

class ToolClass:
    """Main class for <tool_name>."""
    
    def __init__(self):
        """Initialize the tool."""
        # TODO: Initialize state
        pass
    
    def process(self, input_file: Path) -> None:
        """
        Process the input file.
        
        Args:
            input_file: Path to input file
            
        Raises:
            ValueError: If input is invalid
        """
        # TODO: Implement processing logic
        pass

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='<Tool description>'
    )
    parser.add_argument('input', type=Path, help='Input file')
    parser.add_argument('-o', '--output', type=Path, 
                       help='Output file (default: stdout)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # TODO: Implement main logic
    try:
        tool = ToolClass()
        tool.process(args.input)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
```

### Assembler-Specific Patterns

#### Instruction Encoding Helper

```python
def encode_r_type(opcode: int, rd: int, funct3: int, 
                  rs1: int, rs2: int, funct7: int) -> int:
    """
    Encode R-type instruction.
    
    Format: funct7[31:25] | rs2[24:20] | rs1[19:15] | 
            funct3[14:12] | rd[11:7] | opcode[6:0]
    
    Args:
        opcode: 7-bit opcode
        rd: Destination register (0-31)
        funct3: 3-bit function code
        rs1: Source register 1 (0-31)
        rs2: Source register 2 (0-31)
        funct7: 7-bit function code
        
    Returns:
        32-bit encoded instruction
    """
    # TODO: Implement bit packing
    # Validate inputs are in range
    # Pack into 32-bit instruction
    pass
```

#### Label Resolution Pattern

```python
class Assembler:
    def __init__(self):
        self.labels: Dict[str, int] = {}
        self.current_address = 0
    
    def first_pass(self, lines: List[str]) -> None:
        """
        First pass: collect labels and their addresses.
        
        Args:
            lines: Assembly source lines
        """
        for line in lines:
            # TODO: Parse line
            # If it's a label, record address
            # If it's an instruction, increment address
            pass
    
    def second_pass(self, lines: List[str]) -> List[int]:
        """
        Second pass: encode instructions, resolve labels.
        
        Args:
            lines: Assembly source lines
            
        Returns:
            List of encoded instructions
        """
        instructions = []
        # TODO: For each line, encode instruction
        # Use self.labels to resolve branch/jump targets
        return instructions
```

---

## Haskell Compiler Skeletons

### Module Structure

```haskell
{-|
Module      : <ModuleName>
Description : <Brief description>
Copyright   : (c) <Year>
License     : MIT
Maintainer  : <email>

<Longer description if needed>
-}

module ModuleName
    ( -- * Types
      TypeName(..)
    , OtherType
      -- * Functions
    , mainFunction
    , helperFunction
    ) where

import Control.Monad (when)
import Data.Maybe (fromMaybe)

-- | Main data type
data TypeName = Constructor1 Type
              | Constructor2 Type Type
              deriving (Show, Eq)

-- | Main function
mainFunction :: Input -> Output
mainFunction input = undefined
    -- TODO: Implement function logic
```

### Parser Combinator Pattern

```haskell
-- | Parse an expression
parseExpr :: Parser Expr
parseExpr = parseTerm `chainl1` addOp
  where
    addOp = do
        op <- oneOf "+-"
        return $ \l r -> BinOp (if op == '+' then Add else Sub) l r

-- | Parse a term
parseTerm :: Parser Expr
parseTerm = parseFactor `chainl1` mulOp
  where
    mulOp = do
        op <- oneOf "*/"
        return $ \l r -> BinOp (if op == '*' then Mul else Div) l r

-- | Parse a factor (number or parenthesized expression)
parseFactor :: Parser Expr
parseFactor = parseNumber <|> parseParens

-- TODO: Implement parseNumber and parseParens
```

### Code Generation Pattern

```haskell
-- | Generate RISC-V assembly for an expression
genExpr :: Expr -> CodeGen [Instruction]
genExpr expr = case expr of
    IntLit n -> do
        -- TODO: Load immediate into register
        reg <- allocReg
        return [Li reg n]
    
    BinOp op e1 e2 -> do
        -- TODO: Generate code for operands
        -- TODO: Apply operation
        code1 <- genExpr e1
        code2 <- genExpr e2
        -- ...
        return $ code1 ++ code2 ++ [opInstr]
    
    Var name -> do
        -- TODO: Load variable from stack/register
        undefined
```

---

## C Code Skeletons

### Kernel Module Pattern

```c
/**
 * @file <filename>.c
 * @brief <Brief description>
 * 
 * <Longer description>
 */

#include "kernel.h"
#include "types.h"

/* Constants */
#define CONSTANT_NAME  value

/* Internal state */
static struct {
    // TODO: Define internal state
} module_state;

/**
 * @brief Initialize the module
 * 
 * @return 0 on success, negative on error
 */
int module_init(void) {
    // TODO: Initialize internal state
    return 0;
}

/**
 * @brief Main module function
 * 
 * @param arg Description of argument
 * @return Description of return value
 */
int module_function(int arg) {
    // TODO: Implement function logic
    return -1;  // Not implemented
}
```

### System Call Pattern

```c
/**
 * System call: open
 * 
 * Opens a file and returns a file descriptor.
 * 
 * Args:
 *   a0: pathname (const char *)
 *   a1: flags (int)
 *   a2: mode (int)
 * 
 * Returns:
 *   a0: file descriptor on success, negative errno on error
 */
long sys_open(const char *pathname, int flags, int mode) {
    struct file *file;
    int fd;
    
    // TODO: Validate pathname
    if (!validate_user_ptr(pathname)) {
        return -EFAULT;
    }
    
    // TODO: Allocate file descriptor
    fd = alloc_fd();
    if (fd < 0) {
        return -EMFILE;
    }
    
    // TODO: Open file
    // TODO: Store in fd table
    
    return fd;
}
```

### Network Stack Pattern

```c
/**
 * Process incoming TCP packet
 */
void tcp_recv(struct netif *netif, struct ip_hdr *ip_hdr, 
              struct tcp_hdr *tcp_hdr, void *data, size_t len) {
    struct tcp_conn *conn;
    
    // TODO: Find matching connection
    conn = tcp_find_conn(ip_hdr->src_addr, tcp_hdr->src_port,
                         ip_hdr->dst_addr, tcp_hdr->dst_port);
    
    if (!conn) {
        // TODO: Handle new connection (SYN)
        if (tcp_hdr->flags & TCP_SYN) {
            conn = tcp_accept(netif, ip_hdr, tcp_hdr);
        } else {
            // TODO: Send RST
            return;
        }
    }
    
    // TODO: Process based on connection state
    switch (conn->state) {
    case TCP_LISTEN:
        // TODO: Handle SYN
        break;
    case TCP_SYN_SENT:
        // TODO: Handle SYN+ACK
        break;
    case TCP_ESTABLISHED:
        // TODO: Handle data
        break;
    // ... more states
    }
}
```

---

## Testing Patterns (cocotb)

### Test Structure

```python
"""
Test suite for <module_name>

Tests:
    - test_basic: Basic functionality
    - test_edge_cases: Edge cases and error conditions
    - test_timing: Timing and performance
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
from cocotb.types import LogicArray

async def reset_dut(dut):
    """Apply reset to DUT."""
    dut.rst_n.value = 0
    await Timer(100, units='ns')
    dut.rst_n.value = 1
    await Timer(10, units='ns')

@cocotb.test()
async def test_basic(dut):
    """Test basic functionality."""
    # Start clock
    clock = Clock(dut.clk, 10, units='ns')  # 100 MHz
    cocotb.start_soon(clock.start())
    
    # Reset
    await reset_dut(dut)
    
    # TODO: Apply test vectors
    dut.input_signal.value = 0x42
    await RisingEdge(dut.clk)
    
    # TODO: Check outputs
    assert dut.output_signal.value == expected_value, \
        f"Expected {expected_value}, got {dut.output_signal.value}"
    
    dut._log.info("Test passed!")

@cocotb.test()
async def test_edge_cases(dut):
    """Test edge cases."""
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    
    # TODO: Test boundary conditions
    # TODO: Test error conditions
    pass
```

### UART Test Helper

```python
async def uart_send_byte(dut, byte_val, baud_period_ns):
    """Send a byte over UART."""
    # Start bit
    dut.rx_pin.value = 0
    await Timer(baud_period_ns, units='ns')
    
    # Data bits (LSB first)
    for i in range(8):
        bit = (byte_val >> i) & 1
        dut.rx_pin.value = bit
        await Timer(baud_period_ns, units='ns')
    
    # Stop bit
    dut.rx_pin.value = 1
    await Timer(baud_period_ns, units='ns')

async def uart_recv_byte(dut, baud_period_ns):
    """Receive a byte from UART."""
    # Wait for start bit
    while dut.tx_pin.value == 1:
        await RisingEdge(dut.clk)
    
    # Wait half bit period to sample in middle
    await Timer(baud_period_ns // 2, units='ns')
    
    # Sample data bits
    byte_val = 0
    for i in range(8):
        await Timer(baud_period_ns, units='ns')
        if dut.tx_pin.value == 1:
            byte_val |= (1 << i)
    
    # Wait for stop bit
    await Timer(baud_period_ns, units='ns')
    
    return byte_val
```

---

## Section-Specific Guidance

### Section 2: Bringup (UART, LED)

**Key Points:**
- First exposure to SystemVerilog - explain syntax carefully
- Emphasize timing (baud rate calculations, clock dividers)
- Show waveforms for serial protocols
- Test incrementally (TX before RX)

**Skeleton Emphasis:**
- Clear state machine diagrams
- Bit timing comments
- MMIO register maps in comments

**Example Guidance:**
```
When creating the UART transmitter:
1. Calculate CLKS_PER_BIT from clock frequency and baud rate
2. Use a counter to track bit timing
3. Use a shift register to hold data bits
4. State machine: IDLE → START → DATA → STOP → IDLE
5. tx_pin is: high when idle, low for start, data bits LSB-first, high for stop
```

### Section 3: Processor (RISC-V CPU)

**Key Points:**
- Pipeline hazards are critical - explain with diagrams
- Register file has 2 read ports, 1 write port
- ALU operations must cover all RISC-V operations
- Branch resolution in EX stage (2-cycle penalty)

**Skeleton Emphasis:**
- Clear pipeline stage boundaries
- Forwarding path comments
- Stall condition explanations

**Example Guidance:**
```
When creating the hazard unit:
1. Check for RAW hazards: ID/EX uses register that EX/MEM or MEM/WB writes
2. Forward from EX/MEM if available (1-cycle old result)
3. Forward from MEM/WB if available (2-cycle old result)
4. Stall for load-use: ID/EX is load and EX/MEM needs its result
5. Flush on branch: clear IF/ID and ID/EX if branch taken
```

### Section 4: Compiler

**Key Points:**
- Recursive descent parsing
- Symbol table for variables
- Register allocation (simple: unlimited virtual regs, then map to physical)
- Stack frame layout: saved regs, locals, args

**Skeleton Emphasis:**
- AST node types with comments
- Parser structure (one function per grammar rule)
- Code generation templates

**Example Guidance:**
```
When generating code for function calls:
1. Evaluate arguments right-to-left
2. First 8 args go in a0-a7, rest on stack
3. Save caller-saved registers if live
4. jal ra, function_name
5. Result is in a0
6. Restore saved registers
7. Adjust stack if args were pushed
```

### Section 5: Operating System (MMU, Kernel)

**Key Points:**
- Virtual memory translation (2-level page table)
- TLB is critical for performance
- Process context switching
- System call interface (trap from user to kernel)

**Skeleton Emphasis:**
- Memory layout diagrams in comments
- Page table entry format
- Process state structure

**Example Guidance:**
```
When implementing page table walking:
1. Extract VPN[1] (bits 31:22) and VPN[0] (bits 21:12)
2. Read root page table at satp.PPN
3. Index with VPN[1] to get PTE
4. Check PTE.V (valid bit)
5. If PTE.R|W|X == 0, it's a pointer to next level
6. Read second-level table at PTE.PPN
7. Index with VPN[0] to get final PTE
8. Check permissions (R/W/X, U/S)
9. Physical address = PTE.PPN | offset (bits 11:0)
```

### Section 6: Networking (TCP/IP)

**Key Points:**
- Layered protocol stack
- TCP state machine (11 states)
- Sequence numbers and acknowledgments
- Socket abstraction

**Skeleton Emphasis:**
- Packet header structures
- State transition comments
- Buffer management

**Example Guidance:**
```
When implementing TCP connect:
1. Allocate connection structure
2. Initialize: seq = random(), state = SYN_SENT
3. Send SYN packet with seq number
4. Wait for SYN+ACK
5. Verify ack == seq + 1
6. Update: seq = ack, ack = recv_seq + 1
7. Send ACK
8. Transition to ESTABLISHED
9. Return socket fd to user
```

### Section 7: Physical Hardware

**Key Points:**
- JTAG state machine (16 states)
- Bitstream format (vendor-specific)
- Power sequencing
- Signal integrity

**Skeleton Emphasis:**
- JTAG timing diagrams
- Boundary scan operations
- Error handling

---

## Memory Maps and Constants

Provide these references when relevant:

### MMIO Map

```
0xF000_0000: UART Base
    +0x00: TXDATA (W) - Write byte to transmit
    +0x04: RXDATA (R) - Read received byte
    +0x08: STATUS (R) - [1]=tx_ready, [0]=rx_valid
    +0x0C: CTRL (RW) - [1]=tx_enable, [0]=rx_enable

0xF000_1000: Ethernet Base
    +0x00: TX_DATA (W) - Write frame data
    +0x04: TX_CTRL (W) - [0]=start_tx
    +0x08: RX_DATA (R) - Read frame data
    +0x0C: RX_STATUS (R) - [0]=frame_ready

0xF000_2000: SD Card Base
    +0x00: CMD (W) - Command register
    +0x04: ARG (W) - Argument register
    +0x08: RESP (R) - Response register
    +0x0C: DATA (RW) - Data buffer
```

### RISC-V Register ABI

```
x0  = zero (hardwired 0)
x1  = ra   (return address)
x2  = sp   (stack pointer)
x3  = gp   (global pointer)
x4  = tp   (thread pointer)
x5  = t0   (temporary)
x6  = t1
x7  = t2
x8  = s0/fp (saved / frame pointer)
x9  = s1   (saved)
x10 = a0   (arg 0 / return value)
x11 = a1   (arg 1)
x12 = a2   (arg 2)
...
x17 = a7   (arg 7)
x18 = s2   (saved)
...
x27 = s11
x28 = t3   (temporary)
...
x31 = t6
```

### System Call Numbers

```
SYS_exit     = 93
SYS_read     = 63
SYS_write    = 64
SYS_open     = 1024
SYS_close    = 57
SYS_fork     = 220
SYS_execve   = 221
SYS_wait     = 260
```

---

## Common Request Patterns

### "Create a skeleton for [component]"

**Response pattern:**
1. Ask clarifying questions if needed (parameters, interfaces)
2. Provide complete file with:
   - Header comment with description
   - All required sections (ports, signals, logic blocks)
   - Comprehensive inline comments
   - Clear TODO markers
   - Example usage in comments
3. Explain the structure: "This skeleton has 3 main parts: ..."
4. Highlight what the user needs to fill in
5. Suggest testing approach

### "How do I implement [feature]?"

**Response pattern:**
1. Reference relevant section docs and summary.md
2. Break down into steps
3. Provide pseudocode or algorithm
4. Show relevant existing code as example
5. Mention common pitfalls
6. Suggest incremental testing

### "My [component] isn't working"

**Response pattern:**
1. Ask for symptoms (simulation output, waveforms, error messages)
2. Suggest debugging steps:
   - Add debug prints/signals
   - Check one stage at a time
   - Verify assumptions with assertions
3. Review critical sections (state machines, edge cases, timing)
4. Offer to review their code if shared

### "Explain [concept]"

**Response pattern:**
1. Start with high-level intuition
2. Provide concrete example
3. Show how it fits in the overall system
4. Draw ASCII diagrams if helpful
5. Relate to previous sections or future sections
6. Suggest resources for deeper understanding

---

## Code Style Guide

### SystemVerilog

```systemverilog
// Constants: UPPER_CASE
parameter CLK_FREQ = 50_000_000;
localparam ADDR_WIDTH = 32;

// Signals: snake_case
logic [7:0] data_out;
logic write_enable;

// Modules: snake_case
module uart_transmitter (...);

// Always blocks: clear separation
always_ff @(posedge clk or negedge rst_n) begin
    // Sequential logic
end

always_comb begin
    // Combinational logic
end

// Indentation: 4 spaces
// Comments: explain WHY, not WHAT
// Keep lines under 80 characters when possible
```

### Python

```python
# Follow PEP 8
# Functions: snake_case
def encode_instruction(opcode, rd, rs1):
    pass

# Classes: PascalCase
class RiscvAssembler:
    pass

# Constants: UPPER_CASE
MAX_REGISTERS = 32

# Type hints everywhere
def parse_line(line: str) -> Optional[Instruction]:
    pass

# Docstrings for all public functions
def main() -> None:
    """Entry point for assembler."""
    pass
```

### C

```c
// Follow Linux kernel style
// Functions: snake_case
int sys_open(const char *pathname, int flags);

// Types: snake_case with _t suffix
typedef struct tcp_conn tcp_conn_t;

// Structs: snake_case
struct process {
    int pid;
    // ...
};

// Macros: UPPER_CASE
#define PAGE_SIZE  4096

// Indentation: 4 spaces
// Braces: K&R style for functions, inline for control
int function(void)
{
    if (condition) {
        // ...
    }
}
```

### Haskell

```haskell
-- Follow standard Haskell conventions
-- Types: PascalCase
data Expr = IntLit Int
          | BinOp Op Expr Expr

-- Functions: camelCase
parseExpression :: Parser Expr
parseExpression = ...

-- Use type signatures always
genCode :: Expr -> [Instruction]

-- Pattern matching preferred over if-then-else
evaluate (IntLit n) = n
evaluate (BinOp Add e1 e2) = evaluate e1 + evaluate e2
```

---

## Integration Checklist

When creating skeletons that integrate with existing components:

### Hardware Integration
- [ ] Clock domain correct?
- [ ] Reset polarity correct? (active-low)
- [ ] Bus widths match?
- [ ] MMIO addresses don't conflict?
- [ ] Timing constraints met?

### Software Integration
- [ ] Calling convention followed? (a0-a7 args, a0 return)
- [ ] Stack alignment preserved? (16-byte)
- [ ] Registers saved/restored correctly? (callee-saved)
- [ ] Error codes match POSIX? (negative errno)
- [ ] Structures packed correctly? (`__attribute__((packed))`)

### Cross-layer Integration
- [ ] MMIO registers match hardware?
- [ ] Interrupt vectors correct?
- [ ] Physical/virtual addresses distinguished?
- [ ] Endianness consistent? (little-endian)

---

## Progressive Disclosure

When helping the user, reveal complexity gradually:

**Level 1: Basic Skeleton**
- Just structure, no implementation
- Placeholder TODOs
- Core functionality only

**Level 2: Guided Implementation**
- Algorithm hints in comments
- Pseudocode for tricky parts
- References to similar code

**Level 3: Full Example**
- Complete implementation
- Optimizations
- Edge case handling
- Comprehensive tests

Start with Level 1 unless the user asks for more detail.

---

## Debugging Support

When helping debug, provide:

1. **Systematic approach**:
   - Isolate the problem (which stage/module?)
   - Reproduce minimally (simplest failing case)
   - Check assumptions (assertions, prints)
   - Binary search (disable half the code)

2. **Common bugs by section**:
   - **Section 2 (UART)**: Baud rate off by one, start/stop bit logic
   - **Section 3 (CPU)**: Forwarding path wrong, branch not flushing, sign extension
   - **Section 4 (Compiler)**: Register spilling, stack frame alignment, calling convention
   - **Section 5 (OS)**: Page table permissions, TLB not flushed, race conditions
   - **Section 6 (TCP)**: Sequence number wraparound, out-of-order handling, checksum

3. **Tool usage**:
   - GTKWave for signals
   - GDB for software
   - Printf debugging (via UART)
   - Assertions (SystemVerilog `assert`, C `assert()`)

---

## Consistency Examples

### Similar Patterns Should Look Similar

**State machines** across different modules should use the same style:

```systemverilog
// UART TX state machine
typedef enum logic [1:0] {
    IDLE, START, DATA, STOP
} uart_tx_state_t;

// SD card state machine  
typedef enum logic [2:0] {
    IDLE, CMD, WAIT_RESP, DATA_RX, DATA_TX
} sd_state_t;

// Always: typedef enum, _t suffix, UPPER_CASE states
```

**Error handling** should be consistent:

```c
// Kernel functions return negative errno
int sys_read(int fd, void *buf, size_t count) {
    if (fd < 0 || fd >= MAX_FDS) {
        return -EBADF;  // Bad file descriptor
    }
    // ...
}

// All error codes negative, success is 0 or positive count
```

**Parser structure** should match across languages:

```python
# Python assembler
def parse_instruction(tokens):
    opcode = tokens[0]
    if opcode == 'ADD':
        return parse_r_type(tokens)
    elif opcode == 'ADDI':
        return parse_i_type(tokens)
    # ...
```

```haskell
-- Haskell compiler
parseStmt :: Parser Stmt
parseStmt = parseIf
        <|> parseWhile
        <|> parseReturn
        <|> parseExprStmt
```

Both follow the same "dispatch by first token" pattern.

---

## Advanced: When to Go Beyond Skeleton

Sometimes the user needs more than a skeleton:

### Provide Complete Implementation If:
1. **Concept is novel and complex** (e.g., TLB replacement policy)
2. **Debugging is the focus**, not learning (e.g., fixing a subtle race condition)
3. **Boilerplate is extensive** (e.g., full ELF header generation)
4. **Reference implementation requested** (e.g., "show me a working version")

### Stick to Skeleton If:
1. **Core learning objective** (e.g., ALU in CPU)
2. **Pattern is already shown elsewhere** (e.g., 5th state machine)
3. **User specifically asks for guidance**, not solution
4. **Time allows incremental work**

**Always ask** if unsure: "Would you like a complete implementation, or a skeleton with TODOs for you to fill in?"

---

## Final Checklist for AI Assistants

Before providing any code or guidance:

- [ ] Have I read summary.md and the relevant section doc?
- [ ] Does my code follow the project's style conventions?
- [ ] Does my skeleton have clear TODOs and comments?
- [ ] Have I explained the WHY, not just the WHAT?
- [ ] Have I suggested how to test this?
- [ ] Have I mentioned integration points (MMIO, calling convention, etc.)?
- [ ] Is my response educational, not just functional?
- [ ] Have I provided the right level of detail (skeleton vs full)?

---

## Summary

This guide enables any AI assistant to:

1. **Understand the project**: Via summary.md and section docs
2. **Maintain consistency**: Via style guides and patterns
3. **Create quality skeletons**: Via templates and examples
4. **Guide effectively**: Via section-specific advice
5. **Debug systematically**: Via common pitfalls and tools

With these three documents (assist.md, summary.md, section doc), an AI can provide expert-level assistance throughout the entire Centurion project, from first LED blinker to final web browser on custom hardware.

Remember: **The goal is learning, not just working code.** Always prioritize understanding over completion.
