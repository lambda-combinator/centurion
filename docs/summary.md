# Centurion Project Summary

## High-Level Overview

**Centurion** is a comprehensive educational project that builds a complete computer system from logic gates to a web browser. The project spans **12 total units** and takes you from understanding FPGAs and SystemVerilog through building a RISC-V processor, C compiler, Unix-like operating system, TCP/IP networking stack, and finally deploying everything to custom hardware.

```
PROJECT STACK (Bottom to Top)
=============================

┌─────────────────────────────────────────────────────────────────┐
│  Section 7: Physical Hardware                                   │
│  Custom FPGA board, USB-JTAG programmer, real-world deployment  │
├─────────────────────────────────────────────────────────────────┤
│  Section 6: Networking & Browser                                │
│  TCP/IP stack, telnetd, dynamic linker, text-based web browser  │
├─────────────────────────────────────────────────────────────────┤
│  Section 5: Operating System                                    │
│  MMU, kernel, processes, filesystem, shell                      │
├─────────────────────────────────────────────────────────────────┤
│  Section 4: Compiler & Toolchain                                │
│  C compiler (Haskell), linker, libc, ethernet controller        │
├─────────────────────────────────────────────────────────────────┤
│  Section 3: Processor                                           │
│  RISC-V RV32I CPU (5-stage pipeline), assembler, bootrom        │
├─────────────────────────────────────────────────────────────────┤
│  Section 2: Bringup & Peripherals                               │
│  LED blinker, UART (TX/RX/MMIO)                                 │
├─────────────────────────────────────────────────────────────────┤
│  Section 1: Environment & Foundations                           │
│  FPGAs, SystemVerilog, Verilator, cocotb                        │
└─────────────────────────────────────────────────────────────────┘
```

### Core Technologies

| Layer | Language/Tool | Purpose |
|-------|---------------|---------|
| Hardware | SystemVerilog | RTL design (CPU, MMU, peripherals) |
| Simulation | Verilator + cocotb | Fast simulation with Python tests |
| Assembler | Python (~500 lines) | RISC-V assembly → machine code |
| Compiler | Haskell (~2000 lines) | C → RISC-V assembly |
| Linker | Python (~300 lines) | Object files → ELF executables |
| Kernel/Apps | C | Operating system and userspace |
| PCB/Firmware | KiCad + C | Physical board and JTAG adapter |

---

## Section-by-Section Summary

---

### Section 1: Intro — Cheating Our Way Past the Transistor

**Duration:** 0.5  
**Key Concept:** Accept transistors as our atomic primitive; build everything above them

#### What You Learn
- **FPGA Architecture**: LUTs (lookup tables), CLBs (configurable logic blocks), BRAM, programmable interconnect
- **HDL Fundamentals**: Combinational vs sequential logic, `always_comb` vs `always_ff`
- **Simulation**: Verilator compiles SystemVerilog to C++ for fast simulation
- **Testing**: cocotb enables Python-based testbenches with async/await

#### Key Insight
A 4-input LUT can implement ANY 4-input boolean function by storing a 16-entry truth table. Modern FPGAs use 6-input LUTs (64 SRAM cells each). FPGAs are reconfigurable arrays of these universal logic blocks.

#### Project Structure
```
centurion/
├── rtl/              # SystemVerilog source (core, soc, peripherals)
├── sim/               # cocotb testbenches
├── tools/            # assembler, linker, compiler
├── sw/               # bootrom, bootloader, libc, kernel, user programs
└── synth/            # synthesis scripts
```

#### Tools Required
- Verilator (≥5.0), cocotb, GTKWave, Python 3.9+, GHC + Stack (Haskell)

---

### Section 2: Bringup — What Language is Hardware Coded In?

**Duration:** 0.5   
**Key Concept:** Learn SystemVerilog through practical peripherals

#### Projects

**1. LED Blinker** (~50 lines SV, ~50 lines Python test)
- Clock-driven counter that toggles LED at configurable frequency
- Introduces: `always_ff`, parameters, `$clog2`, reset logic

**2. UART** (~200 lines SV, ~150 lines Python test)
- Full transmitter and receiver with MMIO interface
- Baud rate: 115200 (434 clock cycles per bit at 50MHz)
- Frame: 1 start bit + 8 data bits + 1 stop bit
- MMIO Registers:
  - `0x00` TXDATA (W): Write byte to transmit
  - `0x04` RXDATA (R): Read received byte
  - `0x08` STATUS (R): tx_ready, rx_valid flags
  - `0x0C` CTRL (R/W): enable bits, loopback mode

#### Key Files
- `rtl/peripherals/led_blinker.sv`
- `rtl/peripherals/uart_tx.sv`, `uart_rx.sv`, `uart_top.sv`
- `sim/test_led_blinker.py`, `sim/test_uart.py`

#### Why It Matters
UART is the primary debug interface for all future development. Every section relies on UART for console output and initial program loading.

---

### Section 3: Processor — What is a Processor Anyway?

**Duration:** 3   
**Key Concept:** Build a complete RISC-V RV32I processor from scratch

#### Projects

**1. RISC-V Assembler** (Python, ~500 lines)
- Two-pass assembly: collect labels, then encode instructions
- Supports all RV32I instructions + pseudo-instructions
- Output: raw binary or ELF format
- Pipeline: Lexer → Parser → Encoder → Output

**2. 5-Stage RISC-V CPU** (SystemVerilog, ~2000 lines)
- Classic pipeline: Fetch → Decode → Execute → Memory → Writeback
- Cache hierarchy: 16KB I-Cache + 16KB D-Cache (direct-mapped, 64-byte lines)
- Memory controller: Arbitrates cache requests, handles burst transfers (16 words)
- Main memory: 1MB total (Boot ROM 64KB, Program 448KB, Data RAM 512KB)
- Hazard handling: forwarding for data hazards, stalling for load-use
- Branch resolution in Execute stage (2-cycle penalty if taken)

**3. Boot ROM** (Assembly, ~40 lines)
- Initial code that loads programs via UART into RAM

#### RISC-V RV32I Key Facts
- 32 registers: x0 (hardwired zero) through x31
- ABI names: zero, ra, sp, gp, tp, t0-t6, s0-s11, a0-a7
- Instruction formats: R, I, S, B, U, J (all 32-bit aligned)
- ~47 base instructions: arithmetic, logical, shifts, loads/stores, branches, jumps

#### Pipeline Stages
| Stage | Function |
|-------|----------|
| IF | Fetch instruction from memory, increment PC |
| ID | Decode opcode, read registers, generate immediate |
| EX | ALU operation, branch comparison, address calculation |
| MEM | Load/store data memory access |
| WB | Write result back to register file |

#### Hazard Solutions
- **Data hazards (RAW)**: Forwarding from EX/MEM or MEM/WB stages
- **Load-use hazard**: 1-cycle stall + forwarding
- **Control hazards**: Flush 2 instructions if branch taken

#### Key Files
- `tools/assembler/riscv_asm.py`, `lexer.py`, `parser.py`, `encoder.py`
- `rtl/core/fetch.sv`, `decode.sv`, `execute.sv`, `memory.sv`, `writeback.sv`
- `rtl/core/icache.sv`, `dcache.sv`, `mem_controller.sv`, `memory_model.sv`
- `rtl/core/hazard_unit.sv`, `register_file.sv`, `alu.sv`
- `sw/bootrom/bootrom.s`

---

### Section 4: Compiler — A "High" Level Language

**Duration:** 3   
**Key Concept:** Enable C programming on our custom hardware

#### Projects

**1. C Compiler** (Haskell, ~2000 lines)
- Pipeline: Lexer → Parser → Semantic Analysis → Code Generation
- No optimization pass (output is valid but not efficient)
- Outputs RISC-V assembly (then assembled with our assembler)

**2. Linker** (Python, ~300 lines)
- Combines object files, resolves symbols
- Produces ELF executables

**3. libc** (C, ~500 lines)
- Minimal standard library: printf, malloc, string functions, syscall wrappers

**4. Ethernet Controller** (SystemVerilog, ~200 lines)
- MII interface to PHY chip
- Frame TX/RX with CRC

**5. Bootloader** (C, ~300 lines)
- Loads kernel image over ethernet using simple protocol

#### Supported C Subset
```
Types:      int, char, void, pointers, arrays, struct
Statements: if/else, while, for, return, compound blocks
Expressions: arithmetic, comparison, logical, bitwise, 
            assignment, address-of, dereference, 
            array subscript, struct member, function calls, sizeof, casts
NOT supported: float/double, union, enum, switch, goto, 
               multi-dim arrays, variadic functions, preprocessor
```

#### RISC-V Calling Convention (ILP32)
- Arguments: a0-a7 (first 8 args), rest on stack
- Return: a0
- Caller-saved: t0-t6, a0-a7
- Callee-saved: s0-s11, sp
- Stack: grows downward, 16-byte aligned

#### Key Files
- `tools/compiler/Main.hs`, `Lexer.hs`, `Parser.hs`, `Semantic.hs`, `CodeGen.hs`
- `tools/linker/linker.py`
- `sw/libc/` (stdio.c, stdlib.c, string.c, etc.)
- `rtl/peripherals/ethernet_mac.sv`
- `sw/bootloader/main.c`

---

### Section 5: Operating System — Software We Take for Granted

**Duration:** 3   
**Key Concept:** Build a Unix-like OS with virtual memory and processes

#### Projects

**1. MMU** (SystemVerilog, ~1000 lines)
- RISC-V Sv32: two-level page tables, 4KB pages
- TLB (Translation Lookaside Buffer): fully associative cache of translations
- Page Table Walker: handles TLB misses
- Permission checking: R/W/X, User/Supervisor

**2. Kernel** (C, ~2500 lines)
- Process management: fork, exec, wait, exit
- Memory management: page allocation, virtual address spaces
- Scheduler: simple round-robin
- Trap handling: syscalls, page faults, interrupts

**3. SD Card Controller** (SystemVerilog, ~150 lines)
- SPI mode interface to SD card

**4. FAT Filesystem** (C, ~300 lines)
- Read/write files on SD card

**5. User Programs** (C, ~250 lines)
- Shell, cat, ls, echo

#### Virtual Memory (Sv32)
```
Virtual Address (32-bit):
┌────────────────┬────────────────┬──────────────┐
│   VPN[1]       │   VPN[0]       │   Offset     │
│   (10 bits)    │   (10 bits)    │  (12 bits)   │
└────────────────┴────────────────┴──────────────┘

Page Table Entry:
┌──────────────────────┬─┬─┬─┬─┬─┬─┬─┬─┐
│        PPN           │D│A│G│U│X│W│R│V│
│      (22 bits)       │ │ │ │ │ │ │ │ │
└──────────────────────┴─┴─┴─┴─┴─┴─┴─┴─┘
V=Valid, R=Read, W=Write, X=Execute, U=User, 
G=Global, A=Accessed, D=Dirty
```

#### Memory Map
```
0x0000_0000 - 0x7FFF_FFFF: User space (2GB per process)
0x8000_0000 - 0xFFFF_FFFF: Kernel space (2GB shared)
  0x8000_0000: Kernel code/data
  0x8010_0000: Kernel heap
  0x8100_0000: Direct-mapped physical memory
  0xF000_0000: MMIO (UART, Ethernet, SD Card)
```

#### System Calls
- File: open, read, write, close, lseek
- Process: fork, execve, wait, exit, getpid, sleep
- Memory: mmap, munmap, mprotect, brk

#### Key Files
- `rtl/core/mmu.sv`, `tlb.sv`, `ptw.sv`
- `rtl/peripherals/sd_controller.sv`
- `sw/kernel/main.c`, `trap.c`, `syscall.c`, `proc.c`, `vm.c`, `fs.c`
- `sw/user/shell.c`, `cat.c`, `ls.c`

---

### Section 6: Browser — Coming Online

**Duration:** 1 week  
**Key Concept:** Implement networking to connect the system to the internet

#### Projects

**1. TCP/IP Stack** (C, ~500 lines)
- Socket interface: socket, bind, listen, accept, connect, send, recv, close
- TCP state machine: CLOSED → SYN_SENT → ESTABLISHED → FIN_WAIT → CLOSED
- Three-way handshake, four-way teardown
- Sequence numbers, acknowledgments, flow control

**2. telnetd** (C, ~50 lines)
- Remote shell access over TCP port 23

**3. Dynamic Linker** (C, ~300 lines)
- Shared library support (.so files)
- Symbol resolution at load time

**4. Web Browser** (C, ~500+ lines)
- Text-based HTTP/1.0 client
- DNS resolution
- HTML parsing (minimal)
- Renders to console

#### Network Stack Layers
```
Application:  HTTP, telnet, DNS
Transport:    TCP (reliable), UDP (unreliable)
Network:      IP, ICMP (ping), ARP (address resolution)
Link:         Ethernet frames
Hardware:     MAC controller (from Section 4)
```

#### TCP Connection Lifecycle
```
Client:  CLOSED → SYN_SENT → ESTABLISHED → FIN_WAIT → TIME_WAIT → CLOSED
Server:  CLOSED → LISTEN → SYN_RCVD → ESTABLISHED → CLOSE_WAIT → LAST_ACK → CLOSED
```

#### Key Files
- `sw/kernel/net/tcp.c`, `udp.c`, `ip.c`, `arp.c`, `ethernet.c`
- `sw/kernel/net/socket.c`
- `sw/user/telnetd.c`
- `sw/user/browser.c`

---

### Section 7: Physical — Running on Real Hardware

**Duration:** 1 week  
**Key Concept:** Deploy everything to a custom FPGA board

#### Projects

**1. USB-JTAG Interface** (C, ~200 lines)
- RP2040 (Raspberry Pi Pico) firmware
- JTAG bit-banging via GPIO (or PIO for speed)
- TinyUSB library for USB communication
- Commands: shift IR/DR, read ID, program bitstream

**2. FPGA Board Design** (PCB)
- FPGA: Xilinx Spartan-7 or Lattice ECP5
- Memory: 1-4 MB SRAM (16/32-bit)
- Clock: 50 MHz oscillator
- Peripherals: USB (JTAG + UART), Ethernet PHY, SD card slot
- User I/O: LEDs, buttons, expansion header

**3. Board Bringup**
- Power supply verification
- JTAG connectivity test
- LED blinker test
- UART test
- Full system boot

#### JTAG Signals
| Signal | Direction | Purpose |
|--------|-----------|---------|
| TCK | Out | Test clock |
| TMS | Out | Test mode select (state machine control) |
| TDI | Out | Test data in |
| TDO | In | Test data out |
| TRST | Out | Test reset (optional) |

#### Board Components
- **Power**: USB 5V → 3.3V LDO → 1.0V/1.8V core regulators
- **FPGA**: XC7S50 or LFE5U-25F
- **SRAM**: IS61WV102416 (1M×16, 10ns)
- **Ethernet**: LAN8720A PHY + RJ-45 with magnetics
- **SD Card**: Micro SD in SPI mode

#### Key Files
- `hw/jtag_adapter/main.c`, `pio_jtag.pio`
- `tools/fpga_prog/fpga_prog.py`
- `hw/pcb/` (KiCad schematic and layout)

---

## Line Count Summary

| Component | Language | Lines |
|-----------|----------|-------|
| LED Blinker | SystemVerilog | ~50 |
| UART | Syst+ Caches | SystemVerilog | ~2000 |
| MMU | SystemVerilog | ~1000 |
| SD Controller | SystemVerilog | ~150 |
| Ethernet MAC | SystemVerilog | ~200 |
| **Total Hardware** | **SystemVerilog** | **~36
| **Total Hardware** | **SystemVerilog** | **~3100** |
| Assembler | Python | ~500 |
| Linker | Python | ~300 |
| Tests | Python | ~500 |
| C Compiler | Haskell | ~2000 |
| Boot ROM | Assembly | ~40 |
| Bootloader | C | ~300 |
| libc | C | ~500 |
| Kernel | C | ~2500 |
| TCP/IP Stack | C | ~500 |
| User Programs | C | ~800 |
| JTAG Firmware | C | ~200 |
| **Total Software** | **Mixed** | **~8140** |
7
**Grand Total: ~11,240 lines** (excluding tests and scripts)

---

## Dependency Chain

```
Section 1 (Environment)
    │
    ▼
Section 2 (UART) ─────────────────────────────────────┐
    │                                                 │
    ▼                                                 │ (debug console)
Section 3 (CPU + Assembler) ──────────────────────────┤
    │                                                 │
    ▼                                                 │
Section 4 (Compiler + Ethernet) ──────────────────────┤
    │                                                 │
    ▼                                                 │
Section 5 (OS + MMU) ─────────────────────────────────┤
    │                                                 │
    ▼                                                 │
Section 6 (TCP/IP + Browser) ─────────────────────────┘
    │
    ▼
Section 7 (Physical Hardware)
```

Every section depends on all previous sections. The UART from Section 2 remains the primary debug interface throughout. The CPU from Section 3 runs all subsequent software. The compiler from Section 4 enables writing the OS and applications. The OS from Section 5 provides the environment for networking and user programs.

---

## Quick Reference: What Each Section Produces

| Section | Hardware Output | Software Output | Capability Gained |
|---------|-----------------|-----------------|-------------------|
| 1 | — | Dev environment | Can simulate SystemVerilog |
| 2 | LED blinke, I-Cache, D-Cache, Memory Controller | Assembler, bootrom | Can execute machine code with cachingrial |
| 3 | RISC-V CPU | Assembler, bootrom | Can execute machine code |
| 4 | Ethernet MAC | Compiler, linker, libc, bootloader | Can compile and run C programs |
| 5 | MMU, SD controller | Kernel, filesystem, shell | Can run multiple processes with isolation |
| 6 | — | TCP stack, telnetd, browser | Can connect to internet, browse web |
| 7 | FPGA board, JTAG adapter | Programming tools | Can run on real hardware |

---

## How to Use This Summary

When working on any section, refer back here to understand:

1. **What came before**: What hardware and software already exists
2. **What you're building**: The specific deliverables for this section
3. **What comes next**: How your work enables future sections
4. **Interfaces**: How your component connects to others (MMIO addresses, syscall numbers, calling conventions)

This summary provides the "big picture" context that makes each section's detailed implementation guide more meaningful.
