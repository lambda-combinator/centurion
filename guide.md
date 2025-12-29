# PyRTL Quick Reference Guide

A comprehensive guide for writing hardware designs in PyRTL for AI accelerators, processors, and complex digital systems.

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Basic Building Blocks](#basic-building-blocks)
3. [Wires and Logic Operations](#wires-and-logic-operations)
4. [Registers and State](#registers-and-state)
5. [Memory Structures](#memory-structures)
6. [Simulation and Testing](#simulation-and-testing)
7. [Control Flow and Conditional Logic](#control-flow-and-conditional-logic)
8. [Common Patterns for Processors and Accelerators](#common-patterns-for-processors-and-accelerators)
9. [Debugging and Analysis](#debugging-and-analysis)
10. [Export and Integration](#export-and-integration)

---

## Core Concepts

### What is PyRTL?

PyRTL is a Python-based Register Transfer Level (RTL) framework that allows you to:
- Specify hardware designs using Python
- Simulate hardware behavior before synthesis
- Export designs to Verilog
- Analyze timing, area, and power consumption
- Perform hardware optimization and transformation

### Key Principle: Elaboration vs. Execution

**Elaboration** happens when you write Python code that constructs hardware. This creates the logic graph.
**Execution** happens during simulation when the hardware runs on test inputs.

```python
import pyrtl

# Elaboration: Define what hardware to build
a = pyrtl.Input(8, 'a')
b = pyrtl.Input(8, 'b')
result = pyrtl.Output(9, 'result')
result <<= a + b  # This BUILDS an adder, doesn't compute it

# Execution: Run the hardware on inputs
sim = pyrtl.Simulation()
sim.step({'a': 5, 'b': 3})  # Now the adder executes
```

---

## Basic Building Blocks

### WireVector: The Fundamental Type

All hardware in PyRTL is built from `WireVector` objects, representing multi-bit wires.

```python
# Create a generic WireVector
wire = pyrtl.WireVector(bitwidth=8, name='my_wire')

# Access bitwidth
bitwidth = len(wire)  # Returns 8

# Slicing (like Python lists)
lsb = wire[0]           # Least significant bit
upper_bits = wire[4:8]  # Bits 4-7
```

### Input Pins

Define circuit inputs:

```python
# Create input pins
a = pyrtl.Input(bitwidth=8, name='a')
b = pyrtl.Input(bitwidth=8, name='b')
clk = pyrtl.Input(bitwidth=1, name='clk')

# Inputs are used as operands for logic operations
sum_val = a + b
```

### Output Pins

Define circuit outputs:

```python
# Create output pins
result = pyrtl.Output(bitwidth=8, name='result')
ready = pyrtl.Output(bitwidth=1, name='ready')

# Connect logic to outputs using the <<= operator
result <<= sum_val
ready <<= 1  # Can assign constants
```

### Constants

Define fixed values:

```python
# Create constants in multiple ways
const1 = pyrtl.Const(5, bitwidth=8)              # Value 5 in 8 bits
const2 = pyrtl.Const("8'b11110000")              # Verilog-style: 11110000
const3 = pyrtl.Const("16'hDEAD")                 # Hex: DEAD in 16 bits
const4 = pyrtl.Const("32'o755")                  # Octal: 755 in 32 bits
```

---

## Wires and Logic Operations

### Arithmetic Operations

```python
a = pyrtl.Input(8, 'a')
b = pyrtl.Input(8, 'b')

# Basic arithmetic
sum_ab = a + b              # Addition (result width auto-expanded)
diff = a - b                # Subtraction (2's complement)
product = a * b             # Multiplication
quotient = a // b           # Integer division
remainder = a % b           # Modulo

# Bitwise operations
and_result = a & b          # AND
or_result = a | b           # OR
xor_result = a ^ b          # XOR
not_result = ~a             # NOT (bitwise complement)
```

### Comparison Operations

```python
# Comparisons return 1-bit results
eq = (a == b)               # Equality
ne = (a != b)               # Not equal
lt_unsigned = a < b         # Less than (unsigned)
le = a <= b                 # Less than or equal
gt = a > b                  # Greater than
ge = a >= b                 # Greater than or equal

# For signed comparisons, use helpers
lt_signed = pyrtl.signed_lt(a, b)
```

### Bit Shifting and Rotation

```python
# Shift operations
lshift = a << 2             # Left shift by 2 bits
rshift = a >> 2             # Right shift by 2 bits

# Rotation (from rtllib)
from pyrtl.rtllib import barrel
rotated = barrel.barrel_shifter(a, rotation_amount=2)
```

### Wire Concatenation and Slicing

```python
# Concatenate wires (rightmost is LSB)
concat_val = pyrtl.concat(msb_wire, mid_wire, lsb_wire)

# Slicing by index
bit0 = wire[0]              # Bit 0 (LSB)
bits_4_to_7 = wire[4:8]     # Bits 4-7
all_but_msb = wire[:-1]     # All bits except MSB

# Extending wires
zero_extended = wire.zero_extended(bitwidth=16)    # Pad with zeros
sign_extended = wire.sign_extended(bitwidth=16)    # Pad with sign bit
truncated = wire.truncate(bitwidth=4)               # Keep only lower bits
```

### Multiplexing (Selection)

```python
# 2-to-1 multiplexer (ternary operator)
result = pyrtl.select(select_signal, 
                      truecase=wire_a, 
                      falsecase=wire_b)

# Multi-way multiplexer (prioritized)
result = pyrtl.select(select_signal,
                      falsecase=default_val,
                      {0: val_0, 1: val_1, 2: val_2})

# select with chaining conditions
with pyrtl.conditional_assignment:
    with select_condition_a:
        result |= value_a
    with select_condition_b:
        result |= value_b
    with pyrtl.otherwise:
        result |= default_value
```

### Reduction Operations

```python
# Combine all bits in a wire
all_bits_and = pyrtl.rtl_all(wire)      # AND all bits
any_bit_set = pyrtl.rtl_any(wire)       # OR all bits
bits_differ = pyrtl.rtl_xor(wire)       # XOR all bits
count_ones = pyrtl.popcount(wire)       # Count 1 bits
```

---

## Registers and State

### Basic Register

A register holds a value across clock cycles:

```python
# Create a register
counter = pyrtl.Register(bitwidth=8, name='counter')

# Read current value
current_count = counter

# Set next value (takes effect at next clock edge)
counter.next <<= counter + 1

# In simulation, registers automatically handle state
sim = pyrtl.Simulation()
for _ in range(10):
    sim.step()
    print(sim.value[counter])  # Access register value at current cycle
```

### Register Initialization

```python
# Set initial value when creating register
state = pyrtl.Register(bitwidth=3, name='state', reset_value=0)

# Or set in simulation
initial_values = {state: 5}
sim = pyrtl.Simulation(tracer=..., memory_value_map=None)
```

### Simple State Machine

```python
class State(enum.IntEnum):
    IDLE = 0
    ACTIVE = 1
    DONE = 2

state = pyrtl.Register(2, 'state', reset_value=State.IDLE)
request = pyrtl.Input(1, 'request')
done = pyrtl.Output(1, 'done')

# Conditional state transitions
with pyrtl.conditional_assignment:
    with state == State.IDLE:
        with request:
            state.next |= State.ACTIVE
    with state == State.ACTIVE:
        state.next |= State.DONE
    with state == State.DONE:
        state.next |= State.IDLE

done <<= (state == State.DONE)
```

---

## Memory Structures

### MemBlock (Read-Write Memory)

```python
# Create a memory
mem = pyrtl.MemBlock(bitwidth=32,        # Data width
                     addrwidth=10,        # Address width (1024 locations)
                     name='my_memory')

# Memory inputs
raddr = pyrtl.Input(10, 'raddr')         # Read address
waddr = pyrtl.Input(10, 'waddr')         # Write address
wdata = pyrtl.Input(32, 'wdata')         # Write data
we = pyrtl.Input(1, 'we')                # Write enable

# Memory outputs
rdata = pyrtl.Output(32, 'rdata')        # Read data

# Read operation (combinational)
rdata <<= mem[raddr]

# Write operation (with enable)
mem[waddr] <<= pyrtl.MemBlock.EnabledWrite(wdata, we)
```

### RomBlock (Read-Only Memory)

```python
# Define ROM data
def rom_init_function(addr):
    # Return data for each address
    return addr * 2  # Example: return 2*addr

# Create ROM
rom = pyrtl.RomBlock(bitwidth=8,         # Data width
                     addrwidth=4,         # 16 locations
                     romdata=rom_init_function,
                     name='lookup_table')

# Or use a list
rom_data = [1, 2, 4, 8, 16, 32, 64, 128]
rom = pyrtl.RomBlock(8, 3, romdata=rom_data, name='lut')

# Read operation
addr = pyrtl.Input(3, 'addr')
data = pyrtl.Output(8, 'data')
data <<= rom[addr]
```

### Memory in Simulation

```python
# Initialize memory before simulation
mem = pyrtl.MemBlock(32, 8, name='mem')

# Create initial values dictionary
init_mem = {
    0: 0x12345678,
    1: 0x9ABCDEF0,
    2: 0xFEDCBA98
}

memory_map = {mem: init_mem}
sim = pyrtl.Simulation(memory_value_map=memory_map)

# Access memory contents after simulation
final_mem = sim.inspect_mem(mem)
```

---

## Simulation and Testing

### Basic Simulation

```python
import pyrtl

# Create circuit
a = pyrtl.Input(8, 'a')
b = pyrtl.Input(8, 'b')
result = pyrtl.Output(8, 'result')
result <<= a + b

# Create and run simulation
sim = pyrtl.Simulation()

# Step through one cycle
sim.step({'a': 5, 'b': 3})

# Access output
print(sim.inspect(result))  # Prints: 8

# Step multiple cycles with arrays
sim.step_multiple({
    'a': [1, 2, 3, 4, 5],
    'b': [5, 4, 3, 2, 1]
})
```

### Working with Simulation Traces

```python
# Simulation automatically records traces
sim = pyrtl.Simulation()
sim.step_multiple({'a': [0, 1, 2, 3], 'b': [3, 2, 1, 0]})

# Access trace data
trace = sim.tracer.trace
a_values = trace['a']           # Get all 'a' values
b_values = trace['b']           # Get all 'b' values

# Access specific cycle
cycle_2_a = sim.inspect('a')    # Latest value
cycle_2_a_at_cycle = trace['a'][2]  # Value at cycle 2

# Render as ASCII waveform
sim.tracer.render_trace()

# Render with custom settings
sim.tracer.render_trace(
    trace_list=['a', 'b', 'result'],
    symbol_len=2,               # Symbol width
    repr_func=lambda v: f"{v:02x}"  # Custom formatter
)
```

### Fast Simulation (JIT Compilation)

For better performance with large circuits:

```python
# Use FastSimulation instead of Simulation
from pyrtl import FastSimulation

sim = FastSimulation()
sim.step_multiple({'a': [1, 2, 3], 'b': [4, 5, 6]})

# Interface is identical to Simulation
```

### Compiled Simulation (C Backend)

For maximum performance:

```python
from pyrtl import CompiledSimulation

sim = CompiledSimulation()
sim.step_multiple({'a': [1, 2, 3], 'b': [4, 5, 6]})
```

### Random Testing

```python
import random

sim = pyrtl.Simulation()
for _ in range(1000):
    a_val = random.randint(0, 255)
    b_val = random.randint(0, 255)
    sim.step({'a': a_val, 'b': b_val})
    expected = (a_val + b_val) & 0xFF
    assert sim.inspect(result) == expected
```

---

## Control Flow and Conditional Logic

### Conditional Assignment

The `conditional_assignment` context manager builds multiplexers:

```python
signal = pyrtl.Input(1, 'signal')
value_a = pyrtl.Input(8, 'value_a')
value_b = pyrtl.Input(8, 'value_b')
output = pyrtl.Output(8, 'output')

# All conditions are evaluated (no sequential behavior)
with pyrtl.conditional_assignment:
    with signal == 1:
        output |= value_a
    with pyrtl.otherwise:
        output |= value_b

# Nested conditionals
with pyrtl.conditional_assignment:
    with condition1:
        output |= value1
        with condition2:
            output |= value2
    with condition3:
        output |= value3
```

### Register Updates with Conditionals

```python
state = pyrtl.Register(2, 'state')

with pyrtl.conditional_assignment:
    with external_reset:
        state.next |= 0
    with enable_signal:
        state.next |= state + 1
    # If neither condition is true, register keeps its value
```

### Mux-based Conditional

```python
# When you need maximum clarity about a 2-way select
sel = pyrtl.Input(1, 'sel')
true_val = pyrtl.Input(8, 'true_val')
false_val = pyrtl.Input(8, 'false_val')
result = pyrtl.Output(8, 'result')

result <<= pyrtl.select(sel, truecase=true_val, falsecase=false_val)
```

---

## Common Patterns for Processors and Accelerators

### Pipeline Stage

```python
class PipelineStage:
    def __init__(self, stage_width=32):
        self.stage_width = stage_width
    
    def create_stage(self, data_in, enable):
        """Create a single pipeline register stage"""
        reg = pyrtl.Register(self.stage_width, name=f'pipe_stage')
        
        with pyrtl.conditional_assignment:
            with enable:
                reg.next |= data_in
        
        return reg

# Usage
pipeline = PipelineStage(stage_width=64)
stage1_out = pipeline.create_stage(input_data, enable)
stage2_out = pipeline.create_stage(stage1_out, enable)
```

### Simple Counter

```python
def create_counter(bitwidth=8, increment=1, enable_signal=None):
    """Create a counter with optional enable"""
    counter = pyrtl.Register(bitwidth, name='counter')
    
    if enable_signal is None:
        counter.next <<= counter + increment
    else:
        with pyrtl.conditional_assignment:
            with enable_signal:
                counter.next |= counter + increment
    
    return counter
```

### ALU (Arithmetic Logic Unit)

```python
def simple_alu(a, b, opcode):
    """
    Simple ALU supporting multiple operations
    opcode: 00=add, 01=sub, 10=and, 11=or
    """
    add_result = a + b
    sub_result = a - b
    and_result = a & b
    or_result = a | b
    
    # 4-input multiplexer
    result = pyrtl.select(opcode, 
        falsecase=add_result,
        {
            0: add_result,
            1: sub_result,
            2: and_result,
            3: or_result
        })
    
    return result
```

### Barrel Shifter

```python
from pyrtl.rtllib.barrel import barrel_shifter

def create_shifter(data, shift_amount, shift_type='logical'):
    """
    Create a barrel shifter
    shift_type: 'logical', 'arithmetic', or 'rotate'
    """
    if shift_type == 'logical':
        # Zero-fill
        shifted = pyrtl.concat(*[pyrtl.Const(0, 1) for _ in range(shift_amount)], data)
    elif shift_type == 'arithmetic':
        # Sign-fill
        sign_bit = data[-1]
        shifted = pyrtl.concat(*[sign_bit for _ in range(shift_amount)], data)
    else:  # rotate
        shifted = barrel_shifter(data, shift_amount)
    
    return shifted
```

### Pipelined Adder

```python
def pipelined_adder_stages(a, b, num_stages=3):
    """
    Create a pipelined N-bit adder across multiple stages
    """
    current_a = a
    current_b = b
    
    for stage in range(num_stages - 1):
        # Register A and B inputs for next stage
        reg_a = pyrtl.Register(len(a), name=f'pipe_a_{stage}')
        reg_b = pyrtl.Register(len(b), name=f'pipe_b_{stage}')
        
        reg_a.next <<= current_a
        reg_b.next <<= current_b
        
        current_a = reg_a
        current_b = reg_b
    
    # Final addition in last stage
    result = current_a + current_b
    return result
```

### Multiplier with Pipelining

```python
from pyrtl.rtllib.multipliers import tree_multiplier

def pipelined_multiplier(a, b, num_stages=2):
    """Pipelined multiplier"""
    # Register inputs
    pipeline_regs_a = []
    pipeline_regs_b = []
    
    current_a = a
    current_b = b
    
    for i in range(num_stages - 1):
        reg_a = pyrtl.Register(len(a), name=f'mult_a_{i}')
        reg_b = pyrtl.Register(len(b), name=f'mult_b_{i}')
        
        reg_a.next <<= current_a
        reg_b.next <<= current_b
        
        current_a = reg_a
        current_b = reg_b
    
    # Multiply in final stage
    result = tree_multiplier(current_a, current_b)
    return result
```

### Memory with Read/Write Control

```python
def create_dual_port_memory(data_width=32, addr_width=10):
    """Create a simple dual-port RAM"""
    mem = pyrtl.MemBlock(data_width, addr_width, name='dual_port_mem')
    
    # Read port 1
    read_addr1 = pyrtl.Input(addr_width, 'read_addr1')
    read_data1 = pyrtl.Output(data_width, 'read_data1')
    read_data1 <<= mem[read_addr1]
    
    # Read port 2
    read_addr2 = pyrtl.Input(addr_width, 'read_addr2')
    read_data2 = pyrtl.Output(data_width, 'read_data2')
    read_data2 <<= mem[read_addr2]
    
    # Write port
    write_addr = pyrtl.Input(addr_width, 'write_addr')
    write_data = pyrtl.Input(data_width, 'write_data')
    write_enable = pyrtl.Input(1, 'write_enable')
    
    mem[write_addr] <<= pyrtl.MemBlock.EnabledWrite(write_data, write_enable)
    
    return mem
```

---

## Debugging and Analysis

### Probing Internal Signals

```python
# Probe without creating output
pyrtl.probe(internal_wire, 'debug_probe_name')

# Probe inline (returns the wire)
output <<= pyrtl.probe(intermediate_result, 'intermediate_probe') & mask

# Access probe data from tracer
sim.tracer.render_trace(trace_list=['debug_probe_name'])
```

### Timing Analysis

```python
# Analyze critical path
timing = pyrtl.TimingAnalysis()
max_delay = timing.max_length()
print(f"Critical path delay: {max_delay}")

# Get critical paths
paths = timing.critical_path()
for path in paths:
    print(f"Path: {' -> '.join([str(net) for net in path])}")
```

### Area Estimation

```python
# Estimate area in nm^2
logic_area, mem_area = pyrtl.area_estimation(tech_in_nm=65)
total_area = logic_area + mem_area
print(f"Logic area: {logic_area} um^2")
print(f"Memory area: {mem_area} um^2")
print(f"Total area: {total_area} um^2")
```

### Circuit Inspection

```python
# Print the entire circuit
print(pyrtl.working_block())

# Get all wires in circuit
all_wires = pyrtl.working_block().wirevector_set

# Get inputs/outputs
inputs = pyrtl.working_block().wirevector_subset(pyrtl.Input)
outputs = pyrtl.working_block().wirevector_subset(pyrtl.Output)
registers = pyrtl.working_block().wirevector_subset(pyrtl.Register)

# Get a wire by name
wire = pyrtl.working_block().get_wirevector_by_name('signal_name')

# Inspect simulation values
current_value = sim.inspect('signal_name')
all_values = sim.tracer.trace['signal_name']
```

### Assertions in Hardware

```python
# Use Python assertions to verify design properties
a = pyrtl.Input(8, 'a')
b = pyrtl.Input(8, 'b')

assert len(a) == 8
assert len(b) == 8

# Check invariants during simulation
sim = pyrtl.Simulation()
for i in range(10):
    sim.step({'a': i, 'b': 10-i})
    assert sim.inspect('result') <= 255
```

---

## Export and Integration

### Export to Verilog

```python
import io

# Export circuit to Verilog
with open('design.v', 'w') as f:
    pyrtl.output_to_verilog(f)

# Or use StringIO for inspection
with io.StringIO() as f:
    pyrtl.output_to_verilog(f)
    verilog_text = f.getvalue()
    print(verilog_text)
```

### Export Simulation Trace as Testbench

```python
# Generate a Verilog testbench from simulation
with open('testbench.v', 'w') as f:
    pyrtl.output_verilog_testbench(f)
```

### Export Simulation Trace to VCD

```python
# VCD format for waveform viewers (gtkwave, etc.)
with open('trace.vcd', 'w') as f:
    sim.tracer.print_vcd(f)
```

### Import from BLIF

```python
# Import circuit from BLIF (Berkeley Logic Interchange Format)
blif_string = """
.model my_circuit
.inputs a b
.outputs z
.names a b z
11 1
.end
"""

pyrtl.input_from_blif(blif_string)

# Or import from file
pyrtl.input_from_blif(filename='circuit.blif')
```

### Import from Verilog

PyRTL doesn't directly import Verilog, but BLIF format is widely supported. Use `yosys` to convert:

```bash
# Convert Verilog to BLIF using yosys
yosys -p "read_verilog design.v; write_blif design.blif"
```

---

## Synthesis and Optimization

### Synthesis

Convert high-level operations to primitive gates:

```python
# Before synthesis: Circuit uses high-level operations
print("Pre-synthesis gate count:")
print(pyrtl.working_block())

# Synthesize down to 1-bit operations and primitive gates
pyrtl.synthesize()

print("\nPost-synthesis gate count:")
print(pyrtl.working_block())
```

### Optimization

Remove dead logic and simplify:

```python
# Optimize circuit
pyrtl.optimize()

# Check result
timing = pyrtl.TimingAnalysis()
print(f"Optimized critical path: {timing.max_length()}")
```

### Reset Working Block

Clear all hardware for starting fresh:

```python
pyrtl.reset_working_block()
# Now all previous wires are invalid, start over
```

---

## Advanced: Working Block Context

### Multiple Blocks

```python
# Get current working block
current_block = pyrtl.working_block()

# Create new block
new_block = pyrtl.Block()

# Set as working block
pyrtl.set_working_block(new_block)

# Wires created now belong to new_block
a = pyrtl.Input(8, 'a')
```

### Block as Subcomponent

```python
def create_adder_block(bitwidth):
    """Create an adder as a separate block"""
    block = pyrtl.Block()
    
    with pyrtl.set_working_block(block):
        a = pyrtl.Input(bitwidth, 'a')
        b = pyrtl.Input(bitwidth, 'b')
        result = pyrtl.Output(bitwidth+1, 'result')
        result <<= a + b
    
    return block
```

---

## Best Practices for Processor and Accelerator Design

### 1. Structural Design

```python
# Organize your design hierarchically
def create_datapath(bitwidth=32):
    """Create the datapath component"""
    # ALU, registers, muxes
    pass

def create_control(bitwidth=32):
    """Create the control logic"""
    # State machine, opcode decoder
    pass

# Wire them together in main
datapath = create_datapath()
control = create_control()
```

### 2. Naming Convention

```python
# Use clear, hierarchical names
alu_result = pyrtl.Output(32, 'alu/result')
fetch_addr = pyrtl.Output(32, 'fetch/addr')
decode_opcode = pyrtl.Output(8, 'decode/opcode')
```

### 3. Constant Definition

```python
# Define constants at the top
DATAPATH_WIDTH = 32
ADDR_WIDTH = 16
NUM_REGS = 32
OPCODE_WIDTH = 8

# Use them consistently
alu_a = pyrtl.Input(DATAPATH_WIDTH, 'alu_a')
memory_addr = pyrtl.Input(ADDR_WIDTH, 'mem_addr')
```

### 4. Register File Example

```python
def create_register_file(bitwidth=32, num_regs=32):
    """Create a register file with multiple read/write ports"""
    registers = []
    
    # Create registers
    for i in range(num_regs):
        reg = pyrtl.Register(bitwidth, name=f'reg_{i}')
        registers.append(reg)
    
    # Read ports
    read_addr1 = pyrtl.Input(log2(num_regs), 'read_addr1')
    read_addr2 = pyrtl.Input(log2(num_regs), 'read_addr2')
    read_data1 = pyrtl.Output(bitwidth, 'read_data1')
    read_data2 = pyrtl.Output(bitwidth, 'read_data2')
    
    # Create multiplexer for reads
    read_data1 <<= pyrtl.select(read_addr1, 
                                falsecase=registers[0],
                                {i: registers[i] for i in range(num_regs)})
    read_data2 <<= pyrtl.select(read_addr2,
                                falsecase=registers[0],
                                {i: registers[i] for i in range(num_regs)})
    
    # Write port
    write_addr = pyrtl.Input(log2(num_regs), 'write_addr')
    write_data = pyrtl.Input(bitwidth, 'write_data')
    write_enable = pyrtl.Input(1, 'write_enable')
    
    # Update registers based on write
    for i, reg in enumerate(registers):
        with pyrtl.conditional_assignment:
            with (write_addr == i) & write_enable:
                reg.next |= write_data
    
    return registers
```

### 5. Testing Strategy

```python
def test_alu(alu_func, test_cases):
    """Generic ALU test"""
    sim = pyrtl.Simulation()
    
    for opcode, a, b, expected in test_cases:
        sim.step({
            'opcode': opcode,
            'input_a': a,
            'input_b': b
        })
        result = sim.inspect('alu_result')
        assert result == expected, f"Failed: {opcode} {a} {b} -> {result} != {expected}"
    
    print("All tests passed!")
```

---

## Common Gotchas and Tips

### Elaboration Time vs. Runtime

```python
# WRONG: This builds the same adder N times
def bad_pipeline(data, stages=3):
    for i in range(stages):
        data = data + 1  # This elaborates a chain of adders
    return data

# CORRECT: Use registers to create stages
def good_pipeline(data, stages=3):
    result = data
    for i in range(stages):
        reg = pyrtl.Register(len(data), name=f'pipe_{i}')
        reg.next <<= result
        result = reg
    return result
```

### Bitwidth Handling

```python
# IMPORTANT: Be explicit about bitwidths
a = pyrtl.Input(8, 'a')
b = pyrtl.Input(8, 'b')

# This creates a 9-bit result (expanded automatically)
sum_ab = a + b
assert len(sum_ab) == 9

# Explicitly truncate if needed
truncated = sum_ab.truncate(8)
assert len(truncated) == 8

# Sign-extend before operations on signed numbers
signed_a = a.sign_extended(9)
signed_b = b.sign_extended(9)
result = signed_a + signed_b  # Now a true signed operation
```

### Wire Reuse

```python
# CORRECT: Wires can appear in multiple operations
a = pyrtl.Input(8, 'a')
b = a + 1
c = a & 0x0F
d = a >> 2
# 'a' is reused without issue

# WRONG: Don't reassign input wires
a <<= something  # ERROR: Can't assign to Input
```

### Memory Initialization

```python
# Initialize memory with values
mem = pyrtl.MemBlock(32, 8, name='mem')
init_vals = {i: i*2 for i in range(256)}

# Only works during simulation
sim = pyrtl.Simulation(memory_value_map={mem: init_vals})
```

---

## Helpful Helper Functions

```python
import math

# Get log2 of a number
log_value = math.ceil(math.log2(num_values))

# Match bitwidths for operations
wire_a, wire_b = pyrtl.match_bitwidth(wire_a, wire_b)

# Sign extension for signed arithmetic
from pyrtl import signed_add, signed_sub, signed_lt, signed_le

# Concatenation helper
from pyrtl import concat, concat_list
result = concat_list([msb, mid, lsb])

# Mux helper for multi-input selection
from pyrtl import mux
result = pyrtl.select(selector, falsecase=default, 
                      {0: val0, 1: val1, 2: val2})

# Population count (number of 1 bits)
num_ones = pyrtl.popcount(wire)

# Reduction operations
all_and = pyrtl.rtl_all(wire)      # AND all bits
any_or = pyrtl.rtl_any(wire)       # OR all bits
parity = pyrtl.rtl_xor(wire)       # XOR all bits
```

---

## References and Resources

- **Official Documentation**: https://pyrtl.readthedocs.io/
- **GitHub Repository**: https://github.com/UCSBarchlab/PyRTL
- **Example Circuits**: https://github.com/UCSBarchlab/PyRTL/tree/development/examples
- **Installation**: `pip install pyrtl`

---

## Quick Start Template

```python
import pyrtl

# Reset for clean start
pyrtl.reset_working_block()

# Define inputs
input_a = pyrtl.Input(32, 'input_a')
input_b = pyrtl.Input(32, 'input_b')
enable = pyrtl.Input(1, 'enable')

# Define outputs
output = pyrtl.Output(33, 'output')

# Define internal signals
result = input_a + input_b

# Connect to outputs
output <<= result

# Simulate
sim = pyrtl.Simulation()
sim.step_multiple({
    'input_a': [1, 2, 3, 4, 5],
    'input_b': [5, 4, 3, 2, 1],
    'enable': [1, 1, 1, 1, 1]
})

# View results
sim.tracer.render_trace()

# Export to Verilog
with open('output.v', 'w') as f:
    pyrtl.output_to_verilog(f)
```

---

**Last Updated**: December 2025  
**PyRTL Version**: Latest  
**Use Case**: AI Accelerators, Processors, and Complex Hardware Design
