import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles


@cocotb.test()
async def test_counter(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    dut.rst.value = 1
    await ClockCycles(dut.clk, 1)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)
    
    await ClockCycles(dut.clk, 20)