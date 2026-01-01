import sys
import shutil
import re
import os
from pathlib import Path
from cocotb_test.simulator import run

# Usage: 
# Basic run:
# python3 run_test.py peripherals/counter

# Run specific test:
# python3 run_test.py peripherals/counter test_counter

# Run with parameters:
# python3 run_test.py peripherals/counter test_counter WIDTH=32 DEPTH=16

# One parameter only:
# python3 run_test.py peripherals/counter WIDTH=8

def find_dependencies(rtl_files, rtl_base_dir):
    all_sources = set(rtl_files)
    to_process = list(rtl_files)
    processed = set()

    module_map = {}
    for rtl_dir in rtl_base_dir.rglob("*.sv"):
        content = rtl_dir.read_text()
        matches = re.findall(r"^\s*module\s+(\w+)", content, re.MULTILINE)
        for mod in matches:
            module_map[mod] = rtl_dir
    
    for rtl_dir in rtl_base_dir.rglob("*.v"):
        content = rtl_dir.read_text()
        matches = re.findall(r"^\s*module\s+(\w+)", content, re.MULTILINE)
        for mod in matches:
            module_map[mod] = rtl_dir
    
    while to_process:
        current_file = to_process.pop()
        if current_file in processed:
            continue
        processed.add(current_file)

        try:
            content = current_file.read_text()

            instantiations = re.findall(r"^\s*(\w+)\s+\w+\s*\(", content, re.MULTILINE)
            includes = re.findall(r'^\s*`include\s+"([^"]+)"', content)

            for inst in instantiations:
                if inst in module_map:
                    dep_file = module_map[inst]
                    if dep_file not in all_sources:
                        all_sources.add(dep_file)
                        to_process.append(dep_file)
            
            for inc in includes:
                inc_path = current_file.parent / inc
                if inc_path.exists() and inc_path not in all_sources:
                    all_sources.add(inc_path)
                    to_process.append(inc_path)
        
        except Exception as e:
            print(f"Warning: Could not parse {current_file}: {e}")
    
    return sorted(all_sources)

def run_test(module_path, test_name=None, waves=True, params=None):
    parts = module_path.split("/")
    if len(parts) < 2:
        print(f"Error: module_path must be 'category/module', got {module_path}")
        sys.exit(1)
    
    category, module = parts

    rtl_dir = Path("..") / "rtl" / category
    rtl_base = Path("..") / "rtl"
    tb_file = Path(f"test_{module}.py")
    work_dir = Path(".")

    if not tb_file.exists():
        print(f"Error: Testbench file not found: {tb_file}")
        sys.exit(1)
    
    verilog_sources = []
    if rtl_dir.exists():
        verilog_sources.extend(rtl_dir.rglob("*.v"))
        verilog_sources.extend(rtl_dir.rglob("*.sv"))
    
    if not verilog_sources:
        print(f"Warning: No RTL sources found in {rtl_dir}")
    print(f"Finding dependencies for {module}...")
    all_sources = find_dependencies(verilog_sources, rtl_base)
    for src in all_sources:
        print(f"  - {src}")

    extra_args = []
    plus_args = []
    if waves:
        extra_args.extend(["--trace", "--trace-structs", "--trace-fst"])
        plus_args.extend(["--trace"])

    run(
        verilog_sources=[str(src) for src in all_sources],
        toplevel=module,
        module=f"test_{module}",
        simulator="verilator",
        work_dir=str(work_dir),
        waves=waves,
        extra_args=extra_args,
        plus_args=plus_args,
        testcase=test_name,
        parameters=params or {},
        python_search=[str(Path("."))],
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_cocotb_test.py <module_path> [test_name] [param=value ...]")
        sys.exit(1)
    
    module_path = sys.argv[1]
    test_name = sys.argv[2] if len(sys.argv) > 2 and "=" not in sys.argv[2] else None
    
    # add more parameters here if not by command line
    params = {}
     
    start_idx = 3 if test_name else 2
    for arg in sys.argv[start_idx:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            try:
                params[key] = int(value)
            except ValueError:
                params[key] = value

    if params:
        print(f"Parameters: {params}")

    run_test(module_path, test_name, params=params)