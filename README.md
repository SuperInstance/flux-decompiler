# flux-decompiler

> Bytecode-to-assembly decompiler with jump target resolution, control flow annotation, and label generation for FLUX programs.

## What This Is

`flux-decompiler` is a Python module that **converts FLUX bytecode back to human-readable assembly** — it decodes all 30+ opcodes, resolves jump targets into named labels, annotates control flow (↕ conditional, ↓ unconditional, ↻ loop), and produces both clean and annotated assembly output.

## Role in the FLUX Ecosystem

The decompiler bridges the gap between raw bytes and understanding:

- **`flux-disasm`** (C) provides a lightweight C disassembler; decompiler adds Python convenience and annotations
- **`flux-signatures`** uses decoded instructions for pattern analysis
- **`flux-timeline`** references decoded mnemonics in execution traces
- **`flux-profiler`** maps opcode counts to human-readable names
- **`flux-debugger`** shows current instruction names during stepping
- **`flux-coverage`** identifies which decoded instructions were hit

## Key Features

| Feature | Description |
|---------|-------------|
| **Full ISA Coverage** | 30+ opcodes: arithmetic, logical, comparison, jump, stack, move |
| **Jump Resolution** | JZ/JNZ/JMP/LOOP targets resolved to named labels |
| **Control Flow Markers** | ↕ conditional, ↓ unconditional, ↻ loop annotations |
| **Two Output Formats** | Clean `to_asm()` and annotated `to_annotated()` with stats |
| **Mnemonic Statistics** | Counts per instruction type in annotated output |
| **Signed Operand Handling** | Correct sign extension for imm8 and imm16 values |
| **Unknown Opcode Handling** | Gracefully marks unknown bytes as DATA |

## Quick Start

```python
from flux_decompiler import FluxDecompiler

# Decompile a factorial program
bytecode = [0x18, 0, 6, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, -6, 0, 0x00]
dec = FluxDecompiler(bytecode)
result = dec.decompile()

print(f"Instructions: {result.total_instructions}, Bytes: {result.total_bytes}")
print(f"Jumps: {result.jump_count}, Labels: {len(result.labels)}")

# Clean assembly output
print(result.to_asm())

# Annotated output with stats and control flow markers
print(result.to_annotated())
```

## Running Tests

```bash
python -m pytest tests/ -v
# or
python decompiler.py
```

## Related Fleet Repos

- [`flux-disasm`](https://github.com/SuperInstance/flux-disasm) — C disassembler (lightweight alternative)
- [`flux-signatures`](https://github.com/SuperInstance/flux-signatures) — Pattern detection
- [`flux-timeline`](https://github.com/SuperInstance/flux-timeline) — Execution tracing
- [`flux-profiler`](https://github.com/SuperInstance/flux-profiler) — Performance profiling
- [`flux-debugger`](https://github.com/SuperInstance/flux-debugger) — Step debugger

## License

Part of the [SuperInstance](https://github.com/SuperInstance) FLUX fleet.
