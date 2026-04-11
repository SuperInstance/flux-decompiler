# FLUX Decompiler — Bytecode to Assembly

Convert FLUX bytecode back to human-readable assembly with labels, jump targets, and stats.

## Features
- Instruction-level disassembly with PC offsets
- Automatic label generation for jump targets
- Control flow markers (↕ conditional, ↓ unconditional, ↻ loop)
- Mnemonic frequency stats
- Two output formats: clean assembly and annotated

## Usage
```python
from decompiler import FluxDecompiler
d = FluxDecompiler([0x18, 0, 42, 0x00])
result = d.decompile()
print(result.to_asm())
print(result.to_annotated())
```

10 tests passing.
