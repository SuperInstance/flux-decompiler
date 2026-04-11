"""
FLUX Decompiler — convert bytecode back to human-readable assembly.

Supports:
- Full disassembly with PC offsets
- Symbolic register names
- Jump target resolution
- Comment annotation
- Control flow reconstruction
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class JumpType(Enum):
    CONDITIONAL = "conditional"
    UNCONDITIONAL = "unconditional"
    LOOP = "loop"


@dataclass
class DecodedInstruction:
    offset: int
    opcode: int
    mnemonic: str
    operands: List[str]
    raw_bytes: List[int]
    size: int
    jump_target: Optional[int] = None
    jump_type: Optional[JumpType] = None
    comment: str = ""


@dataclass
class DecompilationResult:
    instructions: List[DecodedInstruction]
    labels: Dict[int, str]
    total_bytes: int
    total_instructions: int
    jump_count: int
    mnemonic_counts: Dict[str, int]
    
    def to_asm(self) -> str:
        """Generate clean assembly output."""
        lines = []
        for inst in self.instructions:
            label = self.labels.get(inst.offset)
            if label:
                lines.append(f"{label}:")
            
            op_str = ", ".join(inst.operands)
            hex_bytes = " ".join(f"{b:02x}" for b in inst.raw_bytes)
            comment = f"  ; {inst.comment}" if inst.comment else ""
            lines.append(f"  {inst.offset:4d}: {hex_bytes:12s} {inst.mnemonic:10s} {op_str}{comment}")
        
        return "\n".join(lines)
    
    def to_annotated(self) -> str:
        """Generate annotated output with control flow markers."""
        lines = ["; FLUX Bytecode Decompilation\n"]
        for inst in self.instructions:
            label = self.labels.get(inst.offset)
            if label:
                lines.append(f"{label}:")
            
            marker = ""
            if inst.jump_type == JumpType.CONDITIONAL:
                marker = "↕ "
            elif inst.jump_type == JumpType.UNCONDITIONAL:
                marker = "↓ "
            elif inst.jump_type == JumpType.LOOP:
                marker = "↻ "
            
            op_str = ", ".join(inst.operands)
            comment = f"  ; {inst.comment}" if inst.comment else ""
            lines.append(f"  {marker}{inst.offset:4d}: {inst.mnemonic:10s} {op_str}{comment}")
        
        # Add stats
        lines.append(f"\n; --- Stats ---")
        lines.append(f"; {self.total_instructions} instructions, {self.total_bytes} bytes")
        lines.append(f"; {self.jump_count} jumps, {len(self.labels)} labels")
        for mn, cnt in sorted(self.mnemonic_counts.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"; {mn}: {cnt}x")
        
        return "\n".join(lines)


OP_SPECS = {
    0x00: ("HALT", 1, []), 0x01: ("NOP", 1, []),
    0x08: ("INC", 2, ["reg"]), 0x09: ("DEC", 2, ["reg"]),
    0x0A: ("NOT", 2, ["reg"]), 0x0B: ("NEG", 2, ["reg"]),
    0x0C: ("PUSH", 2, ["reg"]), 0x0D: ("POP", 2, ["reg"]),
    0x17: ("STRIPCONF", 2, ["reg"]),
    0x18: ("MOVI", 3, ["reg", "imm8"]), 0x19: ("ADDI", 3, ["reg", "imm8"]),
    0x1A: ("SUBI", 3, ["reg", "imm8"]),
    0x20: ("ADD", 4, ["reg", "reg", "reg"]), 0x21: ("SUB", 4, ["reg", "reg", "reg"]),
    0x22: ("MUL", 4, ["reg", "reg", "reg"]), 0x23: ("DIV", 4, ["reg", "reg", "reg"]),
    0x24: ("MOD", 4, ["reg", "reg", "reg"]), 0x25: ("AND", 4, ["reg", "reg", "reg"]),
    0x26: ("OR", 4, ["reg", "reg", "reg"]), 0x27: ("XOR", 4, ["reg", "reg", "reg"]),
    0x28: ("SHL", 4, ["reg", "reg", "reg"]), 0x29: ("SHR", 4, ["reg", "reg", "reg"]),
    0x2A: ("MIN", 4, ["reg", "reg", "reg"]), 0x2B: ("MAX", 4, ["reg", "reg", "reg"]),
    0x2C: ("CMP_EQ", 4, ["reg", "reg", "reg"]), 0x2D: ("CMP_LT", 4, ["reg", "reg", "reg"]),
    0x2E: ("CMP_GT", 4, ["reg", "reg", "reg"]), 0x2F: ("CMP_NE", 4, ["reg", "reg", "reg"]),
    0x3A: ("MOV", 4, ["reg", "reg", "reg"]), 0x3C: ("JZ", 4, ["reg", "imm8"]),
    0x3D: ("JNZ", 4, ["reg", "imm8"]), 0x40: ("MOVI16", 4, ["reg", "imm16"]),
    0x43: ("JMP", 4, ["imm16"]), 0x46: ("LOOP", 4, ["reg", "imm16"]),
}

JUMP_OPS = {0x3C, 0x3D, 0x43, 0x46}


class FluxDecompiler:
    """Decompile FLUX bytecode to assembly."""
    
    def __init__(self, bytecode: List[int]):
        self.bytecode = bytes(bytecode)
    
    def _signed8(self, b):
        return b - 256 if b > 127 else b
    
    def _signed16(self, lo, hi):
        v = lo | (hi << 8)
        return v - 0x10000 if v > 0x7FFF else v
    
    def decompile(self) -> DecompilationResult:
        instructions = []
        labels = {}
        mnemonic_counts = {}
        jump_count = 0
        i = 0
        
        # Pass 1: Decode all instructions
        while i < len(self.bytecode):
            op = self.bytecode[i]
            spec = OP_SPECS.get(op)
            
            if spec is None:
                instructions.append(DecodedInstruction(
                    offset=i, opcode=op, mnemonic=f"DATA", operands=[f"0x{op:02x}"],
                    raw_bytes=[op], size=1, comment=f"unknown opcode"
                ))
                mnemonic_counts["DATA"] = mnemonic_counts.get("DATA", 0) + 1
                i += 1
                continue
            
            mnemonic, size, operand_types = spec
            raw = list(self.bytecode[i:i+size])
            operands = []
            jump_target = None
            jump_type = None
            comment = ""
            
            for j, otype in enumerate(operand_types):
                byte_idx = i + 1 + j
                val = self.bytecode[byte_idx] if byte_idx < len(self.bytecode) else 0
                
                if otype == "reg":
                    operands.append(f"R{val}")
                elif otype == "imm8":
                    signed_val = self._signed8(val)
                    operands.append(str(signed_val))
                elif otype == "imm16":
                    if byte_idx + 1 < len(self.bytecode):
                        signed_val = self._signed16(val, self.bytecode[byte_idx + 1])
                        operands.append(str(signed_val))
                        val = signed_val
                    else:
                        operands.append(str(val))
            
            # Handle jumps
            if op in JUMP_OPS:
                jump_count += 1
                if op == 0x3C:  # JZ
                    offset = self._signed8(raw[2]) if len(raw) > 2 else 0
                    target = i + offset
                    jump_target = target
                    jump_type = JumpType.CONDITIONAL
                    labels[target] = f"lbl_{target:03d}"
                    comment = f"if R{raw[1]}==0 goto lbl_{target:03d}"
                elif op == 0x3D:  # JNZ
                    offset = self._signed8(raw[2]) if len(raw) > 2 else 0
                    target = i + offset
                    jump_target = target
                    jump_type = JumpType.CONDITIONAL
                    labels[target] = f"lbl_{target:03d}"
                    comment = f"if R{raw[1]}!=0 goto lbl_{target:03d}"
                elif op == 0x43:  # JMP
                    offset = self._signed16(raw[2], raw[3]) if len(raw) > 3 else 0
                    target = i + offset
                    jump_target = target
                    jump_type = JumpType.UNCONDITIONAL
                    labels[target] = f"lbl_{target:03d}"
                elif op == 0x46:  # LOOP
                    labels[i] = f"loop_{i:03d}"
                    jump_type = JumpType.LOOP
                    comment = f"R{raw[1]} decrements"
            
            mnemonic_counts[mnemonic] = mnemonic_counts.get(mnemonic, 0) + 1
            
            instructions.append(DecodedInstruction(
                offset=i, opcode=op, mnemonic=mnemonic, operands=operands,
                raw_bytes=raw, size=size, jump_target=jump_target,
                jump_type=jump_type, comment=comment
            ))
            
            i += size
        
        return DecompilationResult(
            instructions=instructions, labels=labels,
            total_bytes=len(self.bytecode), total_instructions=len(instructions),
            jump_count=jump_count, mnemonic_counts=mnemonic_counts
        )


# ── Tests ──────────────────────────────────────────────

import unittest


class TestDecompiler(unittest.TestCase):
    def test_halt(self):
        d = FluxDecompiler([0x00])
        r = d.decompile()
        self.assertEqual(r.total_instructions, 1)
        self.assertEqual(r.instructions[0].mnemonic, "HALT")
    
    def test_movi(self):
        d = FluxDecompiler([0x18, 0, 42, 0x00])
        r = d.decompile()
        self.assertEqual(r.instructions[0].mnemonic, "MOVI")
        self.assertIn("R0", r.instructions[0].operands)
        self.assertIn("42", r.instructions[0].operands)
    
    def test_add(self):
        d = FluxDecompiler([0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00])
        r = d.decompile()
        adds = [i for i in r.instructions if i.mnemonic == "ADD"]
        self.assertEqual(len(adds), 1)
    
    def test_jump_labels(self):
        # counter loop
        d = FluxDecompiler([0x18, 0, 5, 0x18, 1, 0, 0x08, 1, 0x09, 0, 0x3D, 0, 0xFC, 0, 0x00])
        r = d.decompile()
        self.assertGreater(len(r.labels), 0)
        self.assertGreater(r.jump_count, 0)
    
    def test_asm_output(self):
        d = FluxDecompiler([0x18, 0, 42, 0x00])
        r = d.decompile()
        asm = r.to_asm()
        self.assertIn("MOVI", asm)
        self.assertIn("HALT", asm)
    
    def test_annotated_output(self):
        d = FluxDecompiler([0x18, 0, 42, 0x00])
        r = d.decompile()
        ann = r.to_annotated()
        self.assertIn("Stats", ann)
    
    def test_mnemonic_counts(self):
        d = FluxDecompiler([0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00])
        r = d.decompile()
        self.assertEqual(r.mnemonic_counts.get("MOVI", 0), 2)
        self.assertEqual(r.mnemonic_counts.get("ADD", 0), 1)
    
    def test_factorial(self):
        bc = [0x18, 0, 6, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        self.assertIn("MUL", [i.mnemonic for i in r.instructions])
        self.assertIn("JNZ", [i.mnemonic for i in r.instructions])
    
    def test_unknown_opcode(self):
        d = FluxDecompiler([0xFE, 0x00])
        r = d.decompile()
        self.assertEqual(r.instructions[0].mnemonic, "DATA")
    
    def test_full_program(self):
        bc = [0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        self.assertEqual(r.total_bytes, 11)
        self.assertEqual(r.total_instructions, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
