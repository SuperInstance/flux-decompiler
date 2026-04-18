"""
Comprehensive tests for FLUX Decompiler.

Covers:
- Every opcode decompilation (HALT, NOP, INC, DEC, NOT, NEG, PUSH, POP,
  STRIPCONF, MOVI, ADDI, SUBI, ADD, SUB, MUL, DIV, MOD, AND, OR, XOR,
  SHL, SHR, MIN, MAX, CMP_EQ, CMP_LT, CMP_GT, CMP_NE, MOV, JZ, JNZ,
  MOVI16, JMP, LOOP)
- Unknown opcode handling
- Jump labels and target resolution
- Control flow type detection
- Output formats (to_asm, to_annotated)
- Mnemonic frequency counts
- Stats accuracy
- Signed/unsigned immediate handling
- Round-trip: decompile produces correct assembly for known programs
- Edge cases (empty bytecode, single-byte program)
"""
import pytest
from decompiler import (
    FluxDecompiler, DecodedInstruction, DecompilationResult,
    JumpType, OP_SPECS
)


# ── Basic Opcodes ──────────────────────────────────────

class TestHalt:
    def test_halt_single_byte(self):
        d = FluxDecompiler([0x00])
        r = d.decompile()
        assert r.total_instructions == 1
        assert r.total_bytes == 1
        assert r.instructions[0].mnemonic == "HALT"
        assert r.instructions[0].offset == 0
        assert r.instructions[0].size == 1
        assert r.instructions[0].raw_bytes == [0x00]
        assert r.instructions[0].operands == []


class TestNop:
    def test_nop_single_byte(self):
        d = FluxDecompiler([0x01])
        r = d.decompile()
        assert r.instructions[0].mnemonic == "NOP"
        assert r.instructions[0].size == 1

    def test_nop_in_sequence(self):
        d = FluxDecompiler([0x01, 0x01, 0x00])
        r = d.decompile()
        nops = [i for i in r.instructions if i.mnemonic == "NOP"]
        assert len(nops) == 2


# ── Single-Register Opcodes ───────────────────────────

class TestInc:
    def test_inc(self):
        d = FluxDecompiler([0x08, 0, 0x00])
        r = d.decompile()
        assert r.instructions[0].mnemonic == "INC"
        assert r.instructions[0].operands == ["R0"]
        assert r.instructions[0].size == 2

    def test_inc_different_register(self):
        d = FluxDecompiler([0x08, 5, 0x00])
        r = d.decompile()
        assert r.instructions[0].operands == ["R5"]


class TestDec:
    def test_dec(self):
        d = FluxDecompiler([0x09, 0, 0x00])
        r = d.decompile()
        assert r.instructions[0].mnemonic == "DEC"
        assert r.instructions[0].operands == ["R0"]


class TestNot:
    def test_not(self):
        d = FluxDecompiler([0x0A, 0, 0x00])
        r = d.decompile()
        assert r.instructions[0].mnemonic == "NOT"
        assert r.instructions[0].operands == ["R0"]


class TestNeg:
    def test_neg(self):
        d = FluxDecompiler([0x0B, 0, 0x00])
        r = d.decompile()
        assert r.instructions[0].mnemonic == "NEG"


class TestPush:
    def test_push(self):
        d = FluxDecompiler([0x0C, 0, 0x00])
        r = d.decompile()
        assert r.instructions[0].mnemonic == "PUSH"
        assert r.instructions[0].operands == ["R0"]


class TestPop:
    def test_pop(self):
        d = FluxDecompiler([0x0D, 0, 0x00])
        r = d.decompile()
        assert r.instructions[0].mnemonic == "POP"
        assert r.instructions[0].operands == ["R0"]


class TestSTRIPCONF:
    def test_stripconf(self):
        d = FluxDecompiler([0x17, 0, 0x00])
        r = d.decompile()
        assert r.instructions[0].mnemonic == "STRIPCONF"
        assert r.instructions[0].operands == ["R0"]


# ── Immediate Opcodes ─────────────────────────────────

class TestMOVI:
    def test_movi_positive(self):
        d = FluxDecompiler([0x18, 0, 42, 0x00])
        r = d.decompile()
        inst = r.instructions[0]
        assert inst.mnemonic == "MOVI"
        assert inst.operands == ["R0", "42"]

    def test_movi_zero(self):
        d = FluxDecompiler([0x18, 0, 0, 0x00])
        r = d.decompile()
        assert r.instructions[0].operands == ["R0", "0"]

    def test_movi_negative(self):
        """0xFF should be decoded as -1."""
        d = FluxDecompiler([0x18, 0, 0xFF, 0x00])
        r = d.decompile()
        assert r.instructions[0].operands == ["R0", "-1"]

    def test_movi_max_positive(self):
        d = FluxDecompiler([0x18, 0, 127, 0x00])
        r = d.decompile()
        assert r.instructions[0].operands == ["R0", "127"]

    def test_movi_min_negative(self):
        d = FluxDecompiler([0x18, 0, 128, 0x00])
        r = d.decompile()
        assert r.instructions[0].operands == ["R0", "-128"]

    def test_movi_different_register(self):
        d = FluxDecompiler([0x18, 7, 99, 0x00])
        r = d.decompile()
        assert r.instructions[0].operands == ["R7", "99"]


class TestADDI:
    def test_addi_positive(self):
        d = FluxDecompiler([0x19, 0, 5, 0x00])
        r = d.decompile()
        assert r.instructions[0].mnemonic == "ADDI"
        assert r.instructions[0].operands == ["R0", "5"]

    def test_addi_negative(self):
        d = FluxDecompiler([0x19, 0, 0xFF, 0x00])
        r = d.decompile()
        assert r.instructions[0].operands == ["R0", "-1"]


class TestSUBI:
    def test_subi(self):
        d = FluxDecompiler([0x1A, 0, 10, 0x00])
        r = d.decompile()
        assert r.instructions[0].mnemonic == "SUBI"
        assert r.instructions[0].operands == ["R0", "10"]

    def test_subi_negative(self):
        d = FluxDecompiler([0x1A, 0, 0xFC, 0x00])
        r = d.decompile()
        assert r.instructions[0].operands == ["R0", "-4"]


# ── Three-Register ALU Opcodes ─────────────────────────

class TestALUThreeReg:
    """Tests for ADD, SUB, MUL, DIV, MOD, AND, OR, XOR, SHL, SHR, MIN, MAX, CMP_EQ, CMP_LT, CMP_GT, CMP_NE, MOV."""

    @pytest.mark.parametrize("opcode,mnemonic", [
        (0x20, "ADD"), (0x21, "SUB"), (0x22, "MUL"), (0x23, "DIV"),
        (0x24, "MOD"), (0x25, "AND"), (0x26, "OR"), (0x27, "XOR"),
        (0x28, "SHL"), (0x29, "SHR"), (0x2A, "MIN"), (0x2B, "MAX"),
        (0x2C, "CMP_EQ"), (0x2D, "CMP_LT"), (0x2E, "CMP_GT"), (0x2F, "CMP_NE"),
    ])
    def test_alu_opcode_decodes(self, opcode, mnemonic):
        d = FluxDecompiler([opcode, 0, 1, 2, 0x00])
        r = d.decompile()
        inst = r.instructions[0]
        assert inst.mnemonic == mnemonic
        assert inst.operands == ["R0", "R1", "R2"]
        assert inst.size == 4

    def test_mov_three_reg(self):
        d = FluxDecompiler([0x3A, 0, 1, 2, 0x00])
        r = d.decompile()
        inst = r.instructions[0]
        assert inst.mnemonic == "MOV"
        assert inst.operands == ["R0", "R1", "R2"]


# ── Jump Instructions ─────────────────────────────────

class TestJZ:
    def test_jz_decodes(self):
        d = FluxDecompiler([0x3C, 0, 5, 0, 0x00])
        r = d.decompile()
        inst = r.instructions[0]
        assert inst.mnemonic == "JZ"
        assert inst.operands == ["R0", "5"]
        assert inst.jump_type == JumpType.CONDITIONAL

    def test_jz_creates_label(self):
        d = FluxDecompiler([0x01, 0x3C, 0, 0x04, 0, 0x00])  # NOP; JZ R0, 4 (offset from PC=1)
        r = d.decompile()
        assert len(r.labels) > 0
        # Jump from offset 1 with signed offset 4 -> target 5
        assert 5 in r.labels

    def test_jz_negative_offset(self):
        d = FluxDecompiler([0x18, 0, 5, 0x09, 0, 0x3C, 0, 0xFA, 0, 0x00])
        r = d.decompile()
        jz_inst = [i for i in r.instructions if i.mnemonic == "JZ"][0]
        assert jz_inst.jump_type == JumpType.CONDITIONAL
        assert jz_inst.comment != ""


class TestJNZ:
    def test_jnz_decodes(self):
        d = FluxDecompiler([0x3D, 0, 5, 0, 0x00])
        r = d.decompile()
        inst = r.instructions[0]
        assert inst.mnemonic == "JNZ"
        assert inst.operands == ["R0", "5"]
        assert inst.jump_type == JumpType.CONDITIONAL

    def test_jnz_creates_label(self):
        d = FluxDecompiler([0x01, 0x3D, 0, 0x04, 0, 0x00])
        r = d.decompile()
        assert len(r.labels) > 0

    def test_jnz_comment(self):
        d = FluxDecompiler([0x3D, 3, 0x04, 0, 0x00])
        r = d.decompile()
        inst = r.instructions[0]
        assert "R3" in inst.comment
        assert "!=" in inst.comment


class TestJMP:
    def test_jmp_decodes(self):
        d = FluxDecompiler([0x43, 0, 0x10, 0x00, 0x00])  # JMP offset 0x0010 = 16
        r = d.decompile()
        inst = r.instructions[0]
        assert inst.mnemonic == "JMP"
        assert inst.jump_type == JumpType.UNCONDITIONAL

    def test_jmp_creates_label(self):
        d = FluxDecompiler([0x43, 0, 0x10, 0x00, 0x00])
        r = d.decompile()
        assert len(r.labels) > 0


class TestLOOP:
    def test_loop_decodes(self):
        d = FluxDecompiler([0x46, 0, 0x10, 0x00, 0x00])
        r = d.decompile()
        inst = r.instructions[0]
        assert inst.mnemonic == "LOOP"
        assert inst.jump_type == JumpType.LOOP

    def test_loop_creates_label_at_start(self):
        d = FluxDecompiler([0x46, 0, 0x10, 0x00, 0x00])
        r = d.decompile()
        assert 0 in r.labels  # LOOP labels itself at its offset
        assert "loop_000" in r.labels[0]

    def test_loop_comment(self):
        d = FluxDecompiler([0x46, 5, 0x10, 0x00, 0x00])
        r = d.decompile()
        inst = r.instructions[0]
        assert "R5" in inst.comment


# ── MOVI16 ────────────────────────────────────────────

class TestMOVI16:
    def test_mov_i16_positive(self):
        d = FluxDecompiler([0x40, 0, 0x00, 0x10, 0x00])  # MOVI16 R0, 4096
        r = d.decompile()
        inst = r.instructions[0]
        assert inst.mnemonic == "MOVI16"
        assert inst.operands == ["R0", "4096"]

    def test_mov_i16_negative(self):
        d = FluxDecompiler([0x40, 0, 0x00, 0x80, 0x00])  # MOVI16 R0, -32768
        r = d.decompile()
        inst = r.instructions[0]
        assert inst.operands == ["R0", "-32768"]

    def test_mov_i16_zero(self):
        d = FluxDecompiler([0x40, 0, 0x00, 0x00, 0x00])
        r = d.decompile()
        assert r.instructions[0].operands == ["R0", "0"]


# ── Unknown Opcode ────────────────────────────────────

class TestUnknownOpcode:
    def test_single_unknown(self):
        d = FluxDecompiler([0xFE])
        r = d.decompile()
        assert r.instructions[0].mnemonic == "DATA"
        assert r.instructions[0].comment == "unknown opcode"

    def test_unknown_in_sequence(self):
        d = FluxDecompiler([0x00, 0xFE, 0x00])
        r = d.decompile()
        assert r.total_instructions == 3
        assert r.instructions[1].mnemonic == "DATA"

    def test_unknown_counted(self):
        d = FluxDecompiler([0xFE, 0x00])
        r = d.decompile()
        assert r.mnemonic_counts["DATA"] == 1


# ── DecompilationResult Stats ─────────────────────────

class TestResultStats:
    def test_total_bytes(self):
        bc = [0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        assert r.total_bytes == len(bc)

    def test_total_instructions(self):
        bc = [0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        assert r.total_instructions == 4  # MOVI, MOVI, ADD, HALT

    def test_jump_count_zero(self):
        d = FluxDecompiler([0x18, 0, 42, 0x00])
        r = d.decompile()
        assert r.jump_count == 0

    def test_jump_count_with_jnz(self):
        d = FluxDecompiler([0x18, 0, 5, 0x3D, 0, 3, 0, 0x00])
        r = d.decompile()
        assert r.jump_count == 1

    def test_mnemonic_counts(self):
        bc = [0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        assert r.mnemonic_counts.get("MOVI", 0) == 2
        assert r.mnemonic_counts.get("ADD", 0) == 1
        assert r.mnemonic_counts.get("HALT", 0) == 1


# ── Output Formats ────────────────────────────────────

class TestToAsm:
    def test_asm_contains_mnemonic(self):
        d = FluxDecompiler([0x18, 0, 42, 0x00])
        r = d.decompile()
        asm = r.to_asm()
        assert "MOVI" in asm

    def test_asm_contains_halt(self):
        d = FluxDecompiler([0x18, 0, 42, 0x00])
        r = d.decompile()
        asm = r.to_asm()
        assert "HALT" in asm

    def test_asm_has_offset(self):
        d = FluxDecompiler([0x18, 0, 42, 0x00])
        r = d.decompile()
        asm = r.to_asm()
        assert "   0:" in asm  # first instruction at offset 0

    def test_asm_has_hex_bytes(self):
        d = FluxDecompiler([0x00])
        r = d.decompile()
        asm = r.to_asm()
        assert "00" in asm

    def test_asm_has_register_name(self):
        d = FluxDecompiler([0x18, 5, 42, 0x00])
        r = d.decompile()
        asm = r.to_asm()
        assert "R5" in asm

    def test_asm_with_labels(self):
        d = FluxDecompiler([0x18, 0, 5, 0x3D, 0, 3, 0, 0x00])
        r = d.decompile()
        asm = r.to_asm()
        assert "lbl_" in asm


class TestToAnnotated:
    def test_annotated_has_header(self):
        d = FluxDecompiler([0x18, 0, 42, 0x00])
        r = d.decompile()
        ann = r.to_annotated()
        assert "FLUX Bytecode Decompilation" in ann

    def test_annotated_has_stats(self):
        d = FluxDecompiler([0x18, 0, 42, 0x00])
        r = d.decompile()
        ann = r.to_annotated()
        assert "Stats" in ann

    def test_annotated_stats_content(self):
        bc = [0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        ann = r.to_annotated()
        assert "4 instructions" in ann
        assert "11 bytes" in ann

    def test_annotated_conditional_marker(self):
        d = FluxDecompiler([0x3C, 0, 5, 0, 0x00])
        r = d.decompile()
        ann = r.to_annotated()
        assert "↕" in ann

    def test_annotated_unconditional_marker(self):
        d = FluxDecompiler([0x43, 0, 0x05, 0x00, 0x00])
        r = d.decompile()
        ann = r.to_annotated()
        assert "↓" in ann

    def test_annotated_loop_marker(self):
        d = FluxDecompiler([0x46, 0, 0x05, 0x00, 0x00])
        r = d.decompile()
        ann = r.to_annotated()
        assert "↻" in ann

    def test_annotated_has_comment(self):
        d = FluxDecompiler([0x3D, 3, 0x04, 0, 0x00])
        r = d.decompile()
        ann = r.to_annotated()
        assert "R3" in ann


# ── Labels ────────────────────────────────────────────

class TestLabels:
    def test_labels_dict(self):
        d = FluxDecompiler([0x18, 0, 5, 0x3D, 0, 3, 0, 0x00])
        r = d.decompile()
        assert isinstance(r.labels, dict)
        assert len(r.labels) > 0

    def test_label_format(self):
        d = FluxDecompiler([0x3D, 0, 5, 0, 0x00])
        r = d.decompile()
        for target, label in r.labels.items():
            assert label.startswith("lbl_")

    def test_loop_label_format(self):
        d = FluxDecompiler([0x46, 0, 0x10, 0x00, 0x00])
        r = d.decompile()
        for offset, label in r.labels.items():
            assert "loop_" in label

    def test_no_labels_without_jumps(self):
        d = FluxDecompiler([0x18, 0, 42, 0x00])
        r = d.decompile()
        assert len(r.labels) == 0


# ── Raw Bytes ─────────────────────────────────────────

class TestRawBytes:
    def test_raw_bytes_halt(self):
        d = FluxDecompiler([0x00])
        r = d.decompile()
        assert r.instructions[0].raw_bytes == [0x00]

    def test_raw_bytes_movi(self):
        d = FluxDecompiler([0x18, 0, 42, 0x00])
        r = d.decompile()
        assert r.instructions[0].raw_bytes == [0x18, 0, 42]

    def test_raw_bytes_add(self):
        d = FluxDecompiler([0x20, 2, 0, 1, 0x00])
        r = d.decompile()
        assert r.instructions[0].raw_bytes == [0x20, 2, 0, 1]


# ── Jump Target Tracking ──────────────────────────────

class TestJumpTarget:
    def test_jnz_has_target(self):
        d = FluxDecompiler([0x3D, 0, 5, 0, 0x00])
        r = d.decompile()
        assert r.instructions[0].jump_target is not None

    def test_non_jump_no_target(self):
        d = FluxDecompiler([0x18, 0, 42, 0x00])
        r = d.decompile()
        assert r.instructions[0].jump_target is None

    def test_jnz_jump_type(self):
        d = FluxDecompiler([0x3D, 0, 5, 0, 0x00])
        r = d.decompile()
        assert r.instructions[0].jump_type == JumpType.CONDITIONAL

    def test_jz_jump_type(self):
        d = FluxDecompiler([0x3C, 0, 5, 0, 0x00])
        r = d.decompile()
        assert r.instructions[0].jump_type == JumpType.CONDITIONAL

    def test_jmp_jump_type(self):
        d = FluxDecompiler([0x43, 0, 5, 0, 0x00])
        r = d.decompile()
        assert r.instructions[0].jump_type == JumpType.UNCONDITIONAL

    def test_loop_jump_type(self):
        d = FluxDecompiler([0x46, 0, 5, 0, 0x00])
        r = d.decompile()
        assert r.instructions[0].jump_type == JumpType.LOOP


# ── Signed Imm16 ──────────────────────────────────────

class TestSigned16:
    def test_positive_16bit(self):
        d = FluxDecompiler([0x40, 0, 0x00, 0x00, 0x00])
        r = d.decompile()
        assert r.instructions[0].operands == ["R0", "0"]

    def test_max_positive_16bit(self):
        d = FluxDecompiler([0x40, 0, 0xFF, 0x7F, 0x00])  # 0x7FFF = 32767
        r = d.decompile()
        assert r.instructions[0].operands == ["R0", "32767"]

    def test_min_negative_16bit(self):
        d = FluxDecompiler([0x40, 0, 0x00, 0x80, 0x00])  # 0x8000 = -32768
        r = d.decompile()
        assert r.instructions[0].operands == ["R0", "-32768"]


# ── Edge Cases ────────────────────────────────────────

class TestEdgeCases:
    def test_empty_bytecode(self):
        d = FluxDecompiler([])
        r = d.decompile()
        assert r.total_instructions == 0
        assert r.total_bytes == 0
        assert r.jump_count == 0

    def test_single_byte_program(self):
        d = FluxDecompiler([0x00])
        r = d.decompile()
        assert r.total_instructions == 1
        assert r.total_bytes == 1

    def test_incomplete_instruction(self):
        """MOVI needs 3 bytes but we only provide 1 + partial."""
        d = FluxDecompiler([0x18, 0])  # MOVI without immediate
        r = d.decompile()
        # Should still decode without crashing
        assert r.total_instructions >= 1
        assert r.instructions[0].mnemonic == "MOVI"

    def test_all_nops(self):
        d = FluxDecompiler([0x01] * 100)
        r = d.decompile()
        assert r.total_instructions == 100
        assert r.total_bytes == 100


# ── Round-Trip: Known Programs ────────────────────────

class TestRoundTrip:
    def test_simple_add_program(self):
        """MOVI R0, 10; MOVI R1, 20; ADD R2, R0, R1; HALT"""
        bc = [0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        asm = r.to_asm()

        # Verify we can parse out the key instructions from asm output
        # Format: "  offset: hex_bytes MNEMONIC operands"
        mnemonics = []
        for line in asm.strip().split("\n"):
            if ":" in line and "lbl_" not in line:
                parts = line.split(":", 1)[1].strip()
                # Skip hex bytes to get to mnemonic
                tokens = parts.split()
                # Find the mnemonic (first uppercase word after hex bytes)
                for t in tokens:
                    if t.isalpha() and t.isupper():
                        mnemonics.append(t)
                        break
        assert "MOVI" in mnemonics
        assert "ADD" in mnemonics
        assert "HALT" in mnemonics

    def test_factorial_decompile(self):
        """Factorial: MOVI R0, 6; MOVI R1, 1; MUL R1, R1, R0; DEC R0; JNZ R0, -6; HALT"""
        bc = [0x18, 0, 6, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()

        mnemonics = [i.mnemonic for i in r.instructions]
        assert mnemonics.count("MOVI") == 2
        assert "MUL" in mnemonics
        assert "DEC" in mnemonics
        assert "JNZ" in mnemonics
        assert "HALT" in mnemonics
        assert r.jump_count >= 1

    def test_loop_counter(self):
        """Counter loop: MOVI R0, 5; MOVI R1, 0; INC R1; DEC R0; JNZ R0, -3; HALT"""
        bc = [0x18, 0, 5, 0x18, 1, 0, 0x08, 1, 0x09, 0, 0x3D, 0, 0xFC, 0, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()

        mnemonics = [i.mnemonic for i in r.instructions]
        assert "INC" in mnemonics
        assert "DEC" in mnemonics
        assert "JNZ" in mnemonics
        assert len(r.labels) > 0

    def test_push_pop_sequence(self):
        """PUSH R0; PUSH R1; POP R2; POP R3; HALT"""
        bc = [0x18, 0, 10, 0x18, 1, 20, 0x0C, 0, 0x0C, 1, 0x0D, 2, 0x0D, 3, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()

        mnemonics = [i.mnemonic for i in r.instructions]
        assert mnemonics.count("MOVI") == 2
        assert mnemonics.count("PUSH") == 2
        assert mnemonics.count("POP") == 2
        assert mnemonics.count("HALT") == 1

    def test_logical_ops(self):
        """Test AND, OR, XOR decompile."""
        bc = [
            0x18, 0, 0xFF,  # MOVI R0, -1
            0x18, 1, 0x0F,  # MOVI R1, 15
            0x25, 2, 0, 1,  # AND R2, R0, R1
            0x26, 3, 0, 1,  # OR R3, R0, R1
            0x27, 4, 0, 1,  # XOR R4, R0, R1
            0x00             # HALT
        ]
        d = FluxDecompiler(bc)
        r = d.decompile()
        mnemonics = [i.mnemonic for i in r.instructions]
        assert "AND" in mnemonics
        assert "OR" in mnemonics
        assert "XOR" in mnemonics

    def test_comparison_ops(self):
        """Test CMP_EQ, CMP_LT, CMP_GT, CMP_NE decompile."""
        bc = [
            0x18, 0, 10, 0x18, 1, 20,
            0x2C, 2, 0, 1,  # CMP_EQ R2, R0, R1
            0x2D, 3, 0, 1,  # CMP_LT R3, R0, R1
            0x2E, 4, 0, 1,  # CMP_GT R4, R0, R1
            0x2F, 5, 0, 1,  # CMP_NE R5, R0, R1
            0x00
        ]
        d = FluxDecompiler(bc)
        r = d.decompile()
        mnemonics = [i.mnemonic for i in r.instructions]
        assert "CMP_EQ" in mnemonics
        assert "CMP_LT" in mnemonics
        assert "CMP_GT" in mnemonics
        assert "CMP_NE" in mnemonics

    def test_shift_ops(self):
        bc = [
            0x18, 0, 8, 0x18, 1, 1,
            0x28, 2, 0, 1,  # SHL R2, R0, R1
            0x29, 3, 0, 1,  # SHR R3, R0, R1
            0x00
        ]
        d = FluxDecompiler(bc)
        r = d.decompile()
        mnemonics = [i.mnemonic for i in r.instructions]
        assert "SHL" in mnemonics
        assert "SHR" in mnemonics

    def test_min_max_ops(self):
        bc = [
            0x18, 0, 10, 0x18, 1, 20,
            0x2A, 2, 0, 1,  # MIN R2, R0, R1
            0x2B, 3, 0, 1,  # MAX R3, R0, R1
            0x00
        ]
        d = FluxDecompiler(bc)
        r = d.decompile()
        mnemonics = [i.mnemonic for i in r.instructions]
        assert "MIN" in mnemonics
        assert "MAX" in mnemonics

    def test_mod_op(self):
        bc = [0x18, 0, 17, 0x18, 1, 5, 0x24, 2, 0, 1, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        mnemonics = [i.mnemonic for i in r.instructions]
        assert "MOD" in mnemonics


# ── OP_SPECS Coverage ─────────────────────────────────

class TestOpSpecsCoverage:
    def test_all_op_specs_are_decodable(self):
        """Every opcode in OP_SPECS should decompile to the correct mnemonic."""
        for opcode, (mnemonic, size, _) in OP_SPECS.items():
            # Build a minimal bytecode: opcode + padding bytes
            bc = [opcode] + [0x00] * (size + 1)  # extra HALT at end
            d = FluxDecompiler(bc)
            r = d.decompile()
            assert r.instructions[0].mnemonic == mnemonic, \
                f"Opcode {opcode:#x} should decode to {mnemonic}, got {r.instructions[0].mnemonic}"


# ── Multi-Jump Program ────────────────────────────────

class TestMultiJump:
    def test_two_jnz(self):
        bc = [0x18, 0, 5, 0x3D, 0, 3, 0, 0x18, 1, 99, 0x3D, 1, 3, 0, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        assert r.jump_count == 2
        assert len(r.labels) >= 2


# ── Instruction Comments ──────────────────────────────

class TestInstructionComments:
    def test_jz_comment(self):
        d = FluxDecompiler([0x3C, 3, 0x04, 0, 0x00])
        r = d.decompile()
        inst = r.instructions[0]
        assert "R3" in inst.comment
        assert "==" in inst.comment
        assert "goto" in inst.comment

    def test_loop_comment(self):
        d = FluxDecompiler([0x46, 7, 0x10, 0x00, 0x00])
        r = d.decompile()
        inst = r.instructions[0]
        assert "R7" in inst.comment
        assert "decrements" in inst.comment

    def test_no_comment_for_non_jump(self):
        d = FluxDecompiler([0x18, 0, 42, 0x00])
        r = d.decompile()
        inst = r.instructions[0]
        assert inst.comment == ""


# ── Mnemonic Frequency ────────────────────────────────

class TestMnemonicFrequency:
    def test_top_5_in_annotated(self):
        bc = [0x18, 0, 10, 0x18, 1, 20, 0x18, 2, 30, 0x20, 3, 0, 1, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        ann = r.to_annotated()
        assert "MOVI: 3x" in ann
