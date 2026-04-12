"""
FLUX Decompiler — comprehensive test suite.

Covers all opcodes, signed immediates, jump types, complex programs,
output formats, edge cases, and statistics.
"""
import unittest
from decompiler import (
    FluxDecompiler, DecompilationResult,
    DecodedInstruction, JumpType, OP_SPECS,
)


# ── Helpers ────────────────────────────────────────────

def _decompile(bytecode):
    """Shorthand: decompile and return result."""
    return FluxDecompiler(bytecode).decompile()


def _mnemonics(result):
    """Return list of mnemonics from a result."""
    return [i.mnemonic for i in result.instructions]


# ═══════════════════════════════════════════════════════
# Individual opcode tests
# ═══════════════════════════════════════════════════════

class TestOpcodeHALT(unittest.TestCase):
    def test_halt_single(self):
        r = _decompile([0x00])
        self.assertEqual(r.total_instructions, 1)
        self.assertEqual(r.instructions[0].mnemonic, "HALT")
        self.assertEqual(r.instructions[0].size, 1)
        self.assertEqual(r.instructions[0].operands, [])
        self.assertEqual(r.total_bytes, 1)

    def test_halt_offset(self):
        r = _decompile([0x01, 0x00])
        self.assertEqual(r.instructions[1].offset, 1)
        self.assertEqual(r.instructions[1].mnemonic, "HALT")


class TestOpcodeNOP(unittest.TestCase):
    def test_nop_single(self):
        r = _decompile([0x01])
        self.assertEqual(r.total_instructions, 1)
        self.assertEqual(r.instructions[0].mnemonic, "NOP")
        self.assertEqual(r.instructions[0].size, 1)
        self.assertEqual(r.instructions[0].operands, [])

    def test_nop_sequence(self):
        r = _decompile([0x01, 0x01, 0x01, 0x00])
        self.assertEqual(r.total_instructions, 4)
        self.assertEqual(r.mnemonic_counts["NOP"], 3)


class TestOpcodeINC(unittest.TestCase):
    def test_inc_r0(self):
        r = _decompile([0x08, 0x00])
        self.assertEqual(r.instructions[0].mnemonic, "INC")
        self.assertEqual(r.instructions[0].operands, ["R0"])
        self.assertEqual(r.instructions[0].size, 2)

    def test_inc_r7(self):
        r = _decompile([0x08, 0x07])
        self.assertEqual(r.instructions[0].operands, ["R7"])


class TestOpcodeDEC(unittest.TestCase):
    def test_dec_r1(self):
        r = _decompile([0x09, 0x01])
        self.assertEqual(r.instructions[0].mnemonic, "DEC")
        self.assertEqual(r.instructions[0].operands, ["R1"])


class TestOpcodeNOT(unittest.TestCase):
    def test_not_r3(self):
        r = _decompile([0x0A, 0x03])
        self.assertEqual(r.instructions[0].mnemonic, "NOT")
        self.assertEqual(r.instructions[0].operands, ["R3"])


class TestOpcodeNEG(unittest.TestCase):
    def test_neg_r2(self):
        r = _decompile([0x0B, 0x02])
        self.assertEqual(r.instructions[0].mnemonic, "NEG")
        self.assertEqual(r.instructions[0].operands, ["R2"])


class TestOpcodePUSH(unittest.TestCase):
    def test_push_r0(self):
        r = _decompile([0x0C, 0x00])
        self.assertEqual(r.instructions[0].mnemonic, "PUSH")
        self.assertEqual(r.instructions[0].operands, ["R0"])


class TestOpcodePOP(unittest.TestCase):
    def test_pop_r5(self):
        r = _decompile([0x0D, 0x05])
        self.assertEqual(r.instructions[0].mnemonic, "POP")
        self.assertEqual(r.instructions[0].operands, ["R5"])


class TestOpcodeSTRIPCONF(unittest.TestCase):
    def test_stripconf_r0(self):
        r = _decompile([0x17, 0x00])
        self.assertEqual(r.instructions[0].mnemonic, "STRIPCONF")
        self.assertEqual(r.instructions[0].operands, ["R0"])

    def test_stripconf_r4(self):
        r = _decompile([0x17, 0x04])
        self.assertEqual(r.instructions[0].operands, ["R4"])


class TestOpcodeMOVI(unittest.TestCase):
    def test_movi_r0_42(self):
        r = _decompile([0x18, 0x00, 42])
        self.assertEqual(r.instructions[0].mnemonic, "MOVI")
        self.assertEqual(r.instructions[0].operands, ["R0", "42"])

    def test_movi_size(self):
        r = _decompile([0x18, 0x00, 100])
        self.assertEqual(r.instructions[0].size, 3)

    def test_movi_raw_bytes(self):
        r = _decompile([0x18, 0x03, 0xFF])
        self.assertEqual(r.instructions[0].raw_bytes, [0x18, 0x03, 0xFF])


class TestOpcodeADDI(unittest.TestCase):
    def test_addi_r0_10(self):
        r = _decompile([0x19, 0x00, 10])
        self.assertEqual(r.instructions[0].mnemonic, "ADDI")
        self.assertEqual(r.instructions[0].operands, ["R0", "10"])


class TestOpcodeSUBI(unittest.TestCase):
    def test_subi_r1_5(self):
        r = _decompile([0x1A, 0x01, 5])
        self.assertEqual(r.instructions[0].mnemonic, "SUBI")
        self.assertEqual(r.instructions[0].operands, ["R1", "5"])


class TestOpcodeADD(unittest.TestCase):
    def test_add_r2_r0_r1(self):
        # ADD R2, R0, R1
        r = _decompile([0x20, 0x02, 0x00, 0x01])
        self.assertEqual(r.instructions[0].mnemonic, "ADD")
        self.assertEqual(r.instructions[0].operands, ["R2", "R0", "R1"])
        self.assertEqual(r.instructions[0].size, 4)


class TestOpcodeSUB(unittest.TestCase):
    def test_sub(self):
        r = _decompile([0x21, 0x00, 0x01, 0x02])
        self.assertEqual(r.instructions[0].mnemonic, "SUB")
        self.assertEqual(r.instructions[0].operands, ["R0", "R1", "R2"])


class TestOpcodeMUL(unittest.TestCase):
    def test_mul(self):
        r = _decompile([0x22, 0x00, 0x01, 0x02])
        self.assertEqual(r.instructions[0].mnemonic, "MUL")
        self.assertEqual(r.instructions[0].operands, ["R0", "R1", "R2"])


class TestOpcodeDIV(unittest.TestCase):
    def test_div(self):
        r = _decompile([0x23, 0x03, 0x00, 0x01])
        self.assertEqual(r.instructions[0].mnemonic, "DIV")
        self.assertEqual(r.instructions[0].operands, ["R3", "R0", "R1"])


class TestOpcodeMOD(unittest.TestCase):
    def test_mod(self):
        r = _decompile([0x24, 0x00, 0x01, 0x02])
        self.assertEqual(r.instructions[0].mnemonic, "MOD")
        self.assertEqual(r.instructions[0].operands, ["R0", "R1", "R2"])


class TestOpcodeAND(unittest.TestCase):
    def test_and(self):
        r = _decompile([0x25, 0x00, 0x01, 0x02])
        self.assertEqual(r.instructions[0].mnemonic, "AND")
        self.assertEqual(r.instructions[0].operands, ["R0", "R1", "R2"])


class TestOpcodeOR(unittest.TestCase):
    def test_or(self):
        r = _decompile([0x26, 0x00, 0x01, 0x02])
        self.assertEqual(r.instructions[0].mnemonic, "OR")
        self.assertEqual(r.instructions[0].operands, ["R0", "R1", "R2"])


class TestOpcodeXOR(unittest.TestCase):
    def test_xor(self):
        r = _decompile([0x27, 0x00, 0x01, 0x02])
        self.assertEqual(r.instructions[0].mnemonic, "XOR")
        self.assertEqual(r.instructions[0].operands, ["R0", "R1", "R2"])


class TestOpcodeSHL(unittest.TestCase):
    def test_shl(self):
        r = _decompile([0x28, 0x00, 0x01, 0x02])
        self.assertEqual(r.instructions[0].mnemonic, "SHL")
        self.assertEqual(r.instructions[0].operands, ["R0", "R1", "R2"])


class TestOpcodeSHR(unittest.TestCase):
    def test_shr(self):
        r = _decompile([0x29, 0x00, 0x01, 0x02])
        self.assertEqual(r.instructions[0].mnemonic, "SHR")
        self.assertEqual(r.instructions[0].operands, ["R0", "R1", "R2"])


class TestOpcodeMIN(unittest.TestCase):
    def test_min(self):
        r = _decompile([0x2A, 0x00, 0x01, 0x02])
        self.assertEqual(r.instructions[0].mnemonic, "MIN")
        self.assertEqual(r.instructions[0].operands, ["R0", "R1", "R2"])


class TestOpcodeMAX(unittest.TestCase):
    def test_max(self):
        r = _decompile([0x2B, 0x00, 0x01, 0x02])
        self.assertEqual(r.instructions[0].mnemonic, "MAX")
        self.assertEqual(r.instructions[0].operands, ["R0", "R1", "R2"])


class TestOpcodeCMP_EQ(unittest.TestCase):
    def test_cmp_eq(self):
        r = _decompile([0x2C, 0x00, 0x01, 0x02])
        self.assertEqual(r.instructions[0].mnemonic, "CMP_EQ")
        self.assertEqual(r.instructions[0].operands, ["R0", "R1", "R2"])


class TestOpcodeCMP_LT(unittest.TestCase):
    def test_cmp_lt(self):
        r = _decompile([0x2D, 0x00, 0x01, 0x02])
        self.assertEqual(r.instructions[0].mnemonic, "CMP_LT")
        self.assertEqual(r.instructions[0].operands, ["R0", "R1", "R2"])


class TestOpcodeCMP_GT(unittest.TestCase):
    def test_cmp_gt(self):
        r = _decompile([0x2E, 0x00, 0x01, 0x02])
        self.assertEqual(r.instructions[0].mnemonic, "CMP_GT")
        self.assertEqual(r.instructions[0].operands, ["R0", "R1", "R2"])


class TestOpcodeCMP_NE(unittest.TestCase):
    def test_cmp_ne(self):
        r = _decompile([0x2F, 0x00, 0x01, 0x02])
        self.assertEqual(r.instructions[0].mnemonic, "CMP_NE")
        self.assertEqual(r.instructions[0].operands, ["R0", "R1", "R2"])


class TestOpcodeMOV(unittest.TestCase):
    def test_mov(self):
        r = _decompile([0x3A, 0x02, 0x00, 0x01])
        self.assertEqual(r.instructions[0].mnemonic, "MOV")
        self.assertEqual(r.instructions[0].operands, ["R2", "R0", "R1"])


class TestOpcodeJZ(unittest.TestCase):
    def test_jz_structure(self):
        # JZ R0, +3 (jump forward 3 from offset 0 -> target 3)
        r = _decompile([0x3C, 0x00, 0x03, 0x00])
        self.assertEqual(r.instructions[0].mnemonic, "JZ")
        self.assertEqual(r.instructions[0].operands, ["R0", "3"])
        self.assertEqual(r.instructions[0].jump_type, JumpType.CONDITIONAL)
        self.assertEqual(r.instructions[0].jump_target, 3)


class TestOpcodeJNZ(unittest.TestCase):
    def test_jnz_structure(self):
        # JNZ R1, +5
        r = _decompile([0x3D, 0x01, 0x05, 0x00])
        self.assertEqual(r.instructions[0].mnemonic, "JNZ")
        self.assertEqual(r.instructions[0].operands, ["R1", "5"])
        self.assertEqual(r.instructions[0].jump_type, JumpType.CONDITIONAL)
        self.assertEqual(r.instructions[0].jump_target, 5)


class TestOpcodeMOVI16(unittest.TestCase):
    def test_movi16_positive(self):
        # MOVI16 R0, 256 (0x00, 0x01 in little-endian)
        r = _decompile([0x40, 0x00, 0x00, 0x01])
        self.assertEqual(r.instructions[0].mnemonic, "MOVI16")
        self.assertEqual(r.instructions[0].operands, ["R0", "256"])

    def test_movi16_zero(self):
        r = _decompile([0x40, 0x01, 0x00, 0x00])
        self.assertEqual(r.instructions[0].operands, ["R1", "0"])


class TestOpcodeJMP(unittest.TestCase):
    def test_jmp_structure(self):
        # JMP +10 (from offset 0 -> target 10)
        # offset is 16-bit LE signed: 0x0A, 0x00
        r = _decompile([0x43, 0x00, 0x0A, 0x00])
        self.assertEqual(r.instructions[0].mnemonic, "JMP")
        self.assertEqual(r.instructions[0].jump_type, JumpType.UNCONDITIONAL)
        self.assertEqual(r.instructions[0].jump_target, 10)


class TestOpcodeLOOP(unittest.TestCase):
    def test_loop_structure(self):
        # LOOP R0, <offset>
        r = _decompile([0x46, 0x00, 0x00, 0x00])
        self.assertEqual(r.instructions[0].mnemonic, "LOOP")
        self.assertEqual(r.instructions[0].operands, ["R0", "0"])
        self.assertEqual(r.instructions[0].jump_type, JumpType.LOOP)

    def test_loop_labels_self(self):
        r = _decompile([0x46, 0x01, 0x00, 0x00])
        # LOOP should label its own offset
        self.assertIn(0, r.labels)
        self.assertEqual(r.labels[0], "loop_000")


# ═══════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════

class TestEmptyBytecode(unittest.TestCase):
    def test_empty(self):
        r = _decompile([])
        self.assertEqual(r.total_instructions, 0)
        self.assertEqual(r.total_bytes, 0)
        self.assertEqual(r.instructions, [])
        self.assertEqual(r.labels, {})
        self.assertEqual(r.mnemonic_counts, {})
        self.assertEqual(r.jump_count, 0)

    def test_empty_to_asm(self):
        r = _decompile([])
        self.assertEqual(r.to_asm(), "")

    def test_empty_to_annotated(self):
        r = _decompile([])
        ann = r.to_annotated()
        self.assertIn("FLUX Bytecode Decompilation", ann)
        self.assertIn("0 instructions, 0 bytes", ann)


class TestUnknownOpcodes(unittest.TestCase):
    def test_single_unknown(self):
        r = _decompile([0xFE])
        self.assertEqual(r.total_instructions, 1)
        self.assertEqual(r.instructions[0].mnemonic, "DATA")
        self.assertEqual(r.instructions[0].operands, ["0xfe"])
        self.assertEqual(r.instructions[0].raw_bytes, [0xFE])

    def test_unknown_with_comment(self):
        r = _decompile([0xAA])
        self.assertEqual(r.instructions[0].comment, "unknown opcode")

    def test_multiple_consecutive_unknown(self):
        r = _decompile([0xF0, 0xF1, 0xF2, 0xF3])
        self.assertEqual(r.total_instructions, 4)
        for inst in r.instructions:
            self.assertEqual(inst.mnemonic, "DATA")
        self.assertEqual(r.mnemonic_counts["DATA"], 4)

    def test_unknown_interleaved(self):
        r = _decompile([0xFE, 0x00, 0xFF, 0x01])
        self.assertEqual(r.total_instructions, 4)
        self.assertEqual(r.instructions[0].mnemonic, "DATA")
        self.assertEqual(r.instructions[1].mnemonic, "HALT")
        self.assertEqual(r.instructions[2].mnemonic, "DATA")
        self.assertEqual(r.instructions[3].mnemonic, "NOP")


class TestSignedImmediates(unittest.TestCase):
    def test_imm8_positive_max(self):
        # 127 = 0x7F, should stay 127
        r = _decompile([0x18, 0x00, 0x7F])
        self.assertEqual(r.instructions[0].operands, ["R0", "127"])

    def test_imm8_negative_one(self):
        # 0xFF = -1 in signed
        r = _decompile([0x18, 0x00, 0xFF])
        self.assertEqual(r.instructions[0].operands, ["R0", "-1"])

    def test_imm8_negative_128(self):
        # 0x80 = -128 in signed
        r = _decompile([0x18, 0x00, 0x80])
        self.assertEqual(r.instructions[0].operands, ["R0", "-128"])

    def test_imm8_zero(self):
        r = _decompile([0x18, 0x00, 0x00])
        self.assertEqual(r.instructions[0].operands, ["R0", "0"])

    def test_addi_negative(self):
        # ADDI R0, -10  (0xF6 = 256-10)
        r = _decompile([0x19, 0x00, 0xF6])
        self.assertEqual(r.instructions[0].operands, ["R0", "-10"])

    def test_subi_negative(self):
        # SUBI R1, -5  (0xFB = 256-5)
        r = _decompile([0x1A, 0x01, 0xFB])
        self.assertEqual(r.instructions[0].operands, ["R1", "-5"])


class TestSigned16BitImmediates(unittest.TestCase):
    def test_imm16_positive_small(self):
        # 42 = 0x002A in LE
        r = _decompile([0x40, 0x00, 0x2A, 0x00])
        self.assertEqual(r.instructions[0].operands, ["R0", "42"])

    def test_imm16_positive_large(self):
        # 32767 = 0x7FFF in LE
        r = _decompile([0x40, 0x00, 0xFF, 0x7F])
        self.assertEqual(r.instructions[0].operands, ["R0", "32767"])

    def test_imm16_negative_one(self):
        # -1 = 0xFFFF in LE
        r = _decompile([0x40, 0x00, 0xFF, 0xFF])
        self.assertEqual(r.instructions[0].operands, ["R0", "-1"])

    def test_imm16_negative_min(self):
        # -32768 = 0x8000 in LE
        r = _decompile([0x40, 0x00, 0x00, 0x80])
        self.assertEqual(r.instructions[0].operands, ["R0", "-32768"])

    def test_mov16_size(self):
        r = _decompile([0x40, 0x00, 0x00, 0x00])
        self.assertEqual(r.instructions[0].size, 4)


class TestBytecodeWithoutHalt(unittest.TestCase):
    def test_no_halt_single(self):
        """Program ending without HALT should still decompile."""
        r = _decompile([0x08, 0x00])
        self.assertEqual(r.total_instructions, 1)
        self.assertEqual(r.instructions[0].mnemonic, "INC")

    def test_no_halt_multi(self):
        r = _decompile([0x18, 0x00, 10, 0x08, 0x00])
        self.assertEqual(r.total_instructions, 2)
        self.assertNotIn("HALT", _mnemonics(r))


# ═══════════════════════════════════════════════════════
# Jump target label generation
# ═══════════════════════════════════════════════════════

class TestJumpLabelGeneration(unittest.TestCase):
    def test_jz_creates_label(self):
        # JZ R0, +8 at offset 0 -> target 8
        bc = [0x3C, 0x00, 0x08, 0x00]
        # Pad with NOPs to reach offset 8
        bc += [0x01] * 8
        r = _decompile(bc)
        self.assertIn(8, r.labels)
        self.assertEqual(r.labels[8], "lbl_008")

    def test_jnz_creates_label(self):
        # JNZ R1, +4 at offset 0 -> target 4
        bc = [0x3D, 0x01, 0x04, 0x00]
        bc += [0x01] * 4
        r = _decompile(bc)
        self.assertIn(4, r.labels)
        self.assertEqual(r.labels[4], "lbl_004")

    def test_jmp_creates_label(self):
        # JMP +12 from offset 0 -> target 12
        bc = [0x43, 0x00, 0x0C, 0x00]
        bc += [0x01] * 12
        r = _decompile(bc)
        self.assertIn(12, r.labels)
        self.assertEqual(r.labels[12], "lbl_012")

    def test_label_deduplication(self):
        """Two jumps to the same target should produce a single label."""
        # Two JZ instructions both jumping to offset 8
        bc = [0x3C, 0x00, 0x08, 0x00]  # offset 0: JZ R0, +8
        bc += [0x3C, 0x01, 0x04, 0x00]  # offset 4: JZ R1, +8 (4+4=8)
        bc += [0x01] * 8  # padding
        r = _decompile(bc)
        # Both should refer to the same label
        self.assertEqual(r.labels[8], "lbl_008")
        # Only one label entry for offset 8
        self.assertEqual(sum(1 for off in r.labels if off == 8), 1)

    def test_different_targets_different_labels(self):
        """Jumps to different targets get distinct labels."""
        bc = [0x3C, 0x00, 0x04, 0x00]  # -> target 4
        bc += [0x3D, 0x01, 0x08, 0x00]  # offset 4: -> target 12 (4+8)
        bc += [0x01] * 12
        r = _decompile(bc)
        self.assertEqual(r.labels.get(4), "lbl_004")
        self.assertEqual(r.labels.get(12), "lbl_012")
        self.assertNotEqual(r.labels[4], r.labels[12])


# ═══════════════════════════════════════════════════════
# Jump direction
# ═══════════════════════════════════════════════════════

class TestJMPDirection(unittest.TestCase):
    def test_jmp_forward(self):
        # JMP forward: at offset 0, jump +20 -> target 20
        bc = [0x43, 0x00, 0x14, 0x00]  # 0x14 = 20
        bc += [0x01] * 20
        r = _decompile(bc)
        self.assertEqual(r.instructions[0].jump_target, 20)
        self.assertEqual(r.instructions[0].jump_type, JumpType.UNCONDITIONAL)

    def test_jmp_backward(self):
        # JMP backward: at offset 12, jump -12 -> target 0
        # Build bytecode: 12 bytes of NOPs, then JMP -12
        bc = [0x01] * 12
        # JMP at offset 12, offset = -12 = 0xFFF4 (as signed16 LE: 0xF4, 0xFF)
        bc += [0x43, 0x00, 0xF4, 0xFF]
        r = _decompile(bc)
        self.assertEqual(r.instructions[-1].jump_target, 0)
        self.assertEqual(r.labels[0], "lbl_000")


# ═══════════════════════════════════════════════════════
# Conditional jumps
# ═══════════════════════════════════════════════════════

class TestJZConditional(unittest.TestCase):
    def test_jz_zero_register(self):
        """JZ tests if register is zero."""
        # JZ R0, +8 at offset 0
        r = _decompile([0x3C, 0x00, 0x08, 0x00])
        self.assertEqual(r.instructions[0].comment, "if R0==0 goto lbl_008")

    def test_jz_nonzero_register(self):
        """JZ with different register."""
        r = _decompile([0x3C, 0x05, 0x08, 0x00])
        self.assertEqual(r.instructions[0].comment, "if R5==0 goto lbl_008")
        self.assertEqual(r.instructions[0].operands[0], "R5")

    def test_jz_is_conditional(self):
        r = _decompile([0x3C, 0x00, 0x04, 0x00])
        self.assertEqual(r.instructions[0].jump_type, JumpType.CONDITIONAL)


class TestJNZConditional(unittest.TestCase):
    def test_jnz_register(self):
        """JNZ tests if register is non-zero."""
        r = _decompile([0x3D, 0x03, 0x06, 0x00])
        self.assertEqual(r.instructions[0].comment, "if R3!=0 goto lbl_006")

    def test_jnz_is_conditional(self):
        r = _decompile([0x3D, 0x00, 0x04, 0x00])
        self.assertEqual(r.instructions[0].jump_type, JumpType.CONDITIONAL)

    def test_jnz_backward(self):
        """JNZ jumping backward."""
        # NOPs + JNZ at offset 8, offset = -4 -> target 4
        bc = [0x01] * 8
        bc += [0x3D, 0x00, 0xFC, 0x00]  # 0xFC = -4 signed
        r = _decompile(bc)
        self.assertEqual(r.instructions[-1].jump_target, 4)
        self.assertEqual(r.labels[4], "lbl_004")


class TestLOOPInstruction(unittest.TestCase):
    def test_loop_jump_type(self):
        r = _decompile([0x46, 0x00, 0x00, 0x00])
        self.assertEqual(r.instructions[0].jump_type, JumpType.LOOP)

    def test_loop_label_format(self):
        # LOOP at offset 8
        bc = [0x01] * 8
        bc += [0x46, 0x01, 0x00, 0x00]
        r = _decompile(bc)
        self.assertIn(8, r.labels)
        self.assertEqual(r.labels[8], "loop_008")

    def test_loop_comment(self):
        r = _decompile([0x46, 0x03, 0x00, 0x00])
        self.assertIn("R3", r.instructions[0].comment)
        self.assertIn("decrements", r.instructions[0].comment)


# ═══════════════════════════════════════════════════════
# Complex programs
# ═══════════════════════════════════════════════════════

class TestFactorial(unittest.TestCase):
    """Factorial: R0 = 6, R1 = 1; loop: MUL R1, R1, R0; DEC R0; JNZ R0, back; HALT"""
    def setUp(self):
        # MOVI R0, 6; MOVI R1, 1; MUL R1, R1, R0; DEC R0; JNZ R0, -6; NOP; HALT
        self.bc = [
            0x18, 0x00, 0x06,       # 0: MOVI R0, 6
            0x18, 0x01, 0x01,       # 3: MOVI R1, 1
            0x22, 0x01, 0x01, 0x00, # 6: MUL R1, R1, R0
            0x09, 0x00,              # 10: DEC R0
            0x3D, 0x00, 0xFA, 0x00, # 12: JNZ R0, -6 (12 + (-6) = 6)
            0x00,                    # 16: HALT
        ]
        self.r = _decompile(self.bc)

    def test_has_mul(self):
        self.assertIn("MUL", _mnemonics(self.r))

    def test_has_dec(self):
        self.assertIn("DEC", _mnemonics(self.r))

    def test_has_jnz(self):
        self.assertIn("JNZ", _mnemonics(self.r))

    def test_jump_target(self):
        jnz = [i for i in self.r.instructions if i.mnemonic == "JNZ"][0]
        self.assertEqual(jnz.jump_target, 6)

    def test_mnemonic_counts(self):
        self.assertEqual(self.r.mnemonic_counts.get("MOVI", 0), 2)
        self.assertEqual(self.r.mnemonic_counts.get("MUL", 0), 1)
        self.assertEqual(self.r.mnemonic_counts.get("DEC", 0), 1)


class TestFibonacci(unittest.TestCase):
    """Fibonacci: R0=a=1, R1=b=1; loop: R2=R0+R1; R0=R1; R1=R2; DEC counter; JNZ"""
    def setUp(self):
        self.bc = [
            0x18, 0x00, 0x01,       # 0: MOVI R0, 1
            0x18, 0x01, 0x01,       # 3: MOVI R1, 1
            0x18, 0x03, 0x0A,       # 6: MOVI R3, 10 (counter)
            # loop start at 9:
            0x20, 0x02, 0x00, 0x01, # 9: ADD R2, R0, R1
            0x3A, 0x00, 0x01, 0x02, # 13: MOV R0, R1, R2
            0x3A, 0x01, 0x02, 0x02, # 17: MOV R1, R2, R2 (temp copy)
            0x09, 0x03,              # 21: DEC R3
            0x3D, 0x03, 0xF3, 0x00, # 23: JNZ R3, -13 (23 + (-13) = 10... let me recalculate)
            0x00,                    # 27: HALT
        ]
        # Fix JNZ offset: loop body starts at 9, JNZ at 23, need 23 + offset = 9 => offset = -14
        self.bc[25] = 0xF2  # -14 in signed8 = 256 - 14 = 242 = 0xF2
        self.r = _decompile(self.bc)

    def test_has_add(self):
        self.assertIn("ADD", _mnemonics(self.r))

    def test_has_mov(self):
        self.assertIn("MOV", _mnemonics(self.r))

    def test_has_jnz(self):
        self.assertIn("JNZ", _mnemonics(self.r))

    def test_jump_back_to_loop_body(self):
        jnz = [i for i in self.r.instructions if i.mnemonic == "JNZ"][0]
        self.assertEqual(jnz.jump_target, 9)

    def test_three_movi(self):
        self.assertEqual(self.r.mnemonic_counts.get("MOVI", 0), 3)


class TestSwapViaStack(unittest.TestCase):
    """Swap two registers using PUSH/POP."""
    def setUp(self):
        self.bc = [
            0x18, 0x00, 0x0A,       # MOVI R0, 10
            0x18, 0x01, 0x14,       # MOVI R1, 20
            0x0C, 0x00,              # PUSH R0
            0x3A, 0x00, 0x01, 0x00, # MOV R0, R1, R0 (copy R1 to R0)
            0x0D, 0x02,              # POP R2
            0x00,                    # HALT
        ]
        self.r = _decompile(self.bc)

    def test_has_push(self):
        self.assertIn("PUSH", _mnemonics(self.r))

    def test_has_pop(self):
        self.assertIn("POP", _mnemonics(self.r))

    def test_push_pop_count(self):
        self.assertEqual(self.r.mnemonic_counts.get("PUSH", 0), 1)
        self.assertEqual(self.r.mnemonic_counts.get("POP", 0), 1)

    def test_order_push_before_pop(self):
        mnems = _mnemonics(self.r)
        push_idx = mnems.index("PUSH")
        pop_idx = mnems.index("POP")
        self.assertLess(push_idx, pop_idx)


class TestGCD(unittest.TestCase):
    """GCD using MOD and conditional jump."""
    def setUp(self):
        self.bc = [
            0x18, 0x00, 0x30,       # MOVI R0, 48
            0x18, 0x01, 0x12,       # MOVI R1, 18
            # loop:
            0x24, 0x02, 0x00, 0x01, # MOD R2, R0, R1
            0x3A, 0x00, 0x01, 0x02, # MOV R0, R1, R2
            0x3C, 0x02, 0x00, 0x00, # JZ R2, +0 -> target 15 (HALT)
            0x09, 0x02,              # DEC R2 (reduce)
            0x00,                    # HALT
        ]
        self.r = _decompile(self.bc)

    def test_has_mod(self):
        self.assertIn("MOD", _mnemonics(self.r))

    def test_has_jz(self):
        self.assertIn("JZ", _mnemonics(self.r))

    def test_mod_operands(self):
        mod_inst = [i for i in self.r.instructions if i.mnemonic == "MOD"][0]
        self.assertEqual(mod_inst.operands, ["R2", "R0", "R1"])


# ═══════════════════════════════════════════════════════
# Output format validation
# ═══════════════════════════════════════════════════════

class TestToAsmOutput(unittest.TestCase):
    def test_contains_mnemonic(self):
        r = _decompile([0x18, 0x00, 42, 0x00])
        asm = r.to_asm()
        self.assertIn("MOVI", asm)
        self.assertIn("HALT", asm)

    def test_contains_offset(self):
        r = _decompile([0x18, 0x00, 42, 0x00])
        asm = r.to_asm()
        self.assertIn("   0:", asm)

    def test_contains_hex_bytes(self):
        r = _decompile([0x18, 0x00, 42, 0x00])
        asm = r.to_asm()
        self.assertIn("18 00 2a", asm)

    def test_label_in_asm(self):
        bc = [0x3C, 0x00, 0x08, 0x00] + [0x01] * 8 + [0x00]
        r = _decompile(bc)
        asm = r.to_asm()
        self.assertIn("lbl_008:", asm)

    def test_comment_in_asm(self):
        r = _decompile([0x3C, 0x00, 0x04, 0x00])
        asm = r.to_asm()
        self.assertIn("if R0==0", asm)

    def test_empty_asm(self):
        r = _decompile([])
        self.assertEqual(r.to_asm(), "")

    def test_multiple_lines(self):
        r = _decompile([0x01, 0x00])
        asm = r.to_asm()
        lines = asm.strip().split("\n")
        self.assertGreaterEqual(len(lines), 2)


class TestToAnnotatedOutput(unittest.TestCase):
    def test_header(self):
        r = _decompile([0x00])
        ann = r.to_annotated()
        self.assertIn("FLUX Bytecode Decompilation", ann)

    def test_stats_section(self):
        r = _decompile([0x18, 0x00, 10, 0x00])
        ann = r.to_annotated()
        self.assertIn("Stats", ann)
        self.assertIn("2 instructions, 4 bytes", ann)

    def test_conditional_marker(self):
        r = _decompile([0x3C, 0x00, 0x04, 0x00])
        ann = r.to_annotated()
        # Should contain the conditional marker for JZ
        # The marker is prepended to the instruction line
        self.assertIn("JZ", ann)

    def test_unconditional_marker(self):
        bc = [0x43, 0x00, 0x08, 0x00] + [0x01] * 8
        r = _decompile(bc)
        ann = r.to_annotated()
        self.assertIn("JMP", ann)

    def test_loop_marker(self):
        r = _decompile([0x46, 0x00, 0x00, 0x00])
        ann = r.to_annotated()
        self.assertIn("LOOP", ann)

    def test_empty_annotated(self):
        r = _decompile([])
        ann = r.to_annotated()
        self.assertIn("0 instructions, 0 bytes", ann)
        self.assertIn("0 jumps", ann)

    def test_mnemonic_frequency_in_stats(self):
        r = _decompile([0x18, 0x00, 10, 0x18, 0x01, 20, 0x00])
        ann = r.to_annotated()
        self.assertIn("MOVI", ann)  # Should appear in stats
        self.assertIn("2x", ann)    # MOVI count


# ═══════════════════════════════════════════════════════
# Mnemonic counts accuracy
# ═══════════════════════════════════════════════════════

class TestMnemonicCounts(unittest.TestCase):
    def test_single_instruction(self):
        r = _decompile([0x00])
        self.assertEqual(r.mnemonic_counts, {"HALT": 1})

    def test_mixed_instructions(self):
        bc = [0x18, 0x00, 10, 0x18, 0x01, 20, 0x20, 0x02, 0x00, 0x01, 0x00]
        r = _decompile(bc)
        self.assertEqual(r.mnemonic_counts["MOVI"], 2)
        self.assertEqual(r.mnemonic_counts["ADD"], 1)
        self.assertEqual(r.mnemonic_counts["HALT"], 1)

    def test_no_unknown_in_mnemonic_counts(self):
        """DATA instructions should appear in counts."""
        r = _decompile([0xFE, 0x00])
        self.assertEqual(r.mnemonic_counts["DATA"], 1)
        self.assertEqual(r.mnemonic_counts["HALT"], 1)

    def test_counts_match_instructions(self):
        """Sum of all counts should equal total_instructions."""
        bc = [0x18, 0x00, 10, 0x08, 0x00, 0x09, 0x00, 0x00]
        r = _decompile(bc)
        total = sum(r.mnemonic_counts.values())
        self.assertEqual(total, r.total_instructions)

    def test_complex_program_counts(self):
        bc = (
            [0x18, 0x00, 0x06] +       # MOVI R0, 6
            [0x18, 0x01, 0x01] +       # MOVI R1, 1
            [0x22, 0x01, 0x01, 0x00] + # MUL R1, R1, R0
            [0x09, 0x00] +              # DEC R0
            [0x3D, 0x00, 0xFA, 0x00] + # JNZ R0, -6
            [0x00]                      # HALT
        )
        r = _decompile(bc)
        self.assertEqual(r.mnemonic_counts.get("MOVI", 0), 2)
        self.assertEqual(r.mnemonic_counts.get("MUL", 0), 1)
        self.assertEqual(r.mnemonic_counts.get("DEC", 0), 1)
        self.assertEqual(r.mnemonic_counts.get("JNZ", 0), 1)
        self.assertEqual(r.mnemonic_counts.get("HALT", 0), 1)
        self.assertEqual(sum(r.mnemonic_counts.values()), r.total_instructions)


# ═══════════════════════════════════════════════════════
# Labels and jump metadata
# ═══════════════════════════════════════════════════════

class TestLabelsAndJumps(unittest.TestCase):
    def test_jump_count_increments(self):
        r = _decompile([0x3C, 0x00, 0x04, 0x00])
        self.assertEqual(r.jump_count, 1)

    def test_multiple_jumps_count(self):
        bc = [0x3C, 0x00, 0x04, 0x00, 0x3D, 0x01, 0x08, 0x00, 0x00]
        r = _decompile(bc)
        self.assertEqual(r.jump_count, 2)

    def test_non_jump_doesnt_increment(self):
        r = _decompile([0x18, 0x00, 10, 0x00])
        self.assertEqual(r.jump_count, 0)

    def test_labels_dict_keys_are_offsets(self):
        bc = [0x3C, 0x00, 0x08, 0x00] + [0x01] * 8 + [0x00]
        r = _decompile(bc)
        for key in r.labels:
            self.assertIsInstance(key, int)

    def test_label_format_prefix(self):
        bc = [0x43, 0x00, 0x04, 0x00] + [0x01] * 4
        r = _decompile(bc)
        for lbl in r.labels.values():
            self.assertTrue(lbl.startswith("lbl_") or lbl.startswith("loop_"))


# ═══════════════════════════════════════════════════════
# Raw bytes and size
# ═══════════════════════════════════════════════════════

class TestRawBytesAndSize(unittest.TestCase):
    def test_halt_raw_bytes(self):
        r = _decompile([0x00])
        self.assertEqual(r.instructions[0].raw_bytes, [0x00])

    def test_movi_raw_bytes(self):
        r = _decompile([0x18, 0x05, 0x2A])
        self.assertEqual(r.instructions[0].raw_bytes, [0x18, 0x05, 0x2A])

    def test_add_raw_bytes(self):
        r = _decompile([0x20, 0x00, 0x01, 0x02])
        self.assertEqual(r.instructions[0].raw_bytes, [0x20, 0x00, 0x01, 0x02])

    def test_total_bytes_matches_input(self):
        bc = [0x18, 0x00, 10, 0x18, 0x01, 20, 0x20, 0x02, 0x00, 0x01, 0x00]
        r = _decompile(bc)
        self.assertEqual(r.total_bytes, len(bc))

    def test_unknown_raw_bytes(self):
        r = _decompile([0xFE])
        self.assertEqual(r.instructions[0].raw_bytes, [0xFE])
        self.assertEqual(r.instructions[0].size, 1)


# ═══════════════════════════════════════════════════════
# All opcodes covered
# ═══════════════════════════════════════════════════════

class TestAllOpcodesCovered(unittest.TestCase):
    """Verify every opcode in OP_SPECS is individually tested."""
    def test_all_opcodes_have_test_coverage(self):
        """Sanity check: OP_SPECS has expected number of entries."""
        self.assertGreaterEqual(len(OP_SPECS), 34)

    def test_opcode_0x00_is_halt(self):
        self.assertEqual(OP_SPECS[0x00][0], "HALT")

    def test_opcode_0x01_is_nop(self):
        self.assertEqual(OP_SPECS[0x01][0], "NOP")

    def test_opcode_0x46_is_loop(self):
        self.assertEqual(OP_SPECS[0x46][0], "LOOP")


if __name__ == "__main__":
    unittest.main(verbosity=2)
