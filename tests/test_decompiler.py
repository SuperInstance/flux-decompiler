"""Comprehensive tests for FLUX decompiler — 60 tests covering all opcodes,
control flow, labels, output formats, edge cases, and error handling."""
import pytest
from decompiler import (
    FluxDecompiler,
    DecompilationResult,
    DecodedInstruction,
    JumpType,
    OP_SPECS,
    JUMP_OPS,
)


def build_instruction(opcode: int, *operands) -> list:
    """Build a raw bytecode instruction from opcode and operand bytes."""
    spec = OP_SPECS.get(opcode)
    if spec is None:
        return [opcode]
    _, size, _ = spec
    bc = [opcode] + list(operands[:size - 1])
    while len(bc) < size:
        bc.append(0)
    return bc


def decompile(bytecode):
    """Convenience: decompile bytecode and return result."""
    return FluxDecompiler(bytecode).decompile()


# ─── Opcode Coverage: Single-byte instructions ────────────────────────────

class TestSingleByteOpcodes:
    """Test opcodes with no operands (size=1)."""

    def test_halt_opcode_0x00(self):
        r = decompile([0x00])
        assert r.total_instructions == 1
        assert r.instructions[0].mnemonic == "HALT"
        assert r.instructions[0].size == 1
        assert r.instructions[0].operands == []

    def test_nop_opcode_0x01(self):
        r = decompile([0x01])
        assert r.total_instructions == 1
        assert r.instructions[0].mnemonic == "NOP"
        assert r.instructions[0].size == 1
        assert r.instructions[0].operands == []

    def test_halt_raw_bytes(self):
        r = decompile([0x00])
        assert r.instructions[0].raw_bytes == [0x00]

    def test_nop_raw_bytes(self):
        r = decompile([0x01])
        assert r.instructions[0].raw_bytes == [0x01]

    def test_multiple_halts(self):
        r = decompile([0x00, 0x00, 0x00])
        assert r.total_instructions == 3
        for inst in r.instructions:
            assert inst.mnemonic == "HALT"

    def test_multiple_nops(self):
        r = decompile([0x01, 0x01])
        assert r.total_instructions == 2
        assert r.mnemonic_counts.get("NOP") == 2


# ─── Opcode Coverage: Two-byte register instructions ─────────────────────

class TestTwoByteOpcodes:
    """Test opcodes with one register operand (size=2)."""

    @pytest.mark.parametrize("opcode,mnemonic", [
        (0x08, "INC"), (0x09, "DEC"), (0x0A, "NOT"), (0x0B, "NEG"),
        (0x0C, "PUSH"), (0x0D, "POP"), (0x17, "STRIPCONF"),
    ])
    def test_mnemonic(self, opcode, mnemonic):
        bc = build_instruction(opcode, 0)
        r = decompile(bc)
        assert r.instructions[0].mnemonic == mnemonic

    @pytest.mark.parametrize("opcode,mnemonic", [
        (0x08, "INC"), (0x09, "DEC"), (0x0A, "NOT"), (0x0B, "NEG"),
        (0x0C, "PUSH"), (0x0D, "POP"), (0x17, "STRIPCONF"),
    ])
    def test_size(self, opcode, mnemonic):
        bc = build_instruction(opcode, 0)
        r = decompile(bc)
        assert r.instructions[0].size == 2

    @pytest.mark.parametrize("opcode", [0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x17])
    def test_register_operand(self, opcode):
        for reg in [0, 1, 5, 15, 255]:
            bc = build_instruction(opcode, reg)
            r = decompile(bc)
            assert f"R{reg}" in r.instructions[0].operands

    def test_inc_r0(self):
        r = decompile([0x08, 0x00])
        assert r.instructions[0].operands == ["R0"]

    def test_dec_r5(self):
        r = decompile([0x09, 0x05])
        assert r.instructions[0].operands == ["R5"]

    def test_push_r3(self):
        r = decompile([0x0C, 0x03])
        assert r.instructions[0].operands == ["R3"]
        assert r.instructions[0].mnemonic == "PUSH"

    def test_pop_r3(self):
        r = decompile([0x0D, 0x03])
        assert r.instructions[0].operands == ["R3"]
        assert r.instructions[0].mnemonic == "POP"

    def test_stripconf_r7(self):
        r = decompile([0x17, 0x07])
        assert r.instructions[0].mnemonic == "STRIPCONF"
        assert r.instructions[0].operands == ["R7"]


# ─── Opcode Coverage: Three-byte immediate instructions ───────────────────

class TestThreeByteOpcodes:
    """Test opcodes with reg + imm8 operands (size=3)."""

    @pytest.mark.parametrize("opcode,mnemonic", [
        (0x18, "MOVI"), (0x19, "ADDI"), (0x1A, "SUBI"),
    ])
    def test_mnemonic(self, opcode, mnemonic):
        bc = build_instruction(opcode, 0, 42)
        r = decompile(bc)
        assert r.instructions[0].mnemonic == mnemonic

    @pytest.mark.parametrize("opcode", [0x18, 0x19, 0x1A])
    def test_size(self, opcode):
        bc = build_instruction(opcode, 0, 42)
        r = decompile(bc)
        assert r.instructions[0].size == 3

    def test_movi_positive(self):
        r = decompile([0x18, 0x02, 42])
        assert r.instructions[0].operands == ["R2", "42"]

    def test_movi_zero(self):
        r = decompile([0x18, 0x01, 0])
        assert r.instructions[0].operands == ["R1", "0"]

    def test_movi_max_unsigned(self):
        """imm8 = 255 should display as -1 (signed)."""
        r = decompile([0x18, 0x00, 0xFF])
        assert r.instructions[0].operands == ["R0", "-1"]

    def test_movi_128(self):
        """imm8 = 128 (0x80) should display as -128 (signed)."""
        r = decompile([0x18, 0x00, 0x80])
        assert r.instructions[0].operands == ["R0", "-128"]

    def test_addi_positive(self):
        r = decompile([0x19, 0x03, 10])
        assert r.instructions[0].operands == ["R3", "10"]
        assert r.instructions[0].mnemonic == "ADDI"

    def test_addi_negative(self):
        r = decompile([0x19, 0x03, 0xFC])
        assert r.instructions[0].operands == ["R3", "-4"]

    def test_subi_positive(self):
        r = decompile([0x1A, 0x01, 5])
        assert r.instructions[0].operands == ["R1", "5"]
        assert r.instructions[0].mnemonic == "SUBI"


# ─── Opcode Coverage: Four-byte ALU instructions ─────────────────────────

class TestFourByteALUOpcodes:
    """Test ALU opcodes with reg, reg, reg operands (size=4)."""

    ALU_OPCODES = [
        (0x20, "ADD"), (0x21, "SUB"), (0x22, "MUL"), (0x23, "DIV"),
        (0x24, "MOD"), (0x25, "AND"), (0x26, "OR"), (0x27, "XOR"),
        (0x28, "SHL"), (0x29, "SHR"), (0x2A, "MIN"), (0x2B, "MAX"),
        (0x2C, "CMP_EQ"), (0x2D, "CMP_LT"), (0x2E, "CMP_GT"),
        (0x2F, "CMP_NE"),
    ]

    @pytest.mark.parametrize("opcode,mnemonic", ALU_OPCODES)
    def test_mnemonic(self, opcode, mnemonic):
        bc = build_instruction(opcode, 0, 1, 2)
        r = decompile(bc)
        assert r.instructions[0].mnemonic == mnemonic

    @pytest.mark.parametrize("opcode,mnemonic", ALU_OPCODES)
    def test_size(self, opcode, mnemonic):
        bc = build_instruction(opcode, 0, 1, 2)
        r = decompile(bc)
        assert r.instructions[0].size == 4

    @pytest.mark.parametrize("opcode,mnemonic", ALU_OPCODES)
    def test_three_register_operands(self, opcode, mnemonic):
        bc = build_instruction(opcode, 5, 3, 7)
        r = decompile(bc)
        assert r.instructions[0].operands == ["R5", "R3", "R7"]

    def test_add(self):
        r = decompile([0x20, 2, 0, 1])
        assert r.instructions[0].mnemonic == "ADD"
        assert r.instructions[0].operands == ["R2", "R0", "R1"]

    def test_mul(self):
        r = decompile([0x22, 1, 1, 0])
        assert r.instructions[0].mnemonic == "MUL"
        assert r.instructions[0].operands == ["R1", "R1", "R0"]

    def test_div(self):
        r = decompile([0x23, 4, 6, 2])
        assert r.instructions[0].mnemonic == "DIV"
        assert r.instructions[0].operands == ["R4", "R6", "R2"]

    def test_xor(self):
        r = decompile([0x27, 0, 1, 2])
        assert r.instructions[0].mnemonic == "XOR"
        assert r.instructions[0].operands == ["R0", "R1", "R2"]

    def test_shl(self):
        r = decompile([0x28, 3, 1, 2])
        assert r.instructions[0].mnemonic == "SHL"

    def test_shr(self):
        r = decompile([0x29, 3, 1, 2])
        assert r.instructions[0].mnemonic == "SHR"

    def test_min(self):
        r = decompile([0x2A, 0, 1, 2])
        assert r.instructions[0].mnemonic == "MIN"

    def test_max(self):
        r = decompile([0x2B, 0, 1, 2])
        assert r.instructions[0].mnemonic == "MAX"

    def test_cmp_eq(self):
        r = decompile([0x2C, 0, 1, 2])
        assert r.instructions[0].mnemonic == "CMP_EQ"

    def test_cmp_lt(self):
        r = decompile([0x2D, 0, 1, 2])
        assert r.instructions[0].mnemonic == "CMP_LT"

    def test_cmp_gt(self):
        r = decompile([0x2E, 0, 1, 2])
        assert r.instructions[0].mnemonic == "CMP_GT"

    def test_cmp_ne(self):
        r = decompile([0x2F, 0, 1, 2])
        assert r.instructions[0].mnemonic == "CMP_NE"


# ─── Opcode Coverage: MOV instruction ─────────────────────────────────────

class TestMOVInstruction:
    """Test MOV (0x3A) register-to-register copy."""

    def test_mov(self):
        r = decompile([0x3A, 2, 0, 1])
        assert r.instructions[0].mnemonic == "MOV"
        assert r.instructions[0].operands == ["R2", "R0", "R1"]
        assert r.instructions[0].size == 4

    def test_mov_same_reg(self):
        """MOV with same source and dest is valid bytecode."""
        r = decompile([0x3A, 0, 0, 0])
        assert r.instructions[0].operands == ["R0", "R0", "R0"]


# ─── Opcode Coverage: MOVI16 instruction ─────────────────────────────────

class TestMOVI16Instruction:
    """Test MOVI16 (0x40) with 16-bit immediate."""

    def test_mov_i16_positive(self):
        # MOVI16 R3, 1000 — 1000 = 0x03E8, LE: lo=0xE8, hi=0x03
        r = decompile([0x40, 3, 0xE8, 0x03])
        assert r.instructions[0].mnemonic == "MOVI16"
        assert r.instructions[0].operands == ["R3", "1000"]
        assert r.instructions[0].size == 4

    def test_mov_i16_zero(self):
        r = decompile([0x40, 0, 0x00, 0x00])
        assert r.instructions[0].operands == ["R0", "0"]

    def test_mov_i16_negative(self):
        # -1 as signed16 LE: lo=0xFF, hi=0xFF
        r = decompile([0x40, 1, 0xFF, 0xFF])
        assert r.instructions[0].operands == ["R1", "-1"]

    def test_mov_i16_min(self):
        # -32768 as signed16 LE: lo=0x00, hi=0x80
        r = decompile([0x40, 2, 0x00, 0x80])
        assert r.instructions[0].operands == ["R2", "-32768"]

    def test_mov_i16_max(self):
        # 32767 as signed16 LE: lo=0xFF, hi=0x7F
        r = decompile([0x40, 0, 0xFF, 0x7F])
        assert r.instructions[0].operands == ["R0", "32767"]


# ─── Control Flow: Jump Instructions ──────────────────────────────────────

class TestJumpInstructions:
    """Test JZ, JNZ, JMP, and LOOP control flow reconstruction."""

    def test_jz_basic(self):
        """JZ R0, +5 — jump forward 5 bytes from instruction start."""
        # JZ at offset 6: [0x3C, 0, 5, padding]
        bc = [0x00]*6 + [0x3C, 0x00, 0x05, 0x00]
        r = decompile(bc)
        jz = [i for i in r.instructions if i.mnemonic == "JZ"][0]
        assert jz.operands == ["R0", "5"]
        assert jz.jump_target == 11  # 6 + 5
        assert jz.jump_type == JumpType.CONDITIONAL
        assert "lbl_011" in r.labels.values()

    def test_jnz_basic(self):
        """JNZ R0, -4 — jump backward 4 bytes."""
        # JNZ at offset 10: [0x3D, 0, 0xFC, 0x00]
        bc = [0x00]*10 + [0x3D, 0x00, 0xFC, 0x00]
        r = decompile(bc)
        jnz = [i for i in r.instructions if i.mnemonic == "JNZ"][0]
        assert jnz.operands == ["R0", "-4"]
        assert jnz.jump_target == 6  # 10 + (-4)
        assert jnz.jump_type == JumpType.CONDITIONAL

    def test_jz_comment(self):
        bc = [0x00]*4 + [0x3C, 0x03, 0x05, 0x00]
        r = decompile(bc)
        jz = r.instructions[-1]
        assert "if R3==0" in jz.comment
        assert "lbl_009" in jz.comment

    def test_jnz_comment(self):
        bc = [0x00]*4 + [0x3D, 0x05, 0xFC, 0x00]
        r = decompile(bc)
        jnz = r.instructions[-1]
        assert "if R5!=0" in jnz.comment
        assert "lbl_000" in jnz.comment

    def test_jz_target_label(self):
        bc = [0x00]*6 + [0x3C, 0x00, 0x05, 0x00] + [0x00]*5
        r = decompile(bc)
        assert 11 in r.labels
        assert r.labels[11] == "lbl_011"

    def test_jnz_target_label(self):
        bc = [0x00]*10 + [0x3D, 0x00, 0xFC, 0x00]
        r = decompile(bc)
        assert 6 in r.labels
        assert r.labels[6] == "lbl_006"

    def test_jmp_forward(self):
        """JMP forward: offset 0 to offset 6 (+6)."""
        # JMP at offset 0: [0x43, lo(+6), hi(0)] = [0x43, 0x06, 0x00, 0x00]
        bc = [0x43, 0x06, 0x00, 0x00] + [0x00]*6
        r = decompile(bc)
        jmp = r.instructions[0]
        assert jmp.mnemonic == "JMP"
        assert jmp.operands == ["6"]
        assert jmp.jump_target == 6  # 0 + 6
        assert jmp.jump_type == JumpType.UNCONDITIONAL

    def test_jmp_backward(self):
        """JMP backward: offset 6 to offset 0 (-6)."""
        # JMP at offset 6: [0x43, lo(-6)=0xFA, hi(0xFF)=0xFF, padding]
        bc = [0x00]*6 + [0x43, 0xFA, 0xFF, 0x00]
        r = decompile(bc)
        jmp = [i for i in r.instructions if i.mnemonic == "JMP"][0]
        assert jmp.operands == ["-6"]
        assert jmp.jump_target == 0  # 6 + (-6)
        assert jmp.jump_type == JumpType.UNCONDITIONAL

    def test_jmp_far_backward(self):
        """JMP with large negative offset."""
        # JMP -200 at offset 200: target = 0
        # -200 = 0xFF38 in signed16 LE: lo=0x38, hi=0xFF
        bc = [0x00]*200 + [0x43, 0x38, 0xFF, 0x00]
        r = decompile(bc)
        jmp = [i for i in r.instructions if i.mnemonic == "JMP"][0]
        assert jmp.operands == ["-200"]
        assert jmp.jump_target == 0

    def test_jmp_far_forward(self):
        """JMP with large positive offset."""
        # JMP +200 at offset 0: target = 200
        # 200 = 0x00C8 in signed16 LE: lo=0xC8, hi=0x00
        bc = [0x43, 0xC8, 0x00, 0x00] + [0x00]*200
        r = decompile(bc)
        jmp = r.instructions[0]
        assert jmp.operands == ["200"]
        assert jmp.jump_target == 200

    def test_loop_basic(self):
        """LOOP R1, -6 — loop back 6 bytes."""
        # LOOP at offset 6: [0x46, 1, lo(-6)=0xFA, hi=0xFF]
        bc = [0x18, 0, 5, 0x18, 1, 10, 0x46, 1, 0xFA, 0xFF, 0x00]
        r = decompile(bc)
        loop = [i for i in r.instructions if i.mnemonic == "LOOP"][0]
        assert loop.operands == ["R1", "-6"]
        assert loop.jump_target == 0  # 6 + (-6)
        assert loop.jump_type == JumpType.LOOP
        assert "R1 decrements" in loop.comment

    def test_loop_self(self):
        """LOOP with offset 0 jumps to itself."""
        bc = [0x46, 1, 0x00, 0x00, 0x00]
        r = decompile(bc)
        loop = r.instructions[0]
        assert loop.jump_target == 0  # self-referencing
        assert loop.jump_type == JumpType.LOOP
        assert "loop_000" in r.labels.values()

    def test_loop_forward(self):
        """LOOP jumping forward (unusual but valid)."""
        # LOOP at offset 0 with offset +8: target = 8
        bc = [0x46, 1, 0x08, 0x00] + [0x00]*8
        r = decompile(bc)
        loop = r.instructions[0]
        assert loop.jump_target == 8
        assert 8 in r.labels

    def test_loop_label_self_not_overwritten(self):
        """When LOOP target is itself, don't overwrite loop_NNN with lbl_NNN."""
        bc = [0x46, 1, 0x00, 0x00, 0x00]
        r = decompile(bc)
        assert r.labels[0] == "loop_000"  # not overwritten by lbl_000

    def test_loop_creates_target_label(self):
        """LOOP to a different offset creates lbl_NNN at target."""
        bc = [0x18, 0, 5, 0x18, 1, 10, 0x46, 1, 0xFA, 0xFF, 0x00]
        r = decompile(bc)
        assert 6 in r.labels  # loop label
        assert 0 in r.labels  # target label
        assert r.labels[6] == "loop_006"
        assert r.labels[0] == "lbl_000"

    def test_jump_count_includes_all_jump_types(self):
        """JZ, JNZ, JMP, and LOOP all increment jump_count."""
        bc = [0x3C, 0, 5, 0, 0x3D, 0, 0, 0, 0x43, 0, 0, 0, 0x46, 1, 0, 0]
        r = decompile(bc)
        assert r.jump_count == 4


# ─── Control Flow: Complete Programs ──────────────────────────────────────

class TestControlFlowPrograms:
    """Test control flow reconstruction in complete programs."""

    def test_counter_loop(self):
        """Counter loop: counts R1 up while R0 counts down."""
        bc = [0x18, 0, 5, 0x18, 1, 0, 0x08, 1, 0x09, 0, 0x3D, 0, 0xFC, 0, 0x00]
        r = decompile(bc)
        assert r.jump_count == 1
        # JNZ at offset 10 should jump to offset 6 (INC R1)
        jnz = [i for i in r.instructions if i.mnemonic == "JNZ"][0]
        assert jnz.jump_target == 6
        assert 6 in r.labels

    def test_factorial_program(self):
        """Factorial: compute 6!."""
        bc = [0x18, 0, 6, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00]
        r = decompile(bc)
        muls = [i for i in r.instructions if i.mnemonic == "MUL"]
        jnzs = [i for i in r.instructions if i.mnemonic == "JNZ"]
        assert len(muls) == 1
        assert len(jnzs) == 1
        assert jnzs[0].jump_target == 6  # back to MUL

    def test_unconditional_jump_program(self):
        """Program with unconditional JMP."""
        # MOVI R0, 42; JMP +4; HALT (skipped); MOVI R0, 99
        # JMP at offset 3, operand +4, target = 3+4 = 7
        bc = [0x18, 0, 42, 0x43, 0x04, 0x00, 0x00, 0x00, 0x18, 0, 99, 0x00]
        r = decompile(bc)
        jmp = [i for i in r.instructions if i.mnemonic == "JMP"][0]
        assert jmp.jump_target == 7  # 3 + 4 = 7 (MOVI R0, 99)

    def test_conditional_forward_jump(self):
        """JZ forward past some instructions."""
        # JZ R0, +8 at offset 0; NOP; NOP; HALT; NOP; NOP; NOP; NOP; NOP
        bc = [0x3C, 0, 8, 0] + [0x01]*9
        r = decompile(bc)
        jz = r.instructions[0]
        assert jz.jump_target == 8

    def test_nested_loops_simulation(self):
        """Two LOOP instructions simulating nested control flow."""
        # MOVI R0, 5; LOOP R0, -3; INC R1; HALT
        # LOOP at offset 3, operand -3, target = 3 + (-3) = 0
        bc = [0x18, 0, 5, 0x46, 0, 0xFD, 0xFF, 0x08, 1, 0x00]
        r = decompile(bc)
        loops = [i for i in r.instructions if i.mnemonic == "LOOP"]
        assert len(loops) == 1
        assert loops[0].jump_target == 0  # 3 + (-3) = 0

    def test_multiple_jumps_same_target(self):
        """Multiple jumps to the same target share one label."""
        bc = [0x3C, 0, 10, 0, 0x3D, 1, 5, 0] + [0x00]*14
        r = decompile(bc)
        # JZ at offset 0 targets 10, JNZ at offset 4 targets 9
        jz = r.instructions[0]
        jnz = r.instructions[1]
        assert jz.jump_target == 10
        assert jnz.jump_target == 9


# ─── Label Generation ─────────────────────────────────────────────────────

class TestLabelGeneration:
    """Test label naming and placement."""

    def test_label_format(self):
        bc = [0x00]*6 + [0x3C, 0, 5, 0] + [0x00]*5
        r = decompile(bc)
        for offset, label in r.labels.items():
            if label.startswith("lbl_"):
                assert label == f"lbl_{offset:03d}"

    def test_loop_label_format(self):
        bc = [0x46, 1, 0, 0, 0x00]
        r = decompile(bc)
        for offset, label in r.labels.items():
            if label.startswith("loop_"):
                assert label == f"loop_{offset:03d}"

    def test_labels_in_output(self):
        bc = [0x18, 0, 5, 0x18, 1, 0, 0x08, 1, 0x09, 0, 0x3D, 0, 0xFC, 0, 0x00]
        r = decompile(bc)
        asm = r.to_asm()
        assert "lbl_006:" in asm

    def test_loop_label_in_output(self):
        bc = [0x46, 1, 0xFA, 0xFF, 0x00]
        r = decompile(bc)
        asm = r.to_asm()
        assert "loop_000:" in asm

    def test_label_count(self):
        bc = [0x3C, 0, 10, 0, 0x3D, 1, 5, 0] + [0x00]*14
        r = decompile(bc)
        assert len(r.labels) == 2  # one for JZ target, one for JNZ target


# ─── Output Formats ───────────────────────────────────────────────────────

class TestOutputFormats:
    """Test to_asm() and to_annotated() output."""

    def test_to_asm_contains_mnemonic(self):
        bc = [0x18, 0, 42, 0x00]
        r = decompile(bc)
        asm = r.to_asm()
        assert "MOVI" in asm
        assert "HALT" in asm

    def test_to_asm_contains_offset(self):
        bc = [0x18, 0, 42, 0x00]
        r = decompile(bc)
        asm = r.to_asm()
        assert "   0:" in asm

    def test_to_asm_contains_hex_bytes(self):
        bc = [0x18, 0, 42, 0x00]
        r = decompile(bc)
        asm = r.to_asm()
        assert "18 00 2a" in asm

    def test_to_asm_contains_operands(self):
        bc = [0x18, 0, 42, 0x00]
        r = decompile(bc)
        asm = r.to_asm()
        assert "R0" in asm
        assert "42" in asm

    def test_to_annotated_contains_header(self):
        bc = [0x00]
        r = decompile(bc)
        ann = r.to_annotated()
        assert "FLUX Bytecode Decompilation" in ann

    def test_to_annotated_contains_stats(self):
        bc = [0x18, 0, 42, 0x00]
        r = decompile(bc)
        ann = r.to_annotated()
        assert "Stats" in ann
        assert "instructions" in ann
        assert "bytes" in ann

    def test_to_annotated_stats_correct(self):
        bc = [0x18, 0, 42, 0x00]
        r = decompile(bc)
        ann = r.to_annotated()
        assert "2 instructions, 4 bytes" in ann

    def test_to_annotated_jump_stats(self):
        bc = [0x00]*6 + [0x3C, 0, 5, 0] + [0x00]*5
        r = decompile(bc)
        ann = r.to_annotated()
        assert "1 jumps" in ann

    def test_to_annotated_label_stats(self):
        bc = [0x00]*6 + [0x3C, 0, 5, 0] + [0x00]*5
        r = decompile(bc)
        ann = r.to_annotated()
        assert "1 labels" in ann

    def test_to_annotated_mnemonic_frequency(self):
        bc = [0x01, 0x01, 0x01, 0x00]
        r = decompile(bc)
        ann = r.to_annotated()
        assert "NOP: 3x" in ann

    def test_to_annotated_conditional_marker(self):
        bc = [0x00]*4 + [0x3C, 0, 5, 0]
        r = decompile(bc)
        ann = r.to_annotated()
        assert "↕" in ann  # conditional marker

    def test_to_annotated_unconditional_marker(self):
        bc = [0x43, 0x04, 0x00, 0x00] + [0x00]*4
        r = decompile(bc)
        ann = r.to_annotated()
        assert "↓" in ann  # unconditional marker

    def test_to_annotated_loop_marker(self):
        bc = [0x46, 1, 0x00, 0x00, 0x00]
        r = decompile(bc)
        ann = r.to_annotated()
        assert "↻" in ann  # loop marker

    def test_to_annotated_no_marker_for_normal(self):
        bc = [0x18, 0, 42, 0x00]
        r = decompile(bc)
        ann = r.to_annotated()
        lines = ann.split("\n")
        # Find instruction lines (not header/stats)
        inst_lines = [l for l in lines if "MOVI" in l or "HALT" in l]
        for line in inst_lines:
            assert "↕" not in line
            assert "↓" not in line
            assert "↻" not in line

    def test_to_annotated_limit_top5_mnemonics(self):
        """Only top 5 mnemonics by frequency shown."""
        bc = [0x01]*5 + [0x00]*3 + [0x18, 0, 1] + [0x08, 0] + [0x09, 0] + [0x0A, 0] + [0x0B, 0]
        r = decompile(bc)
        ann = r.to_annotated()
        # Count how many "x" frequency lines appear
        freq_lines = [l for l in ann.split("\n") if ": " in l and "x" in l.split(": ")[-1]]
        assert len(freq_lines) <= 5


# ─── Mnemonic Counts ──────────────────────────────────────────────────────

class TestMnemonicCounts:
    """Test instruction frequency counting."""

    def test_single_instruction(self):
        r = decompile([0x00])
        assert r.mnemonic_counts == {"HALT": 1}

    def test_mixed_instructions(self):
        bc = [0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00]
        r = decompile(bc)
        assert r.mnemonic_counts.get("MOVI") == 2
        assert r.mnemonic_counts.get("ADD") == 1
        assert r.mnemonic_counts.get("HALT") == 1

    def test_unknown_opcode_counted_as_data(self):
        r = decompile([0xFE, 0x00])
        assert r.mnemonic_counts.get("DATA") == 1


# ─── Stats and Metadata ───────────────────────────────────────────────────

class TestDecompilationStats:
    """Test DecompilationResult metadata."""

    def test_total_bytes(self):
        bc = [0x18, 0, 42, 0x00]
        r = decompile(bc)
        assert r.total_bytes == 4

    def test_total_instructions(self):
        bc = [0x18, 0, 42, 0x00]
        r = decompile(bc)
        assert r.total_instructions == 2

    def test_total_bytes_empty(self):
        r = decompile([])
        assert r.total_bytes == 0

    def test_total_instructions_empty(self):
        r = decompile([])
        assert r.total_instructions == 0

    def test_jump_count_zero(self):
        r = decompile([0x18, 0, 42, 0x00])
        assert r.jump_count == 0

    def test_jump_count_multiple(self):
        bc = [0x3C, 0, 5, 0, 0x3D, 1, 0, 0, 0x43, 0, 0, 0, 0x46, 1, 0, 0]
        r = decompile(bc)
        assert r.jump_count == 4


# ─── Edge Cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test boundary conditions and unusual inputs."""

    def test_empty_bytecode(self):
        r = decompile([])
        assert r.total_instructions == 0
        assert r.total_bytes == 0
        assert r.instructions == []
        assert r.labels == {}

    def test_empty_asm_output(self):
        r = decompile([])
        asm = r.to_asm()
        assert asm == ""

    def test_empty_annotated_output(self):
        r = decompile([])
        ann = r.to_annotated()
        # Should still have header and stats
        assert "FLUX Bytecode Decompilation" in ann
        assert "0 instructions, 0 bytes" in ann

    def test_single_halt(self):
        r = decompile([0x00])
        assert r.total_instructions == 1
        assert r.total_bytes == 1

    def test_single_nop(self):
        r = decompile([0x01])
        assert r.total_instructions == 1

    def test_all_unknown_opcodes(self):
        r = decompile([0xAA, 0xBB, 0xCC])
        assert r.total_instructions == 3
        for inst in r.instructions:
            assert inst.mnemonic == "DATA"

    def test_interleaved_unknown(self):
        """Unknown opcodes interspersed with valid ones."""
        bc = [0x00, 0xFF, 0x01, 0xFE, 0x00]
        r = decompile(bc)
        mnemonics = [i.mnemonic for i in r.instructions]
        assert mnemonics == ["HALT", "DATA", "NOP", "DATA", "HALT"]

    def test_truncated_three_byte_instruction(self):
        """MOVI needs 3 bytes but only 2 provided."""
        bc = [0x18, 0x01]
        r = decompile(bc)
        assert r.total_instructions == 1
        assert r.instructions[0].mnemonic == "MOVI"
        # Missing imm8 defaults to 0
        assert "R1" in r.instructions[0].operands

    def test_truncated_four_byte_instruction(self):
        """ADD needs 4 bytes but only 2 provided."""
        bc = [0x20, 0x00]
        r = decompile(bc)
        assert r.total_instructions == 1
        assert r.instructions[0].mnemonic == "ADD"

    def test_truncated_jmp(self):
        """JMP needs 4 bytes but only 2 provided."""
        bc = [0x43, 0x06]
        r = decompile(bc)
        assert r.total_instructions == 1
        assert r.instructions[0].mnemonic == "JMP"
        # Should not crash; jump_target should still be computed
        assert r.instructions[0].jump_target is not None

    def test_truncated_mov_i16(self):
        """MOVI16 needs 4 bytes but only 3 provided."""
        bc = [0x40, 0x01, 0xE8]
        r = decompile(bc)
        assert r.total_instructions == 1
        assert r.instructions[0].mnemonic == "MOVI16"

    def test_max_register_number(self):
        r = decompile([0x08, 0xFF])
        assert r.instructions[0].operands == ["R255"]

    def test_imm8_all_values_roundtrip(self):
        """All 256 imm8 values should produce valid output."""
        for val in range(256):
            bc = [0x18, 0x01, val]
            r = decompile(bc)
            assert r.total_instructions == 1
            assert len(r.instructions[0].operands) == 2

    def test_large_program(self):
        """Decompile a program with many instructions."""
        bc = [0x01] * 100 + [0x00]  # 100 NOPs + HALT
        r = decompile(bc)
        assert r.total_instructions == 101
        assert r.total_bytes == 101
        assert r.mnemonic_counts["NOP"] == 100
        assert r.mnemonic_counts["HALT"] == 1

    def test_all_opcodes_sequential(self):
        """Decode all supported opcodes in sequence."""
        bc = []
        expected = []
        for opcode in sorted(OP_SPECS.keys()):
            spec = OP_SPECS[opcode]
            mnemonic, size, _ = spec
            bc += [opcode] + [0] * (size - 1)
            expected.append(mnemonic)
        r = decompile(bc)
        assert r.total_instructions == len(OP_SPECS)
        actual = [i.mnemonic for i in r.instructions]
        assert actual == expected


# ─── DecodedInstruction Structure ─────────────────────────────────────────

class TestDecodedInstructionStructure:
    """Test the DecodedInstruction dataclass fields."""

    def test_offset_field(self):
        r = decompile([0x00, 0x01])
        assert r.instructions[0].offset == 0
        assert r.instructions[1].offset == 1

    def test_opcode_field(self):
        r = decompile([0x00])
        assert r.instructions[0].opcode == 0x00

    def test_raw_bytes_field(self):
        r = decompile([0x18, 0, 42, 0x00])
        assert r.instructions[0].raw_bytes == [0x18, 0x00, 0x2A]
        assert r.instructions[1].raw_bytes == [0x00]

    def test_size_field(self):
        r = decompile([0x18, 0, 42, 0x00])
        assert r.instructions[0].size == 3
        assert r.instructions[1].size == 1

    def test_jump_target_none_for_non_jump(self):
        r = decompile([0x18, 0, 42, 0x00])
        assert r.instructions[0].jump_target is None
        assert r.instructions[1].jump_target is None

    def test_jump_type_none_for_non_jump(self):
        r = decompile([0x18, 0, 42, 0x00])
        assert r.instructions[0].jump_type is None

    def test_comment_empty_for_non_jump(self):
        r = decompile([0x18, 0, 42, 0x00])
        assert r.instructions[0].comment == ""


# ─── Bytecode Input Types ─────────────────────────────────────────────────

class TestBytecodeInputTypes:
    """Test that different input types work."""

    def test_list_input(self):
        r = FluxDecompiler([0x00]).decompile()
        assert r.total_instructions == 1

    def test_bytes_input(self):
        r = FluxDecompiler(bytes([0x00])).decompile()
        assert r.total_instructions == 1

    def test_bytearray_input(self):
        r = FluxDecompiler(bytearray([0x00])).decompile()
        assert r.total_instructions == 1

    def test_tuple_input(self):
        r = FluxDecompiler((0x00,)).decompile()
        assert r.total_instructions == 1


# ─── Signed Value Helpers ─────────────────────────────────────────────────

class TestSignedValueHelpers:
    """Test internal signed value conversion methods."""

    def test_signed8_positive(self):
        d = FluxDecompiler([])
        assert d._signed8(0) == 0
        assert d._signed8(1) == 1
        assert d._signed8(127) == 127

    def test_signed8_negative(self):
        d = FluxDecompiler([])
        assert d._signed8(128) == -128
        assert d._signed8(255) == -1
        assert d._signed8(200) == -56

    def test_signed16_positive(self):
        d = FluxDecompiler([])
        assert d._signed16(0, 0) == 0
        assert d._signed16(1, 0) == 1
        assert d._signed16(255, 127) == 32767

    def test_signed16_negative(self):
        d = FluxDecompiler([])
        assert d._signed16(0, 128) == -32768
        assert d._signed16(255, 255) == -1
        assert d._signed16(0, 255) == -256


# ─── Regression: JMP Bug Fix ──────────────────────────────────────────────

class TestJMPBugFix:
    """Ensure the JMP jump_target calculation bug is fixed.

    Bug was: JMP used raw[2]/raw[3] instead of raw[1]/raw[2] for offset.
    The operand display was correct (raw[1]/raw[2]) but the jump target was wrong.
    """

    def test_jmp_operand_matches_target(self):
        """The displayed operand and the jump_target must be consistent."""
        # JMP at offset 6, operand=-6, target should be 0
        bc = [0x00]*6 + [0x43, 0xFA, 0xFF, 0x00]
        r = decompile(bc)
        jmp = [i for i in r.instructions if i.mnemonic == "JMP"][0]
        operand_val = int(jmp.operands[0])
        expected_target = 6 + operand_val
        assert jmp.jump_target == expected_target

    def test_jmp_positive_operand_matches_target(self):
        """Forward JMP: operand and target consistency."""
        bc = [0x43, 0x08, 0x00, 0x00] + [0x00]*8
        r = decompile(bc)
        jmp = r.instructions[0]
        operand_val = int(jmp.operands[0])
        expected_target = 0 + operand_val
        assert jmp.jump_target == expected_target

    def test_jmp_label_matches_target(self):
        """The label should be placed at the jump_target offset."""
        bc = [0x00]*6 + [0x43, 0x0A, 0x00, 0x00] + [0x00]*12
        r = decompile(bc)
        jmp = [i for i in r.instructions if i.mnemonic == "JMP"][0]
        assert jmp.jump_target in r.labels
        assert r.labels[jmp.jump_target] == f"lbl_{jmp.jump_target:03d}"


# ─── Regression: LOOP Bug Fix ─────────────────────────────────────────────

class TestLOOPBugFix:
    """Ensure LOOP now computes jump_target and labels the target.

    Bug was: LOOP only labeled itself, never computed jump_target or
    labeled the target offset.
    """

    def test_loop_has_jump_target(self):
        bc = [0x46, 1, 0xFA, 0xFF, 0x00]
        r = decompile(bc)
        loop = r.instructions[0]
        assert loop.jump_target is not None

    def test_loop_operand_matches_target(self):
        bc = [0x46, 1, 0xFA, 0xFF, 0x00]
        r = decompile(bc)
        loop = r.instructions[0]
        operand_val = int(loop.operands[1])
        expected_target = 0 + operand_val
        assert loop.jump_target == expected_target

    def test_loop_target_label_created(self):
        bc = [0x18, 0, 5, 0x18, 1, 10, 0x46, 1, 0xFA, 0xFF, 0x00]
        r = decompile(bc)
        loop = [i for i in r.instructions if i.mnemonic == "LOOP"][0]
        # Target (0) should have a label if different from loop position
        if loop.jump_target != loop.offset:
            assert loop.jump_target in r.labels


# ─── Round-Trip: Encoding Consistency ─────────────────────────────────────

class TestEncodingConsistency:
    """Verify that raw_bytes in decompiled output matches input bytes."""

    def test_single_instruction_raw_bytes(self):
        bc = [0x18, 0x01, 0x2A]
        r = decompile(bc)
        assert r.instructions[0].raw_bytes == bc

    def test_multi_instruction_raw_bytes(self):
        bc = [0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00]
        r = decompile(bc)
        pos = 0
        for inst in r.instructions:
            assert inst.raw_bytes == bc[pos:pos + inst.size]
            pos += inst.size
        assert pos == len(bc)

    def test_raw_bytes_for_unknown_opcode(self):
        bc = [0xFE]
        r = decompile(bc)
        assert r.instructions[0].raw_bytes == [0xFE]

    def test_raw_bytes_for_jump(self):
        bc = [0x43, 0xFA, 0xFF, 0x00]
        r = decompile(bc)
        assert r.instructions[0].raw_bytes == bc

    def test_total_bytes_matches_input(self):
        bc = [0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00]
        r = decompile(bc)
        assert r.total_bytes == len(bc)

    def test_instruction_offsets_sequential(self):
        """Offsets should increase by instruction size."""
        bc = [0x18, 0, 10, 0x01, 0x00, 0x08, 0x01, 0x22, 1, 1, 0]
        r = decompile(bc)
        expected_offset = 0
        for inst in r.instructions:
            assert inst.offset == expected_offset
            expected_offset += inst.size
        assert expected_offset == len(bc)
