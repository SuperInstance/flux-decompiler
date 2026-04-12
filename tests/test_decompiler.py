"""Pytest test suite for flux-decompiler."""
import pytest
from decompiler import (
    FluxDecompiler, DecompilationResult, DecodedInstruction,
    JumpType, OP_SPECS, JUMP_OPS
)


# ── Fixtures ──

@pytest.fixture
def simple_halt():
    return [0x00]

@pytest.fixture
def movi_program():
    """MOVI R0, 42; HALT"""
    return [0x18, 0, 42, 0x00]

@pytest.fixture
def add_program():
    """MOVI R0, 10; MOVI R1, 20; ADD R2, R0, R1; HALT"""
    return [0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00]

@pytest.fixture
def loop_program():
    """Counter loop: MOVI R0, 5; MOVI R1, 0; INC R1; DEC R0; JNZ R0, -6; HALT"""
    return [0x18, 0, 5, 0x18, 1, 0, 0x08, 1, 0x09, 0, 0x3D, 0, 0xFC, 0, 0x00]


# ── Basic instruction decoding ──

class TestInstructionDecoding:
    def test_halt_decoded(self, simple_halt):
        d = FluxDecompiler(simple_halt)
        r = d.decompile()
        assert r.total_instructions == 1
        assert r.instructions[0].mnemonic == "HALT"
        assert r.instructions[0].opcode == 0x00

    def test_nop_decoded(self):
        d = FluxDecompiler([0x01])
        r = d.decompile()
        assert r.instructions[0].mnemonic == "NOP"

    def test_movi_decoded(self, movi_program):
        d = FluxDecompiler(movi_program)
        r = d.decompile()
        inst = r.instructions[0]
        assert inst.mnemonic == "MOVI"
        assert "R0" in inst.operands
        assert "42" in inst.operands

    def test_add_decoded(self, add_program):
        d = FluxDecompiler(add_program)
        r = d.decompile()
        adds = [i for i in r.instructions if i.mnemonic == "ADD"]
        assert len(adds) == 1
        assert adds[0].operands == ["R2", "R0", "R1"]

    def test_inc_decoded(self):
        d = FluxDecompiler([0x08, 5, 0x00])
        r = d.decompile()
        assert r.instructions[0].mnemonic == "INC"
        assert r.instructions[0].operands == ["R5"]

    def test_dec_decoded(self):
        d = FluxDecompiler([0x09, 3, 0x00])
        r = d.decompile()
        assert r.instructions[0].mnemonic == "DEC"

    def test_push_pop_decoded(self):
        d = FluxDecompiler([0x0C, 0, 0x0D, 1, 0x00])
        r = d.decompile()
        assert r.instructions[0].mnemonic == "PUSH"
        assert r.instructions[1].mnemonic == "POP"


# ── Signed value handling ──

class TestSignedValues:
    def test_positive_imm8(self, movi_program):
        d = FluxDecompiler(movi_program)
        r = d.decompile()
        assert "42" in r.instructions[0].operands

    def test_negative_imm8(self):
        """MOVI with negative immediate (0xFB = -5 in signed)."""
        d = FluxDecompiler([0x18, 0, 0xFB, 0x00])
        r = d.decompile()
        assert "-5" in r.instructions[0].operands

    @pytest.mark.parametrize("value,expected_byte", [
        (0, 0), (1, 1), (127, 127), (-1, 0xFF), (-128, 0x80), (-5, 0xFB)
    ])
    def test_signed8_decode(self, value, expected_byte):
        d = FluxDecompiler([0x18, 0, expected_byte, 0x00])
        r = d.decompile()
        assert str(value) in r.instructions[0].operands


# ── Unknown opcode handling ──

class TestUnknownOpcodes:
    def test_unknown_opcode_becomes_data(self):
        d = FluxDecompiler([0xFE, 0x00])
        r = d.decompile()
        assert r.instructions[0].mnemonic == "DATA"
        assert r.instructions[0].comment == "unknown opcode"

    def test_unknown_opcode_in_middle(self):
        d = FluxDecompiler([0x18, 0, 42, 0xFE, 0x00])
        r = d.decompile()
        data_instrs = [i for i in r.instructions if i.mnemonic == "DATA"]
        assert len(data_instrs) == 1


# ── Jump handling ──

class TestJumpHandling:
    def test_jnz_creates_label(self, loop_program):
        d = FluxDecompiler(loop_program)
        r = d.decompile()
        assert r.jump_count > 0
        assert len(r.labels) > 0

    def test_jz_creates_label(self):
        d = FluxDecompiler([0x18, 0, 1, 0x3C, 0, -3 & 0xFF, 0, 0x00])
        r = d.decompile()
        jz = [i for i in r.instructions if i.mnemonic == "JZ"]
        assert len(jz) == 1
        assert jz[0].jump_type == JumpType.CONDITIONAL

    def test_jnz_jump_type(self, loop_program):
        d = FluxDecompiler(loop_program)
        r = d.decompile()
        jnz = [i for i in r.instructions if i.mnemonic == "JNZ"]
        assert len(jnz) == 1
        assert jnz[0].jump_type == JumpType.CONDITIONAL

    def test_loop_jump_type(self):
        d = FluxDecompiler([0x18, 0, 5, 0x46, 0, -2 & 0xFF, 0, 0x00])
        r = d.decompile()
        loop = [i for i in r.instructions if i.mnemonic == "LOOP"]
        assert len(loop) == 1
        assert loop[0].jump_type == JumpType.LOOP

    def test_loop_creates_label_at_loop_offset(self):
        d = FluxDecompiler([0x18, 0, 5, 0x46, 0, -2 & 0xFF, 0, 0x00])
        r = d.decompile()
        # LOOP is at offset 3 (after 3-byte MOVI), label is at the LOOP's offset
        assert len(r.labels) > 0


# ── DecompilationResult ──

class TestDecompilationResult:
    def test_total_bytes(self, add_program):
        d = FluxDecompiler(add_program)
        r = d.decompile()
        assert r.total_bytes == 11

    def test_total_instructions(self, add_program):
        d = FluxDecompiler(add_program)
        r = d.decompile()
        assert r.total_instructions == 4  # 2 MOVI + ADD + HALT

    def test_mnemonic_counts(self, add_program):
        d = FluxDecompiler(add_program)
        r = d.decompile()
        assert r.mnemonic_counts.get("MOVI", 0) == 2
        assert r.mnemonic_counts.get("ADD", 0) == 1
        assert r.mnemonic_counts.get("HALT", 0) == 1


# ── Assembly output ──

class TestAssemblyOutput:
    def test_to_asm_contains_mnemonics(self, movi_program):
        d = FluxDecompiler(movi_program)
        r = d.decompile()
        asm = r.to_asm()
        assert "MOVI" in asm
        assert "HALT" in asm

    def test_to_asm_contains_offsets(self, movi_program):
        d = FluxDecompiler(movi_program)
        r = d.decompile()
        asm = r.to_asm()
        assert "0:" in asm

    def test_to_annotated_has_stats(self, movi_program):
        d = FluxDecompiler(movi_program)
        r = d.decompile()
        ann = r.to_annotated()
        assert "Stats" in ann
        assert "instructions" in ann
        assert "bytes" in ann

    def test_to_annotated_has_control_flow_markers(self, loop_program):
        d = FluxDecompiler(loop_program)
        r = d.decompile()
        ann = r.to_annotated()
        # JNZ is conditional, should have ↕ marker
        assert "↕" in ann

    def test_to_annotated_has_labels(self, loop_program):
        d = FluxDecompiler(loop_program)
        r = d.decompile()
        ann = r.to_annotated()
        assert "lbl_" in ann

    def test_to_asm_has_hex_bytes(self, movi_program):
        d = FluxDecompiler(movi_program)
        r = d.decompile()
        asm = r.to_asm()
        assert "18" in asm  # MOVI opcode in hex


# ── Instruction metadata ──

class TestInstructionMetadata:
    def test_raw_bytes(self, movi_program):
        d = FluxDecompiler(movi_program)
        r = d.decompile()
        assert r.instructions[0].raw_bytes == [0x18, 0, 42]

    def test_instruction_size(self, movi_program):
        d = FluxDecompiler(movi_program)
        r = d.decompile()
        assert r.instructions[0].size == 3  # MOVI is 3 bytes

    def test_halt_size(self, simple_halt):
        d = FluxDecompiler(simple_halt)
        r = d.decompile()
        assert r.instructions[0].size == 1

    def test_jnz_comment(self, loop_program):
        d = FluxDecompiler(loop_program)
        r = d.decompile()
        jnz = [i for i in r.instructions if i.mnemonic == "JNZ"]
        assert len(jnz) == 1
        assert "goto" in jnz[0].comment


# ── OP_SPECS coverage ──

class TestOpSpecs:
    def test_halt_in_specs(self):
        assert 0x00 in OP_SPECS

    def test_all_jump_ops_in_specs(self):
        for op in JUMP_OPS:
            assert op in OP_SPECS

    @pytest.mark.parametrize("opcode", [0x00, 0x01, 0x18, 0x20, 0x22, 0x3A, 0x3C, 0x3D])
    def test_known_opcodes_decode(self, opcode):
        """Verify that well-known opcodes all have specs."""
        assert opcode in OP_SPECS


# ── Factorial program ──

class TestFactorialProgram:
    @pytest.fixture
    def factorial_bc(self):
        return [0x18, 0, 6, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00]

    def test_has_mul(self, factorial_bc):
        d = FluxDecompiler(factorial_bc)
        r = d.decompile()
        assert "MUL" in [i.mnemonic for i in r.instructions]

    def test_has_jnz(self, factorial_bc):
        d = FluxDecompiler(factorial_bc)
        r = d.decompile()
        assert "JNZ" in [i.mnemonic for i in r.instructions]

    def test_has_label(self, factorial_bc):
        d = FluxDecompiler(factorial_bc)
        r = d.decompile()
        assert len(r.labels) > 0
