"""Shared fixtures and helpers for flux-decompiler tests."""
import pytest
from decompiler import FluxDecompiler, OP_SPECS


def build_instruction(opcode: int, *operands) -> list[int]:
    """Build a raw bytecode instruction from opcode and operand bytes.

    Looks up the opcode in OP_SPECS to determine size, then pads
    or truncates the operand bytes to match the expected instruction size.
    """
    spec = OP_SPECS.get(opcode)
    if spec is None:
        return [opcode]
    _, size, _ = spec
    bc = [opcode] + list(operands[:size - 1])
    # Pad to expected size with zeros
    while len(bc) < size:
        bc.append(0)
    return bc


def decompile(bytecode: list[int]) -> "DecompilationResult":
    """Convenience: decompile bytecode and return result."""
    from decompiler import FluxDecompiler
    return FluxDecompiler(bytecode).decompile()


@pytest.fixture
def halt_bc():
    """Single HALT instruction."""
    return [0x00]


@pytest.fixture
def nop_bc():
    """Single NOP instruction."""
    return [0x01]


@pytest.fixture
def simple_program():
    """MOVI R0, 10; MOVI R1, 20; ADD R2, R0, R1; HALT"""
    return [0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00]


@pytest.fixture
def loop_program():
    """MOVI R0, 5; MOVI R1, 0; INC R1; DEC R0; JNZ R0, -4; HALT"""
    return [0x18, 0, 5, 0x18, 1, 0, 0x08, 1, 0x09, 0, 0x3D, 0, 0xFC, 0, 0x00]
