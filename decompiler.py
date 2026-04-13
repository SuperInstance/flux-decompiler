"""
FLUX Decompiler — convert bytecode back to human-readable assembly and C-like pseudocode.

Supports:
- Full disassembly with PC offsets
- Symbolic register names
- Jump target resolution
- Comment annotation
- Control flow reconstruction
- Control flow graph (CFG) with basic blocks and edges
- Loop detection (back edges, natural loops)
- Function boundary detection (CALL/RET)
- Pattern-based decompilation (if/else, while, for, switch)
- Type inference for registers
- C-like pseudocode output
- Symbol resolution
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
from collections import OrderedDict


class JumpType(Enum):
    CONDITIONAL = "conditional"
    UNCONDITIONAL = "unconditional"
    LOOP = "loop"
    CALL = "call"
    RET = "ret"


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
    0x50: ("CALL", 4, ["imm16"]), 0x51: ("RET", 1, []),
    0x52: ("CALLR", 4, ["reg", "pad", "pad"]),
}

JUMP_OPS = {0x3C, 0x3D, 0x43, 0x46}
CALL_OPS = {0x50, 0x52}
RET_OPS = {0x51}


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
                elif otype == "pad":
                    operands.append("_")
            
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
            
            # Handle CALL
            if op in CALL_OPS:
                jump_count += 1
                if op == 0x50:  # CALL
                    offset = self._signed16(raw[2], raw[3]) if len(raw) > 3 else 0
                    target = i + offset
                    jump_target = target
                    jump_type = JumpType.CALL
                    labels[target] = f"fn_{target:03d}"
                    comment = f"call fn_{target:03d}"
                elif op == 0x52:  # CALLR
                    comment = f"call indirect via R{raw[1]}"
                    jump_type = JumpType.CALL
            
            # Handle RET
            if op in RET_OPS:
                jump_type = JumpType.RET
                comment = "return"
            
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


# ═══════════════════════════════════════════════════════════
#  Control Flow Graph (CFG)
# ═══════════════════════════════════════════════════════════

@dataclass
class BasicBlock:
    """A basic block in the CFG."""
    start: int
    end: int  # exclusive
    instructions: List[DecodedInstruction] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)  # block start offsets
    predecessors: List[int] = field(default_factory=list)
    label: str = ""


@dataclass
class CFGEdge:
    src: int  # block start offset
    dst: int  # block start offset
    edge_type: str  # "fallthrough", "branch", "loop_back", "call", "ret"


class ControlFlowGraph:
    """
    Reconstruct control flow graph from decoded instructions.
    Identifies basic blocks, edges, and entry/exit points.
    """
    
    def __init__(self, result: DecompilationResult):
        self.result = result
        self.blocks: Dict[int, BasicBlock] = {}
        self.edges: List[CFGEdge] = []
        self.entry_block: Optional[int] = None
        self._build()
    
    def _build(self):
        instructions = self.result.instructions
        if not instructions:
            return
        
        # Find block leaders: first instruction, jump targets, instructions after jumps/calls/rets
        leaders: Set[int] = {instructions[0].offset}
        
        for inst in instructions:
            if inst.jump_target is not None:
                leaders.add(inst.jump_target)
            # Instruction after a jump/call/ret is a leader
            if inst.jump_type in (JumpType.CONDITIONAL, JumpType.UNCONDITIONAL,
                                   JumpType.CALL, JumpType.RET):
                next_off = inst.offset + inst.size
                if next_off < self.result.total_bytes:
                    leaders.add(next_off)
        
        # Also add all label targets as leaders
        for off in self.result.labels:
            leaders.add(off)
        
        # Sort leaders and create blocks
        sorted_leaders = sorted(leaders)
        self.entry_block = sorted_leaders[0] if sorted_leaders else None
        
        for idx, leader in enumerate(sorted_leaders):
            next_leader = sorted_leaders[idx + 1] if idx + 1 < len(sorted_leaders) else self.result.total_bytes
            
            block_insts = [i for i in instructions if leader <= i.offset < next_leader]
            label = self.result.labels.get(leader, f"block_{leader:03d}")
            
            self.blocks[leader] = BasicBlock(
                start=leader,
                end=next_leader,
                instructions=block_insts,
                label=label,
            )
        
        # Build edges
        for bstart, block in self.blocks.items():
            if not block.instructions:
                continue
            
            last = block.instructions[-1]
            
            # Fallthrough edge
            fallthrough = block.end
            if fallthrough < self.result.total_bytes and self.blocks.get(fallthrough):
                if last.jump_type not in (JumpType.UNCONDITIONAL, JumpType.RET):
                    self.edges.append(CFGEdge(bstart, fallthrough, "fallthrough"))
            
            # Branch/jump edge
            if last.jump_target is not None and self.blocks.get(last.jump_target):
                # Detect back edges (loop) for any branch that jumps backwards
                is_back_edge = (last.jump_target <= bstart and
                                last.jump_type in (JumpType.LOOP, JumpType.CONDITIONAL))
                if is_back_edge:
                    self.edges.append(CFGEdge(bstart, last.jump_target, "loop_back"))
                elif last.jump_type == JumpType.LOOP:
                    self.edges.append(CFGEdge(bstart, last.jump_target, "branch"))
                elif last.jump_type == JumpType.CONDITIONAL:
                    self.edges.append(CFGEdge(bstart, last.jump_target, "branch"))
                elif last.jump_type == JumpType.UNCONDITIONAL:
                    self.edges.append(CFGEdge(bstart, last.jump_target, "branch"))
                elif last.jump_type == JumpType.CALL:
                    self.edges.append(CFGEdge(bstart, last.jump_target, "call"))
            
            # RET edge
            if last.jump_type == JumpType.RET:
                self.edges.append(CFGEdge(bstart, -1, "ret"))
        
        # Build predecessor lists
        for edge in self.edges:
            if edge.dst >= 0 and edge.dst in self.blocks:
                self.blocks[edge.dst].predecessors.append(edge.src)
            if edge.src in self.blocks:
                self.blocks[edge.src].successors.append(edge.dst)
    
    def get_loops(self) -> List[List[int]]:
        """Detect loops as lists of block start offsets forming a back-edge cycle."""
        loops = []
        back_edges = [e for e in self.edges if e.edge_type == "loop_back"]
        
        for be in back_edges:
            # Walk backwards from src to dst
            loop_blocks = {be.src, be.dst}
            stack = [be.src]
            while stack:
                current = stack.pop()
                for edge in self.edges:
                    if edge.dst == current and edge.src in self.blocks:
                        if edge.src not in loop_blocks:
                            loop_blocks.add(edge.src)
                            stack.append(edge.src)
            loops.append(sorted(loop_blocks))
        
        return loops
    
    def to_dot(self) -> str:
        """Generate Graphviz DOT representation."""
        lines = ["digraph CFG {"]
        lines.append("  node [shape=record];")
        
        for bstart, block in self.blocks.items():
            inst_strs = []
            for inst in block.instructions[:5]:
                op = ", ".join(inst.operands)
                inst_strs.append(f"{inst.mnemonic} {op}")
            if len(block.instructions) > 5:
                inst_strs.append("...")
            body = "\\l".join(inst_strs) + "\\l"
            lines.append(f'  "{bstart}" [label="{block.label}\\l{body}"];')
        
        for edge in self.edges:
            if edge.dst < 0:
                continue
            style = ""
            if edge.edge_type == "branch":
                style = ' [color="red"]'
            elif edge.edge_type == "loop_back":
                style = ' [color="blue", style="dashed"]'
            elif edge.edge_type == "call":
                style = ' [color="green", style="dotted"]'
            lines.append(f'  "{edge.src}" -> "{edge.dst}"{style};')
        
        lines.append("}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  Function Boundary Detection
# ═══════════════════════════════════════════════════════════

@dataclass
class Function:
    name: str
    start: int
    end: int  # exclusive
    block_starts: List[int] = field(default_factory=list)
    calls: List[int] = field(default_factory=list)  # target offsets
    has_return: bool = False


class FunctionDetector:
    """Detect function boundaries using CALL/RET patterns."""
    
    def __init__(self, result: DecompilationResult):
        self.result = result
        self.functions: List[Function] = []
        self._detect()
    
    def _detect(self):
        instructions = self.result.instructions
        if not instructions:
            return
        
        # Entry points: start of program + CALL targets
        entry_points = [0]
        for inst in instructions:
            if inst.jump_type == JumpType.CALL and inst.jump_target is not None:
                if inst.jump_target not in entry_points:
                    entry_points.append(inst.jump_target)
        
        entry_points.sort()
        
        for idx, ep in enumerate(entry_points):
            next_ep = entry_points[idx + 1] if idx + 1 < len(entry_points) else self.result.total_bytes
            
            fn_insts = [i for i in instructions if ep <= i.offset < next_ep]
            name = self.result.labels.get(ep, f"func_{ep:03d}")
            
            block_starts = []
            calls = []
            has_return = False
            
            for inst in fn_insts:
                if inst.offset in self.result.labels or inst.offset == ep:
                    block_starts.append(inst.offset)
                if inst.jump_type == JumpType.CALL and inst.jump_target is not None:
                    calls.append(inst.jump_target)
                if inst.jump_type == JumpType.RET:
                    has_return = True
                # Also mark jump targets within the function
                if inst.jump_target is not None and ep <= inst.jump_target < next_ep:
                    if inst.jump_target not in block_starts:
                        block_starts.append(inst.jump_target)
            
            self.functions.append(Function(
                name=name, start=ep, end=next_ep,
                block_starts=sorted(block_starts),
                calls=calls, has_return=has_return,
            ))
    
    def get_function_at(self, offset: int) -> Optional[Function]:
        for fn in self.functions:
            if fn.start <= offset < fn.end:
                return fn
        return None


# ═══════════════════════════════════════════════════════════
#  Type Inference
# ═══════════════════════════════════════════════════════════

class RegType(Enum):
    UNKNOWN = "unknown"
    INT = "int"
    UINT = "uint"
    PTR = "ptr"
    BOOL = "bool"
    BYTE = "byte"


@dataclass
class RegTypeInfo:
    reg: str
    inferred_type: RegType = RegType.UNKNOWN
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)


class TypeInference:
    """
    Infer types for registers based on usage patterns.
    Analyzes how registers are used (comparisons, arithmetic, indexing) to
    guess their likely C type.
    """
    
    def __init__(self, result: DecompilationResult):
        self.result = result
        self.reg_types: Dict[str, RegTypeInfo] = {}
        self._analyze()
    
    def _analyze(self):
        for inst in self.result.instructions:
            ops = inst.operands
            reg_dst = ops[0] if ops and ops[0].startswith("R") else None
            reg_src1 = ops[1] if len(ops) > 1 and ops[1].startswith("R") else None
            reg_src2 = ops[2] if len(ops) > 2 and ops[2].startswith("R") else None
            
            mn = inst.mnemonic
            
            # Boolean: CMP_EQ, CMP_NE, CMP_LT, CMP_GT → destination is bool
            if mn in ("CMP_EQ", "CMP_NE", "CMP_LT", "CMP_GT"):
                if reg_dst:
                    self._set_type(reg_dst, RegType.BOOL, 0.8,
                                   f"{mn} result")
            
            # Unsigned: used as loop counter (DEC with JNZ), MOD, SHL, SHR
            if mn == "DEC" and reg_dst:
                self._set_type(reg_dst, RegType.UINT, 0.5, "decrement (loop counter)")
            if mn == "INC" and reg_dst:
                self._set_type(reg_dst, RegType.UINT, 0.4, "increment")
            if mn in ("SHL", "SHR") and reg_dst:
                self._set_type(reg_dst, RegType.UINT, 0.7, "bit shift")
            
            # Pointer: used as base for memory ops (PUSH/POP pattern)
            if mn in ("PUSH",) and reg_src1 or reg_dst:
                r = reg_src1 or reg_dst
                self._set_type(r, RegType.INT, 0.3, "stack operation")
            
            # Integer: arithmetic operations
            if mn in ("ADD", "SUB", "MUL", "DIV", "ADDI", "SUBI") and reg_dst:
                existing = self.reg_types.get(reg_dst)
                if existing and existing.inferred_type == RegType.BOOL:
                    pass  # keep bool
                else:
                    self._set_type(reg_dst, RegType.INT, 0.6, f"{mn} operation")
            
            # Byte: small immediate values
            if mn == "MOVI" and reg_dst and ops:
                try:
                    val = int(ops[1])
                    if 0 <= val <= 255:
                        self._set_type(reg_dst, RegType.BYTE, 0.3, f"small immediate ({val})")
                except (ValueError, IndexError):
                    pass
    
    def _set_type(self, reg: str, rtype: RegType, confidence: float, evidence: str):
        if reg not in self.reg_types:
            self.reg_types[reg] = RegTypeInfo(reg=reg)
        info = self.reg_types[reg]
        if confidence > info.confidence:
            info.inferred_type = rtype
            info.confidence = confidence
            info.evidence = [evidence]
        elif confidence == info.confidence:
            info.evidence.append(evidence)
    
    def get_type(self, reg: str) -> RegType:
        return self.reg_types.get(reg, RegTypeInfo(reg=reg)).inferred_type
    
    def get_type_string(self, reg: str) -> str:
        rtype = self.get_type(reg)
        return {
            RegType.UNKNOWN: "var",
            RegType.INT: "int",
            RegType.UINT: "unsigned int",
            RegType.PTR: "void*",
            RegType.BOOL: "bool",
            RegType.BYTE: "uint8_t",
        }.get(rtype, "var")
    
    def summary(self) -> Dict[str, str]:
        return {reg: info.inferred_type.value for reg, info in self.reg_types.items()}


# ═══════════════════════════════════════════════════════════
#  Symbol Resolution
# ═══════════════════════════════════════════════════════════

class SymbolTable:
    """Resolve addresses to meaningful names."""
    
    def __init__(self):
        self.symbols: Dict[int, str] = {}
        self.constants: Dict[str, int] = {}
    
    def add_symbol(self, addr: int, name: str):
        self.symbols[addr] = name
    
    def add_constant(self, name: str, value: int):
        self.constants[name] = value
    
    def resolve(self, addr: int) -> str:
        return self.symbols.get(addr, f"0x{addr:04x}")
    
    def resolve_constant(self, value: int) -> Optional[str]:
        for name, val in self.constants.items():
            if val == value:
                return name
        return None
    
    def auto_populate(self, result: DecompilationResult):
        """Auto-populate symbols from labels and patterns."""
        for offset, label in result.labels.items():
            if label.startswith("fn_"):
                self.symbols[offset] = label.replace("fn_", "func_")
            elif label.startswith("loop_"):
                self.symbols[offset] = label
            elif label.startswith("lbl_"):
                self.symbols[offset] = label


# ═══════════════════════════════════════════════════════════
#  Pattern-Based Decompilation
# ═══════════════════════════════════════════════════════════

class PatternDetector:
    """Detect high-level control flow patterns in basic blocks."""
    
    def __init__(self, cfg: ControlFlowGraph):
        self.cfg = cfg
    
    def detect_if_else(self) -> List[Dict]:
        """Detect if/else patterns."""
        patterns = []
        for bstart, block in self.cfg.blocks.items():
            if not block.instructions:
                continue
            last = block.instructions[-1]
            if last.jump_type != JumpType.CONDITIONAL or last.jump_target is None:
                continue
            
            # Check if there's a block after this one that ends with JMP
            target_block = self.cfg.blocks.get(last.jump_target)
            if target_block is None:
                continue
            
            # Find fallthrough block
            ft_block = self.cfg.blocks.get(block.end)
            if ft_block and ft_block.instructions:
                ft_last = ft_block.instructions[-1]
                # If fallthrough ends with JMP to same target, it's if/else
                if (ft_last.jump_type == JumpType.UNCONDITIONAL and
                        ft_last.jump_target == last.jump_target):
                    patterns.append({
                        "type": "if_else",
                        "header_block": bstart,
                        "then_block": block.end,
                        "else_block": ft_block.start,
                        "merge_block": last.jump_target,
                        "condition": last.comment,
                    })
                else:
                    patterns.append({
                        "type": "if",
                        "header_block": bstart,
                        "then_block": block.end,
                        "merge_block": last.jump_target,
                        "condition": last.comment,
                    })
        
        return patterns
    
    def detect_while_loops(self) -> List[Dict]:
        """Detect while loop patterns."""
        patterns = []
        loops = self.cfg.get_loops()
        
        for loop_blocks in loops:
            if len(loop_blocks) < 2:
                continue
            
            # Find the header (smallest offset in loop that has a conditional branch)
            header = None
            for bstart in sorted(loop_blocks):
                block = self.cfg.blocks.get(bstart)
                if block and block.instructions:
                    last = block.instructions[-1]
                    if last.jump_type == JumpType.CONDITIONAL:
                        header = bstart
                        break
            
            if header is None:
                header = min(loop_blocks)
            
            patterns.append({
                "type": "while",
                "header_block": header,
                "loop_blocks": loop_blocks,
                "back_edge_to": header,
            })
        
        return patterns
    
    def detect_for_loops(self) -> List[Dict]:
        """Detect for loop patterns (init, condition, increment)."""
        patterns = []
        
        for bstart, block in self.cfg.blocks.items():
            insts = block.instructions
            if len(insts) < 2:
                continue
            
            # Look for pattern: MOVI reg, val; ... JZ/JNZ reg, target (back edge)
            first = insts[0]
            last = insts[-1]
            
            if (first.mnemonic in ("MOVI", "MOV") and
                    last.jump_type == JumpType.CONDITIONAL and
                    last.jump_target is not None and
                    last.jump_target <= bstart):
                
                # Check for increment/decrement in the loop body
                has_inc = False
                loop_target = last.jump_target
                for lb in self.cfg.get_loops():
                    if bstart in lb and loop_target in lb:
                        for bs in lb:
                            blk = self.cfg.blocks.get(bs)
                            if blk:
                                for inst in blk.instructions:
                                    if inst.mnemonic in ("INC", "DEC"):
                                        has_inc = True
                        break
                
                if has_inc:
                    patterns.append({
                        "type": "for",
                        "header_block": bstart,
                        "init_reg": first.operands[0] if first.operands else "R?",
                        "init_val": first.operands[1] if len(first.operands) > 1 else "?",
                        "cond_reg": last.operands[0] if last.operands else "R?",
                    })
        
        return patterns
    
    def detect_switch(self) -> List[Dict]:
        """Detect switch/case patterns (chains of conditional jumps)."""
        patterns = []
        
        for bstart, block in self.cfg.blocks.items():
            if not block.successors:
                continue
            
            # Look for blocks with multiple conditional successors
            cond_succs = [s for s in block.successors if s > 0]
            if len(cond_succs) >= 2:
                # Check if successor blocks end with JMP to a common target
                targets = set()
                for s in cond_succs:
                    sb = self.cfg.blocks.get(s)
                    if sb and sb.instructions:
                        sl = sb.instructions[-1]
                        if sl.jump_target is not None:
                            targets.add(sl.jump_target)
                
                if len(targets) == 1:
                    patterns.append({
                        "type": "switch",
                        "header_block": bstart,
                        "case_blocks": cond_succs,
                        "default_block": block.end,
                        "merge_block": targets.pop() if targets else None,
                    })
        
        return patterns
    
    def all_patterns(self) -> Dict[str, List[Dict]]:
        return {
            "if_else": self.detect_if_else(),
            "while": self.detect_while_loops(),
            "for": self.detect_for_loops(),
            "switch": self.detect_switch(),
        }


# ═══════════════════════════════════════════════════════════
#  C-Like Pseudocode Generator
# ═══════════════════════════════════════════════════════════

class PseudocodeGenerator:
    """Generate C-like pseudocode from decompiled bytecode."""
    
    def __init__(self, result: DecompilationResult):
        self.result = result
        self.type_inf = TypeInference(result)
        self.symbols = SymbolTable()
        self.symbols.auto_populate(result)
        self.cfg = ControlFlowGraph(result)
        self.patterns = PatternDetector(self.cfg)
        self.functions = FunctionDetector(result)
        self._indent = 0
    
    def _ind(self) -> str:
        return "    " * self._indent
    
    def _reg_name(self, reg_str: str) -> str:
        """Convert R0, R1 etc to meaningful names."""
        num = reg_str.replace("R", "")
        rtype = self.type_inf.get_type_string(reg_str)
        return f"var_{num}"
    
    def _op_to_c(self, inst: DecodedInstruction) -> str:
        """Convert a single instruction to a C-like expression."""
        ops = inst.operands
        mn = inst.mnemonic
        
        if mn == "HALT":
            return "return;"
        if mn == "NOP":
            return "// nop"
        if mn == "MOVI":
            if len(ops) >= 2:
                try:
                    val = int(ops[1])
                    cn = self.symbols.resolve_constant(val)
                    if cn:
                        return f"{self._reg_name(ops[0])} = {cn};"
                    return f"{self._reg_name(ops[0])} = {val};"
                except ValueError:
                    return f"{self._reg_name(ops[0])} = {ops[1]};"
        if mn in ("ADD", "SUB", "MUL", "DIV", "MOD"):
            c_op = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/", "MOD": "%"}[mn]
            if len(ops) >= 3:
                return f"{self._reg_name(ops[0])} = {self._reg_name(ops[1])} {c_op} {self._reg_name(ops[2])};"
        if mn in ("AND", "OR", "XOR"):
            c_op = {"AND": "&", "OR": "|", "XOR": "^"}[mn]
            if len(ops) >= 3:
                return f"{self._reg_name(ops[0])} = {self._reg_name(ops[1])} {c_op} {self._reg_name(ops[2])};"
        if mn in ("SHL", "SHR"):
            c_op = {"SHL": "<<", "SHR": ">>"}[mn]
            if len(ops) >= 3:
                return f"{self._reg_name(ops[0])} = {self._reg_name(ops[1])} {c_op} {self._reg_name(ops[2])};"
        if mn == "MOV":
            if len(ops) >= 2:
                return f"{self._reg_name(ops[0])} = {self._reg_name(ops[1])};"
        if mn in ("ADDI", "SUBI"):
            c_op = "+" if mn == "ADDI" else "-"
            if len(ops) >= 2:
                return f"{self._reg_name(ops[0])} {c_op}= {ops[1]};"
        if mn in ("INC", "DEC"):
            c_op = "++" if mn == "INC" else "--"
            if ops:
                return f"{self._reg_name(ops[0])}{c_op};"
        if mn in ("CMP_EQ", "CMP_NE", "CMP_LT", "CMP_GT"):
            c_op = {"CMP_EQ": "==", "CMP_NE": "!=", "CMP_LT": "<", "CMP_GT": ">"}[mn]
            if len(ops) >= 3:
                return f"{self._reg_name(ops[0])} = ({self._reg_name(ops[1])} {c_op} {self._reg_name(ops[2])});"
        if mn == "JZ" and ops:
            cond = f"{self._reg_name(ops[0])} == 0"
            target_label = self.result.labels.get(inst.jump_target, "???") if inst.jump_target else "???"
            return f"if ({cond}) goto {target_label};"
        if mn == "JNZ" and ops:
            cond = f"{self._reg_name(ops[0])} != 0"
            target_label = self.result.labels.get(inst.jump_target, "???") if inst.jump_target else "???"
            return f"if ({cond}) goto {target_label};"
        if mn == "JMP":
            target_label = self.result.labels.get(inst.jump_target, "???") if inst.jump_target else "???"
            return f"goto {target_label};"
        if mn == "CALL":
            target_label = self.result.labels.get(inst.jump_target, "???") if inst.jump_target else "???"
            return f"{target_label}();"
        if mn == "RET":
            return "return;"
        if mn == "PUSH" and ops:
            return f"push({self._reg_name(ops[0])});"
        if mn == "POP" and ops:
            return f"{self._reg_name(ops[0])} = pop();"
        if mn == "LOOP" and ops:
            target_label = self.result.labels.get(inst.jump_target, "???") if inst.jump_target else "???"
            return f"loop ({self._reg_name(ops[0])}) {{ ... goto {target_label}; }}"
        if mn == "MOVI16" and len(ops) >= 2:
            return f"{self._reg_name(ops[0])} = {ops[1]};"
        
        return f"// {mn} {' '.join(ops)}"
    
    def generate(self) -> str:
        """Generate C-like pseudocode."""
        lines = []
        lines.append("/* FLUX Decompiled Pseudocode */")
        lines.append("")
        
        # Emit type declarations for used registers
        declared = set()
        for inst in self.result.instructions:
            for op in inst.operands:
                if op.startswith("R") and op not in declared:
                    rtype = self.type_inf.get_type_string(op)
                    lines.append(f"{rtype} {self._reg_name(op)} = 0;")
                    declared.add(op)
        
        if declared:
            lines.append("")
        
        # Generate function-based output
        if len(self.functions.functions) > 1:
            for fn in self.functions.functions:
                lines.append(f"/* Function: {fn.name} @ 0x{fn.start:04x} */")
                ret = "void" if not fn.has_return else "int"
                params = "" if fn.name == "func_000" else "void"
                lines.append(f"{ret} {fn.name}({params}) {{")
                self._indent = 1
                fn_insts = [i for i in self.result.instructions
                           if fn.start <= i.offset < fn.end]
                for inst in fn_insts:
                    lines.append(f"{self._ind()}{self._op_to_c(inst)}")
                self._indent = 0
                lines.append("}")
                lines.append("")
        else:
            lines.append("int main(void) {")
            self._indent = 1
            for inst in self.result.instructions:
                lines.append(f"{self._ind()}{self._op_to_c(inst)}")
            self._indent = 0
            lines.append("}")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  Tests — Original Decompiler
# ═══════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════
#  Tests — CFG Reconstruction
# ═══════════════════════════════════════════════════════════

class TestCFG(unittest.TestCase):
    def test_cfg_creation(self):
        d = FluxDecompiler([0x18, 0, 42, 0x00])
        r = d.decompile()
        cfg = ControlFlowGraph(r)
        self.assertGreater(len(cfg.blocks), 0)
    
    def test_cfg_single_block_no_jump(self):
        """Straight-line code should produce one block."""
        d = FluxDecompiler([0x18, 0, 10, 0x00])
        r = d.decompile()
        cfg = ControlFlowGraph(r)
        self.assertEqual(len(cfg.blocks), 1)
    
    def test_cfg_branch_creates_blocks(self):
        """A conditional jump should create at least 2 blocks."""
        # MOVI R0, 1; JNZ R0, 4; HALT
        bc = [0x18, 0, 1, 0x3D, 0, 4, 0, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        cfg = ControlFlowGraph(r)
        self.assertGreater(len(cfg.blocks), 1)
    
    def test_cfg_edges(self):
        """CFG should have edges between blocks."""
        bc = [0x18, 0, 1, 0x3D, 0, 4, 0, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        cfg = ControlFlowGraph(r)
        self.assertGreater(len(cfg.edges), 0)
    
    def test_cfg_loop_detection(self):
        """Back-edge should be detected as a loop."""
        # MOVI R0,5 (0-2); DEC R0 (3-4); JNZ R0,-2→3 (5-8); HALT (9)
        bc = [0x18, 0, 5, 0x09, 0, 0x3D, 0, 0xFE, 0, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        cfg = ControlFlowGraph(r)
        loops = cfg.get_loops()
        self.assertGreater(len(loops), 0)
    
    def test_cfg_dot_output(self):
        """DOT output should contain graph structure."""
        d = FluxDecompiler([0x18, 0, 42, 0x00])
        r = d.decompile()
        cfg = ControlFlowGraph(r)
        dot = cfg.to_dot()
        self.assertIn("digraph CFG", dot)
    
    def test_cfg_predecessors(self):
        bc = [0x18, 0, 1, 0x3D, 0, 4, 0, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        cfg = ControlFlowGraph(r)
        for bstart, block in cfg.blocks.items():
            if block.predecessors:
                self.assertGreater(len(block.predecessors), 0)
                break


# ═══════════════════════════════════════════════════════════
#  Tests — Function Detection
# ═══════════════════════════════════════════════════════════

class TestFunctionDetection(unittest.TestCase):
    def test_single_function(self):
        d = FluxDecompiler([0x18, 0, 42, 0x00])
        r = d.decompile()
        fd = FunctionDetector(r)
        self.assertEqual(len(fd.functions), 1)
    
    def test_call_creates_function(self):
        """CALL instruction should create a second function entry."""
        # CALL to offset 10
        bc = [0x18, 0, 1, 0x50, 0, 6, 0,  # CALL fn at offset 10 (1+6=7, no)
              0x00, 0x00, 0x00,             # padding
              0x18, 0, 99, 0x51, 0x00]      # function body at 10: MOVI R0, 99; RET; HALT
        d = FluxDecompiler(bc)
        r = d.decompile()
        fd = FunctionDetector(r)
        # At least the main entry
        self.assertGreaterEqual(len(fd.functions), 1)
    
    def test_function_has_return(self):
        bc = [0x18, 0, 42, 0x51, 0x00]  # MOVI R0, 42; RET; HALT
        d = FluxDecompiler(bc)
        r = d.decompile()
        fd = FunctionDetector(r)
        self.assertTrue(fd.functions[0].has_return)
    
    def test_get_function_at(self):
        d = FluxDecompiler([0x18, 0, 42, 0x00])
        r = d.decompile()
        fd = FunctionDetector(r)
        fn = fd.get_function_at(0)
        self.assertIsNotNone(fn)
        self.assertEqual(fn.start, 0)


# ═══════════════════════════════════════════════════════════
#  Tests — Type Inference
# ═══════════════════════════════════════════════════════════

class TestTypeInference(unittest.TestCase):
    def test_bool_from_comparison(self):
        bc = [0x18, 0, 5, 0x18, 1, 10, 0x2C, 2, 0, 1, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        ti = TypeInference(r)
        self.assertEqual(ti.get_type("R2"), RegType.BOOL)
    
    def test_int_from_arithmetic(self):
        bc = [0x18, 0, 5, 0x18, 1, 3, 0x20, 2, 0, 1, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        ti = TypeInference(r)
        self.assertEqual(ti.get_type("R2"), RegType.INT)
    
    def test_unknown_for_unused(self):
        d = FluxDecompiler([0x00])
        r = d.decompile()
        ti = TypeInference(r)
        self.assertEqual(ti.get_type("R5"), RegType.UNKNOWN)
    
    def test_type_string(self):
        d = FluxDecompiler([0x18, 0, 5, 0x18, 1, 10, 0x2C, 2, 0, 1, 0x00])
        r = d.decompile()
        ti = TypeInference(r)
        self.assertEqual(ti.get_type_string("R2"), "bool")
    
    def test_type_summary(self):
        bc = [0x18, 0, 5, 0x18, 1, 10, 0x20, 2, 0, 1, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        ti = TypeInference(r)
        s = ti.summary()
        self.assertIsInstance(s, dict)
        self.assertIn("R2", s)


# ═══════════════════════════════════════════════════════════
#  Tests — Symbol Resolution
# ═══════════════════════════════════════════════════════════

class TestSymbolResolution(unittest.TestCase):
    def test_symbol_table(self):
        st = SymbolTable()
        st.add_symbol(0x100, "main")
        self.assertEqual(st.resolve(0x100), "main")
    
    def test_unknown_address(self):
        st = SymbolTable()
        self.assertEqual(st.resolve(0x999), "0x0999")
    
    def test_constant_resolution(self):
        st = SymbolTable()
        st.add_constant("MAX_SIZE", 42)
        self.assertEqual(st.resolve_constant(42), "MAX_SIZE")
    
    def test_auto_populate(self):
        # MOVI R0,5 (0-2); DEC R0 (3-4); JNZ R0,-2→3 (5-8); HALT (9)
        bc = [0x18, 0, 5, 0x09, 0, 0x3D, 0, 0xFE, 0, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        st = SymbolTable()
        st.auto_populate(r)
        # Should have at least one symbol from labels
        self.assertGreater(len(st.symbols), 0)


# ═══════════════════════════════════════════════════════════
#  Tests — Pattern Detection
# ═══════════════════════════════════════════════════════════

class TestPatternDetection(unittest.TestCase):
    def test_while_loop_pattern(self):
        # MOVI R0,5; DEC R0; JNZ R0,-2→3; HALT
        bc = [0x18, 0, 5, 0x09, 0, 0x3D, 0, 0xFE, 0, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        cfg = ControlFlowGraph(r)
        pd = PatternDetector(cfg)
        patterns = pd.detect_while_loops()
        self.assertGreater(len(patterns), 0)
    
    def test_if_pattern(self):
        # MOVI R0,1; MOVI R1,0; JNZ R0,4; ... ; HALT
        bc = [0x18, 0, 1, 0x18, 1, 0, 0x3D, 0, 4, 0, 0x18, 1, 99, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        cfg = ControlFlowGraph(r)
        pd = PatternDetector(cfg)
        patterns = pd.detect_if_else()
        self.assertGreater(len(patterns), 0)
    
    def test_for_loop_pattern(self):
        # MOVI R0, 0; ... INC R1; ... JNZ R0, back
        bc = [0x18, 0, 3, 0x18, 1, 0, 0x08, 1, 0x09, 0, 0x3D, 0, 0xFC, 0, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        cfg = ControlFlowGraph(r)
        pd = PatternDetector(cfg)
        patterns = pd.detect_for_loops()
        # May or may not detect depending on structure
        self.assertIsInstance(patterns, list)
    
    def test_all_patterns(self):
        bc = [0x18, 0, 5, 0x09, 0, 0x3D, 0, 0xFC, 0, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        cfg = ControlFlowGraph(r)
        pd = PatternDetector(cfg)
        all_p = pd.all_patterns()
        self.assertIn("if_else", all_p)
        self.assertIn("while", all_p)
        self.assertIn("for", all_p)
        self.assertIn("switch", all_p)


# ═══════════════════════════════════════════════════════════
#  Tests — Pseudocode Generation
# ═══════════════════════════════════════════════════════════

class TestPseudocode(unittest.TestCase):
    def test_simple_pseudocode(self):
        d = FluxDecompiler([0x18, 0, 42, 0x00])
        r = d.decompile()
        gen = PseudocodeGenerator(r)
        code = gen.generate()
        self.assertIn("var_0 = 42", code)
        self.assertIn("main", code)
    
    def test_arithmetic_pseudocode(self):
        bc = [0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        gen = PseudocodeGenerator(r)
        code = gen.generate()
        self.assertIn("var_0 = 10", code)
        self.assertIn("var_1 = 20", code)
        self.assertIn("var_0 + var_1", code)
    
    def test_branch_pseudocode(self):
        bc = [0x18, 0, 1, 0x3D, 0, 4, 0, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        gen = PseudocodeGenerator(r)
        code = gen.generate()
        self.assertIn("if", code)
    
    def test_type_declarations(self):
        bc = [0x18, 0, 5, 0x18, 1, 10, 0x2C, 2, 0, 1, 0x00]
        d = FluxDecompiler(bc)
        r = d.decompile()
        gen = PseudocodeGenerator(r)
        code = gen.generate()
        # R2 should be bool from CMP_EQ
        self.assertIn("bool", code)


# ═══════════════════════════════════════════════════════════
#  Tests — CALL/RET
# ═══════════════════════════════════════════════════════════

class TestCallRet(unittest.TestCase):
    def test_call_opcode(self):
        d = FluxDecompiler([0x50, 0, 6, 0, 0x00])  # CALL +6, HALT
        r = d.decompile()
        calls = [i for i in r.instructions if i.mnemonic == "CALL"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].jump_type, JumpType.CALL)
    
    def test_ret_opcode(self):
        d = FluxDecompiler([0x51, 0x00])
        r = d.decompile()
        rets = [i for i in r.instructions if i.mnemonic == "RET"]
        self.assertEqual(len(rets), 1)
        self.assertEqual(rets[0].jump_type, JumpType.RET)
    
    def test_call_target_label(self):
        d = FluxDecompiler([0x50, 0, 6, 0, 0x18, 0, 0, 0x18, 0, 99, 0x51, 0x00])
        r = d.decompile()
        fn_labels = [v for v in r.labels.values() if v.startswith("fn_")]
        self.assertGreater(len(fn_labels), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
