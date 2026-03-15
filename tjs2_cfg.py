"""TJS2 控制流图（CFG）构建与支配关系分析。

这个模块负责把线性的字节码指令流切分为基本块，再根据跳转关系建立
控制流图。后续的循环检测、if/switch/try 结构恢复都依赖这里提供的
CFG、支配树和后支配树信息。

之所以要额外引入“虚拟入口/出口”节点，是因为真实字节码可能有多个
`ret` / `throw`，统一接到一个虚拟出口后，后支配分析会简单很多。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from tjs2_decompiler import VM, Instruction

@dataclass
class BasicBlock:
    """基本块。

    `start_idx`/`end_idx` 使用的是 `instructions` 列表中的下标区间，
    遵循 Python 常见的左闭右开语义 `[start_idx, end_idx)`。

    `cond_true` / `cond_false` 仅对条件跳转块有意义，用于在结构化恢复时
    区分“条件成立”和“条件不成立”分别会走向哪里。
    """

    id: int
    start_idx: int
    end_idx: int
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    terminator: Optional[str] = None
    cond_true: Optional[int] = None
    cond_false: Optional[int] = None

    idom: Optional[int] = None
    dom_children: List[int] = field(default_factory=list)

    ipdom: Optional[int] = None
    pdom_children: List[int] = field(default_factory=list)

VIRTUAL_ENTRY_ID = -1
VIRTUAL_EXIT_ID = -2

@dataclass
class CFG:
    """完整的控制流图容器。

    - `blocks`: block_id -> BasicBlock
    - `addr_to_block`: 字节码地址 -> 所属基本块
    - `idx_to_block`: 指令下标 -> 所属基本块
    """

    blocks: Dict[int, BasicBlock] = field(default_factory=dict)
    entry_id: int = VIRTUAL_ENTRY_ID
    exit_id: int = VIRTUAL_EXIT_ID
    addr_to_block: Dict[int, int] = field(default_factory=dict)
    idx_to_block: Dict[int, int] = field(default_factory=dict)

    def get_block(self, block_id: int) -> Optional[BasicBlock]:
        """按 block_id 获取基本块，不存在则返回 `None`。"""
        return self.blocks.get(block_id)

    def real_blocks(self) -> List[BasicBlock]:
        """返回真实基本块，排除虚拟入口/出口，并按出现顺序排序。"""
        return sorted(
            [b for b in self.blocks.values() if b.id >= 0],
            key=lambda b: b.start_idx
        )

    def block_instructions(self, block: BasicBlock, instructions: List[Instruction]) -> List[Instruction]:
        """切出某个基本块对应的指令序列。"""
        return instructions[block.start_idx:block.end_idx]

def build_cfg(instructions: List[Instruction]) -> CFG:
    """从线性指令流中构建控制流图。

    整体流程分两步：
    1. 先找 leader（基本块起点）。
    2. 再根据每个块末尾 terminator 的类型连边。

    这里把 `JF/JNF/JMP/ENTRY/RET/THROW/SETF/SETNF` 都当成可能影响分块
    的指令。尤其 `SETF/SETNF` 虽然不是传统跳转，但本项目会用它们识别
    短路逻辑表达式，因此也需要在 CFG 层面保留边界。
    """
    if not instructions:
        cfg = CFG()
        _add_virtual_nodes(cfg)
        return cfg

    addr_to_idx = {ins.addr: i for i, ins in enumerate(instructions)}
    n = len(instructions)

    leaders = set()
    leaders.add(0)

    for i, instr in enumerate(instructions):
        # 条件跳转/无条件跳转：目标地址与顺序下一条都可能是新的 leader。
        if instr.op in (VM.JF, VM.JNF, VM.JMP):
            target_addr = instr.addr + instr.operands[0]
            target_idx = addr_to_idx.get(target_addr)
            if target_idx is not None:
                leaders.add(target_idx)
            if i + 1 < n:
                leaders.add(i + 1)

        elif instr.op in (VM.RET, VM.THROW):
            # `ret` / `throw` 会终结当前控制流；如果后面还有字节码，
            # 它只能作为新的块起点单独存在。
            if i + 1 < n:
                leaders.add(i + 1)

        elif instr.op == VM.ENTRY:
            # `ENTRY` 是 try 区域入口，操作数里保存 catch 偏移。
            catch_addr = instr.addr + instr.operands[0]
            catch_idx = addr_to_idx.get(catch_addr)
            if catch_idx is not None:
                leaders.add(catch_idx)
            if i + 1 < n:
                leaders.add(i + 1)

        elif instr.op in (VM.SETF, VM.SETNF):
            # 这两个指令常出现在 `a && b` / `a || b` 的短路展开中。
            # 例如：
            #   r1 = a
            #   SETF r1, ...
            # 后续语句会以“标志位已决定结果”为界重新开始。
            if i + 1 < n:
                leaders.add(i + 1)

    sorted_leaders = sorted(leaders)
    cfg = CFG()

    for li, leader_idx in enumerate(sorted_leaders):
        if li + 1 < len(sorted_leaders):
            end_idx = sorted_leaders[li + 1]
        else:
            end_idx = n

        block_id = leader_idx
        block = BasicBlock(id=block_id, start_idx=leader_idx, end_idx=end_idx)

        if end_idx > leader_idx:
            last_instr = instructions[end_idx - 1]
            # terminator 只描述“块尾控制流类型”，真正的边稍后统一补。
            if last_instr.op == VM.JMP:
                block.terminator = 'jmp'
            elif last_instr.op == VM.JF:
                block.terminator = 'jf'
            elif last_instr.op == VM.JNF:
                block.terminator = 'jnf'
            elif last_instr.op == VM.RET:
                block.terminator = 'ret'
            elif last_instr.op == VM.THROW:
                block.terminator = 'throw'
            elif last_instr.op == VM.ENTRY:
                block.terminator = 'entry'
            else:
                block.terminator = 'fall'

        cfg.blocks[block_id] = block

        for idx in range(leader_idx, end_idx):
            cfg.idx_to_block[idx] = block_id
            cfg.addr_to_block[instructions[idx].addr] = block_id

    for block in list(cfg.blocks.values()):
        if block.end_idx <= block.start_idx:
            continue

        last_instr = instructions[block.end_idx - 1]

        if block.terminator == 'jmp':
            # 无条件跳转只有一条显式边。
            target_addr = last_instr.addr + last_instr.operands[0]
            target_idx = addr_to_idx.get(target_addr)
            if target_idx is not None and target_idx in cfg.blocks:
                _add_edge(cfg, block.id, target_idx)

        elif block.terminator in ('jf', 'jnf'):
            # 条件跳转有两条边：
            # 1. 顺序落空（fall-through）
            # 2. 显式跳转目标
            #
            # 对 TJS2 来说：
            # - `JF`  ：条件为假时跳
            # - `JNF` ：条件为“非 false”时跳（项目里把它抽象成 true/false 分支）
            if block.end_idx < n and block.end_idx in cfg.blocks:
                fall_through_id = block.end_idx
                _add_edge(cfg, block.id, fall_through_id)
                if block.terminator == 'jnf':
                    block.cond_true = fall_through_id
                else:
                    block.cond_false = fall_through_id

            target_addr = last_instr.addr + last_instr.operands[0]
            target_idx = addr_to_idx.get(target_addr)
            if target_idx is not None and target_idx in cfg.blocks:
                _add_edge(cfg, block.id, target_idx)
                if block.terminator == 'jnf':
                    block.cond_false = target_idx
                else:
                    block.cond_true = target_idx

        elif block.terminator == 'entry':
            # try 入口既会顺序进入 try 主体，也可能在异常时进入 catch。
            if block.end_idx < n and block.end_idx in cfg.blocks:
                _add_edge(cfg, block.id, block.end_idx)

            catch_addr = last_instr.addr + last_instr.operands[0]
            catch_idx = addr_to_idx.get(catch_addr)
            if catch_idx is not None and catch_idx in cfg.blocks:
                _add_edge(cfg, block.id, catch_idx)

        elif block.terminator in ('ret', 'throw'):
            pass

        elif block.terminator == 'fall':
            # 普通块默认顺序流向下一基本块。
            if block.end_idx < n and block.end_idx in cfg.blocks:
                _add_edge(cfg, block.id, block.end_idx)

    _add_virtual_nodes(cfg)

    return cfg

def _add_edge(cfg: CFG, from_id: int, to_id: int):
    """在 CFG 中添加一条边，并同步维护前驱/后继表。"""
    from_block = cfg.blocks.get(from_id)
    to_block = cfg.blocks.get(to_id)
    if from_block is None or to_block is None:
        return
    if to_id not in from_block.successors:
        from_block.successors.append(to_id)
    if from_id not in to_block.predecessors:
        to_block.predecessors.append(from_id)

def _add_virtual_nodes(cfg: CFG):
    """补上统一的虚拟入口和虚拟出口。

    这一步的收益主要体现在后支配分析：
    多个 `ret`/`throw` 会统一汇聚到 `VIRTUAL_EXIT_ID`，从而让
    “某块的最近公共退出点”可直接通过 ipdom 求出。
    """
    entry_block = BasicBlock(id=VIRTUAL_ENTRY_ID, start_idx=-1, end_idx=-1)
    cfg.blocks[VIRTUAL_ENTRY_ID] = entry_block
    cfg.entry_id = VIRTUAL_ENTRY_ID

    if 0 in cfg.blocks:
        _add_edge(cfg, VIRTUAL_ENTRY_ID, 0)

    exit_block = BasicBlock(id=VIRTUAL_EXIT_ID, start_idx=-1, end_idx=-1)
    cfg.blocks[VIRTUAL_EXIT_ID] = exit_block
    cfg.exit_id = VIRTUAL_EXIT_ID

    for block in cfg.blocks.values():
        if block.terminator in ('ret', 'throw'):
            _add_edge(cfg, block.id, VIRTUAL_EXIT_ID)

    for block in cfg.blocks.values():
        if block.id >= 0 and not block.successors:
            _add_edge(cfg, block.id, VIRTUAL_EXIT_ID)

def _compute_rpo(cfg: CFG, entry_id: int, get_successors) -> List[int]:
    """计算反向后序（Reverse Post Order）。

    RPO 是经典数据流分析遍历顺序，支配/后支配计算通常都会使用它，因为
    它比随意顺序更快收敛。
    """
    visited = set()
    post_order = []

    def dfs(block_id):
        if block_id in visited:
            return
        visited.add(block_id)
        for succ_id in get_successors(block_id):
            if succ_id in cfg.blocks:
                dfs(succ_id)
        post_order.append(block_id)

    dfs(entry_id)
    return list(reversed(post_order))

def _intersect(idom: Dict[int, int], rpo_number: Dict[int, int], b1: int, b2: int) -> int:
    """在支配树上求两个结点的交汇点。

    这是 Lengauer-Tarjan 风格算法之外，常见的 Cooper 等人简化版本里
    使用的核心步骤：不断沿着 idom 链向上“抬高”较深的那个结点，直到
    两个指针相遇。
    """
    finger1 = b1
    finger2 = b2
    while finger1 != finger2:
        while rpo_number.get(finger1, float('inf')) > rpo_number.get(finger2, float('inf')):
            finger1 = idom.get(finger1, finger1)
            if finger1 == idom.get(finger1):
                break
        while rpo_number.get(finger2, float('inf')) > rpo_number.get(finger1, float('inf')):
            finger2 = idom.get(finger2, finger2)
            if finger2 == idom.get(finger2):
                break
    return finger1

def compute_dominators(cfg: CFG):
    """计算每个基本块的直接支配者（idom）。

    含义是：从入口到达块 B 的任意路径都必须经过块 A，则 A 支配 B。
    其中“最近”的那个 A 就记为 B 的 `idom`。
    """
    entry_id = cfg.entry_id

    def get_successors(block_id):
        block = cfg.blocks.get(block_id)
        return block.successors if block else []

    rpo = _compute_rpo(cfg, entry_id, get_successors)
    rpo_number = {block_id: i for i, block_id in enumerate(rpo)}

    idom = {}
    idom[entry_id] = entry_id

    changed = True
    while changed:
        changed = False
        for b in rpo:
            if b == entry_id:
                continue

            block = cfg.blocks.get(b)
            if block is None:
                continue

            new_idom = None
            for p in block.predecessors:
                if p in idom:
                    new_idom = p
                    break

            if new_idom is None:
                continue

            for p in block.predecessors:
                if p == new_idom:
                    continue
                if p in idom:
                    new_idom = _intersect(idom, rpo_number, new_idom, p)

            if idom.get(b) != new_idom:
                idom[b] = new_idom
                changed = True

    for block_id, dom_id in idom.items():
        block = cfg.blocks.get(block_id)
        if block:
            block.idom = dom_id
            block.dom_children = []

    for block_id, dom_id in idom.items():
        if block_id != dom_id:
            parent = cfg.blocks.get(dom_id)
            if parent:
                parent.dom_children.append(block_id)

def compute_postdominators(cfg: CFG):
    """计算直接后支配者（ipdom）。

    后支配和支配方向相反：
    如果从块 B 出发到退出点的任意路径都必须经过块 A，则 A 后支配 B。

    在 if/try/switch 结构恢复时，`ipdom` 往往就是分支重新汇合的 merge 点。
    """
    exit_id = cfg.exit_id

    def get_predecessors(block_id):
        block = cfg.blocks.get(block_id)
        return block.predecessors if block else []

    rpo = _compute_rpo(cfg, exit_id, get_predecessors)
    rpo_number = {block_id: i for i, block_id in enumerate(rpo)}

    ipdom = {}
    ipdom[exit_id] = exit_id

    changed = True
    while changed:
        changed = False
        for b in rpo:
            if b == exit_id:
                continue

            block = cfg.blocks.get(b)
            if block is None:
                continue

            reverse_preds = block.successors

            new_ipdom = None
            for s in reverse_preds:
                if s in ipdom:
                    new_ipdom = s
                    break

            if new_ipdom is None:
                continue

            for s in reverse_preds:
                if s == new_ipdom:
                    continue
                if s in ipdom:
                    new_ipdom = _intersect(ipdom, rpo_number, new_ipdom, s)

            if ipdom.get(b) != new_ipdom:
                ipdom[b] = new_ipdom
                changed = True

    for block_id, pdom_id in ipdom.items():
        block = cfg.blocks.get(block_id)
        if block:
            block.ipdom = pdom_id
            block.pdom_children = []

    for block_id, pdom_id in ipdom.items():
        if block_id != pdom_id:
            parent = cfg.blocks.get(pdom_id)
            if parent:
                parent.pdom_children.append(block_id)

def dominates(cfg: CFG, a: int, b: int) -> bool:
    """判断块 `a` 是否支配块 `b`。"""
    if a == b:
        return True
    current = b
    visited = set()
    while current is not None and current not in visited:
        visited.add(current)
        block = cfg.blocks.get(current)
        if block is None:
            return False
        if block.idom == a:
            return True
        if block.idom == current:
            return False
        current = block.idom
    return False

def postdominates(cfg: CFG, a: int, b: int) -> bool:
    """判断块 `a` 是否后支配块 `b`。"""
    if a == b:
        return True
    current = b
    visited = set()
    while current is not None and current not in visited:
        visited.add(current)
        block = cfg.blocks.get(current)
        if block is None:
            return False
        if block.ipdom == a:
            return True
        if block.ipdom == current:
            return False
        current = block.ipdom
    return False

def get_merge_point(cfg: CFG, block_id: int) -> Optional[int]:
    """返回一个块的最近后支配者，通常可视为其“汇合点”。"""
    block = cfg.blocks.get(block_id)
    if block is None:
        return None
    return block.ipdom

def get_back_edges(cfg: CFG) -> List[Tuple[int, int]]:
    """收集回边 `(tail, header)`。

    若后继 `succ_id` 支配当前块 `block.id`，说明控制流从后面跳回了前面，
    这通常意味着循环。
    """
    back_edges = []
    for block in cfg.blocks.values():
        if block.id < 0:
            continue
        for succ_id in block.successors:
            if dominates(cfg, succ_id, block.id):
                back_edges.append((block.id, succ_id))
    return back_edges

def get_natural_loop(cfg: CFG, back_edge: Tuple[int, int]) -> Set[int]:
    """根据回边求自然循环体。

    做法是从回边尾部开始，逆着前驱向上回溯，直到收敛到循环头。
    这会得到经典编译原理中的 natural loop。
    """
    tail, header = back_edge
    loop_blocks = {header, tail}

    if tail == header:
        return loop_blocks

    worklist = [tail]
    while worklist:
        block_id = worklist.pop()
        block = cfg.blocks.get(block_id)
        if block is None:
            continue
        for pred_id in block.predecessors:
            if pred_id not in loop_blocks and pred_id >= 0:
                loop_blocks.add(pred_id)
                worklist.append(pred_id)

    return loop_blocks
