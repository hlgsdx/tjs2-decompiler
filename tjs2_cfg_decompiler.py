"""基于 CFG 的 TJS2 反编译入口。

这个模块把主反编译器里的“结构化恢复”流程串起来：
1. 先分析寄存器与控制流。
2. 再建立 CFG、支配关系和循环信息。
3. 最后把 Region 树重新生成为高级语法树语句。
"""

import sys
from typing import List, Optional

from tjs2_decompiler import (
    Decompiler, BytecodeLoader, CodeObject, Instruction, Stmt,
    ReturnStmt, decode_instructions
)
from tjs2_cfg import (
    build_cfg, compute_dominators, compute_postdominators
)
from tjs2_structuring import (
    detect_loops, build_region_tree, generate_code
)

class CFGDecompiler(Decompiler):
    """使用 CFG + Region 结构化恢复的反编译器实现。"""

    def __init__(self, loader: BytecodeLoader):
        """保存字节码加载器，其他初始化逻辑沿用基类。"""
        super().__init__(loader)

    def _decompile_instructions(self, instructions: List[Instruction],
                                 obj: CodeObject) -> List[Stmt]:
        """把某个对象的指令流反编译成语句列表。

        这里是项目里“从字节码到结构化源码”的核心总调度：
        - `_detect_with_blocks` 识别 `with` 相关区域；
        - `build_cfg` 建图；
        - `compute_dominators` / `compute_postdominators` 为 if/loop/switch
          恢复提供理论基础；
        - `detect_loops` / `build_region_tree` 负责识别结构；
        - `generate_code` 把 Region 转回 AST 语句。
        """
        if not instructions:
            return []

        if not hasattr(self, '_pending_spie'):
            self._reset_state()

        # `with` 会改变属性访问的解释方式，因此必须在正式结构化前先标出范围。
        self._detect_with_blocks(instructions)

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, 10000))
        try:
            # 先让基类完成线性层面的控制流和寄存器分析。
            self._analyze_control_flow(instructions)

            cfg = build_cfg(instructions)

            # 这个步骤用于识别寄存器在不同分支里的拆分与合并，避免把
            # 一个底层临时寄存器错误地映射成同一个高级变量。
            self._analyze_register_splits(instructions, cfg, obj.func_decl_arg_count)

            compute_dominators(cfg)
            compute_postdominators(cfg)

            loops = detect_loops(cfg, instructions)

            region_tree = build_region_tree(cfg, instructions, loops)

            stmts = generate_code(
                region_tree, cfg, instructions, self, obj,
                is_top_level=True
            )

            # TJS2 字节码常在函数末尾保留一个“空返回”，即 `return;`。
            # 为了让输出更接近手写源码，这里把尾部无返回值的 ReturnStmt 去掉。
            while stmts and isinstance(stmts[-1], ReturnStmt) and stmts[-1].value is None:
                stmts.pop()

            return stmts
        finally:
            sys.setrecursionlimit(old_limit)
