# 模块职责与源码阅读顺序

## 1. 总体建议

这个项目不适合“从头到尾硬读一遍”。最好的方式是按调用链和抽象层来读。

推荐阅读顺序：

1. `README.md`
2. `tjs2_decompiler.py` 顶部的数据结构定义
3. `BytecodeLoader`
4. `get_instruction_size()` 与 `decode_instructions()`
5. `Decompiler` 的状态字段和对象级入口
6. `tjs2_cfg.py`
7. `tjs2_cfg_decompiler.py`
8. `tjs2_structuring.py`
9. `tjs2_formatting.py`
10. 回到 `Decompiler._translate_instruction()` 做“按 opcode 精读”

---

## 2. 第一站：`README.md`

目标：先知道这个工具对外提供什么功能。

读的时候重点关注：

- 命令行参数
- 单文件/目录模式
- `-d` 反汇编与 `-i` 信息模式
- 输出编码选择

为什么先读它：

- 你会立刻知道“这个仓库的产品形态是什么”
- 后面看到 `decompile_file()` / `decompile_directory()` 时不会陌生

---

## 3. 第二站：`tjs2_decompiler.py` 顶部定义

这是最重要的“术语表”。

### 3.1 先看 `VM`

这里定义了全部 opcode 枚举。不要一开始就试图背住 128 个名字，只要先理解命名规律：

- 算术/逻辑基础版
- `PD` 点属性版
- `PI` 索引属性版
- `P` 属性引用版

### 3.2 再看 `DataType` / `ContextType`

它们解释了：

- 常量池里数据项的类型
- 一个 `CodeObject` 究竟是顶层、函数、类还是属性

### 3.3 再看 AST 类

先不需要逐个细读所有方法，只要知道项目内部最终会把字节码恢复成这些节点：

- 表达式类
- 语句类

建议你至少看懂下面这些类的 `to_source()`：

- `ConstExpr`
- `VarExpr`
- `PropertyExpr`
- `CallExpr`
- `AssignExpr`
- `IfStmt`
- `WhileStmt`
- `ForStmt`
- `TryStmt`
- `SwitchStmt`

原因很简单：后面很多反编译逻辑其实只是“构造这些节点”。

---

## 4. 第三站：`BytecodeLoader`

这一段是“文件格式入口”。

建议阅读顺序：

1. `load()`
2. `_read_data_area()`
3. `_read_objects()`
4. `_resolve_data()`

### 4.1 你要看懂什么

- 文件头是 `TJS2100\0`
- 文件分成 `DATA` 和 `OBJS`
- `DATA` 里是按类型分开的常量池
- `OBJS` 里是一组代码对象

### 4.2 初学者最值得注意的点

- 字符串按 UTF-16 code unit 读取
- 某些数组段会做 4 字节对齐
- `CodeObject.data` 不是原始索引，而是已经解析后的值
- `INTER_OBJECT` 会形成“对象引用”，这对恢复函数/类很关键

### 4.3 为什么这一段重要

因为如果你不清楚 `data[idx]` 里放的是什么，后面看 `CONST`、`GPD`、`CALLD` 都会一头雾水。

---

## 5. 第四站：指令解码

对应函数：

- `get_instruction_size()`
- `decode_instructions()`

### 5.1 为什么它重要

TJS2 指令不是定长的。只有先正确切分指令，后面的反汇编、CFG、控制流恢复才有意义。

### 5.2 这里要看懂的重点

- 哪些 opcode 长度固定
- 哪些 opcode 属于“基础版/PD/PI/P”家族
- `CALL/CALLD/CALLI/NEW` 的长度为什么要看 `argc`
- `argc == -1`、`argc == -2` 分别意味着什么

### 5.3 学习建议

这一步可以配合 `disassemble_object()` 一起看。你会更容易把“code 数组”理解成真正的人类可读指令流。

---

## 6. 第五站：`Decompiler` 的全局状态

先读：

- `__init__()`
- `_reset_state()`

再读：

- `decompile()`
- `_decompile_object_definition()`
- `_decompile_function()`
- `_decompile_class()`
- `_decompile_property()`

### 6.1 为什么要先读状态

因为 `Decompiler` 不是纯函数式翻译器，而是“带状态的语义恢复器”。

里面最重要的状态包括：

| 状态 | 作用 |
| --- | --- |
| `regs` | 寄存器当前对应的表达式 |
| `local_vars` | 局部槽位到变量名的映射 |
| `flag` / `flag_negated` | 当前条件表达式状态 |
| `pending_arrays` | 延迟恢复数组字面量 |
| `pending_dicts` | 延迟恢复字典字面量 |
| `_pending_spie` | 延迟恢复属性/索引赋值表达式 |
| `loop_context_stack` | 帮助识别 `break/continue` |

### 6.2 这一步看什么

你需要先理解：

- 为什么对象会被分发到“函数/类/属性”不同入口
- 为什么子函数、子类需要重挂到父对象附近输出
- 为什么要先 `_detect_with_blocks()`

### 6.3 初学者常见误区

不要一上来钻进 `_translate_instruction()` 的几千行分支里。先理解“对象级输出组织”比“单条 opcode”更重要。

---

## 7. 第六站：线性控制流与对象内反编译

对应函数：

- `_decompile_object()`
- `_decompile_instructions()`
- `_analyze_control_flow()`

### 7.1 这一层在做什么

这层仍然是“线性视角”，主要做：

- 找 jump target
- 找 back edge
- 记录 loop header

它相当于 CFG 版本之前的一层基础分析。

### 7.2 为什么仍然值得读

即使最终默认走 `CFGDecompiler`，这些基础状态和辅助分析仍然被大量复用。

---

## 8. 第七站：`tjs2_cfg.py`

这是控制流图基础设施模块。

建议顺序：

1. `BasicBlock`
2. `CFG`
3. `build_cfg()`
4. `_add_virtual_nodes()`
5. `compute_dominators()`
6. `compute_postdominators()`
7. `get_merge_point()`
8. `get_back_edges()` / `get_natural_loop()`

### 8.1 你要重点看懂什么

- basic block 如何切分
- `JF/JNF/JMP/ENTRY/RET/THROW` 如何连边
- 为什么要加虚拟入口和虚拟出口
- 支配/后支配在结构化恢复里有什么用

### 8.2 这部分最重要的收获

一旦你看懂 `build_cfg()`，就会明白：

- “if 结构”不是某一条指令，而是几块基本块之间的关系
- “loop 结构”不是简单的回跳，而是回边、出口块和 body 的共同结果

---

## 9. 第八站：`tjs2_cfg_decompiler.py`

这个文件不大，但很关键。它相当于把前面的零件串起来。

建议重点看唯一的核心方法：

- `CFGDecompiler._decompile_instructions()`

### 9.1 它的价值

它把结构化反编译主流程压缩成了一条很清晰的管线：

1. 识别 `with`
2. 基础控制流分析
3. 建 CFG
4. 分析 register split
5. 算支配/后支配
6. 检测循环
7. 建 Region tree
8. 生成结构化代码

### 9.2 这一步应该得到的认识

当你看完这个文件，应该能回答：

“当前仓库真正依赖的高质量反编译主路线是什么？”

如果回答不出，就不要急着往 `tjs2_structuring.py` 深挖。

---

## 10. 第九站：`tjs2_structuring.py`

这是最难的一部分，也是最值得花时间的一部分。

建议阅读顺序：

1. `RegionType`
2. `LoopInfo`
3. `SwitchCase`
4. `Region`
5. `detect_loops()`
6. `detect_switch_at()`
7. `detect_try_at()`
8. `build_region_tree()`
9. `generate_code()`
10. 各类 `_generate_*` 函数

### 10.1 先别急着看细节

先抓住它的核心职责：

- 把 CFG 中的块组合识别成高层结构
- 再把这些结构统一生成 AST

### 10.2 最值得关注的几个识别点

- `if / else`
- `while / do while / infinite`
- `switch`
- `try / catch`
- 短路逻辑 `&&` / `||`
- 条件链和三元表达式

### 10.3 为什么这里有这么多补丁逻辑

因为真实字节码 CFG 往往不“教科书化”：

- 编译器会留下死跳转尾巴
- `try/catch` 会打乱循环自然边界
- `switch` 会和回跳、落空执行混在一起
- 短路表达式会跨块展开

所以这不是“优雅的纯理论实现”，而是“理论 + 工程补丁”的结合体。

---

## 11. 第十站：`tjs2_formatting.py`

这部分不负责恢复语义，而负责提升可读性。

建议先读：

- `format_source()`

再根据兴趣读下面这些子阶段：

- `_fix_anon_func_indent()`
- `_format_long_line()`
- `_restore_default_params()`
- `_restore_extends()`
- `_restore_super_calls()`
- `_merge_else_if()`
- `_rename_catch_var()`
- `_wrap_toplevel_local_vars()`

### 11.1 为什么它重要

反编译器的用户最终看到的是文本，不是 AST。

如果没有这一步，输出虽然语义可能正确，但会显得：

- 太机械
- 太扁平
- 缺少语法糖
- 不利于人工阅读

### 11.2 你要有一个边界意识

这个模块原则上不应该改变程序语义，只做保守的文本重写。读这部分时，要始终区分：

- 语义恢复
- 语法糖恢复
- 纯格式整理

---

## 12. 最后再回头读 `_translate_instruction()`

这是整仓库最密集、也最容易把人劝退的函数。

正确读法不是从头硬啃，而是按主题读。

推荐分 8 轮：

1. 常量、复制、清空：`CONST`、`CP`、`CL`、`CCL`
2. 条件系统：`TT`、`TF`、`NF`、`CEQ`、`CDEQ`、`CLT`、`CGT`、`SETF`、`SETNF`
3. 一元运算与类型转换：`LNOT`、`BNOT`、`CHS`、`INT`、`REAL`、`STR`、`OCTET`
4. 自增自减和复合运算：`INC/DEC` 及 `...PD/...PI/...P`
5. 属性访问：`GPD/GPI/GPDS/GPIS`、`SPD/SPI/...`
6. 调用与构造：`CALL/CALLD/CALLI/NEW`
7. 类型/上下文/引用：`CHKINS`、`CHGTHIS`、`GETP`、`SETP`、`TYPEOF*`
8. 返回/异常/特殊语义：`SRV`、`RET`、`THROW`、`INV`、`EVAL`、`EEXP`

### 12.1 为什么要最后读它

因为只有当你已经知道：

- 操作数来自哪里
- `regs` 是什么
- `flag` 是什么
- `pending_*` 是什么
- CFG 和结构化恢复在后面还会做什么

你才能真正看懂 `_translate_instruction()`，否则只会觉得它是一大坨 `if/elif`。

---

## 13. 面向初学者的最短源码学习路线

如果你只想先快速入门，而不是一次学完，推荐压缩成下面这条路线：

1. `VM`
2. `CodeObject`
3. `Instruction`
4. `BytecodeLoader.load()`
5. `decode_instructions()`
6. `CFGDecompiler._decompile_instructions()`
7. `build_cfg()`
8. `detect_loops()`
9. `build_region_tree()`
10. `_translate_instruction()` 中你最关心的那组 opcode

---

## 14. 面向进阶阅读者的研究路线

如果你之后想继续改进这个项目，推荐按问题导向读：

### 14.1 想提升变量命名质量

重点看：

- `_get_def_use_regs()`
- `_analyze_register_splits()`
- `_get_local_name()`

### 14.2 想提升控制流恢复质量

重点看：

- `build_cfg()`
- `compute_dominators()`
- `compute_postdominators()`
- `detect_loops()`
- `build_region_tree()`

### 14.3 想提升表达式自然度

重点看：

- `_pending_spie`
- `_try_process_short_circuit()`
- `_try_detect_ternary()`
- `pending_arrays` / `pending_dicts`

### 14.4 想提升最终可读性

重点看：

- `format_source()`
- 默认参数恢复
- 继承/`super` 恢复
- 顶层局部变量包裹逻辑

---

## 15. 读完后你应该能回答的 10 个问题

1. 一个 `CodeObject` 在这个项目里表示什么？
2. 为什么 opcode 需要先计算可变长度？
3. `regs` 和 `local_vars` 的区别是什么？
4. `flag` 是如何参与条件恢复的？
5. 为什么数组/字典字面量要延迟恢复？
6. 为什么属性赋值有时会被挂起到 `_pending_spie`？
7. `JF` 和 `JNF` 在 CFG 里如何解释？
8. 支配关系和后支配关系各自解决什么问题？
9. `Region` 为什么是必要的中间层？
10. `format_source()` 为什么不能替代前面的反编译逻辑？

如果这 10 个问题你都能回答出来，这个项目你就已经入门了。
