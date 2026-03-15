# 项目架构总览

## 1. 这个项目在做什么

这个仓库实现了一个 TJS2 字节码反编译器。它把 Kirikiri/Kirikiri2 使用的 `TJS2100` 字节码文件读进来，解析出常量池、代码对象和指令流，再逐步恢复成更接近人类书写风格的 TJS2 源码。

从“输入字节码”到“输出源码”，中间大致经历 5 层：

1. 文件解析层：把 `.tjs` 字节码文件拆成 `DATA` 和 `OBJS`。
2. 指令解码层：把 16 位代码流切成一条条 `Instruction`。
3. 语义恢复层：把寄存器操作恢复成表达式、赋值、调用、返回等 AST。
4. 结构化控制流层：把跳转图恢复成 `if/else`、循环、`switch`、`try/catch`。
5. 文本整理层：把语义正确但略机械的源码格式化得更接近人写的 TJS。

## 2. 仓库的核心文件

这个仓库高度集中，核心逻辑几乎全部在 5 个文件里：

| 文件 | 角色 |
| --- | --- |
| `tjs2_decompiler.py` | 主模块。定义 VM opcode、AST、字节码加载器、线性反编译器、CLI。 |
| `tjs2_cfg.py` | 把线性指令流切成基本块并建立 CFG，计算支配/后支配关系。 |
| `tjs2_cfg_decompiler.py` | 用 CFG 版流程接管主反编译器，是当前真正走的高质量反编译入口。 |
| `tjs2_structuring.py` | 从 CFG 恢复高层控制流结构，是“编译原理味道”最重的模块。 |
| `tjs2_formatting.py` | 对最终源码做保守格式化和语法糖恢复。 |

## 3. 反编译主流程

当前项目默认走的是 CFG 结构化版本，而不是仅靠线性跳转的老路线。实际主路径如下：

```text
decompile_file()
  -> BytecodeLoader.load()
  -> CFGDecompiler(loader).decompile()
     -> Decompiler.decompile()
        -> _decompile_object() / _decompile_object_definition()
           -> decode_instructions()
           -> CFGDecompiler._decompile_instructions()
              -> _detect_with_blocks()
              -> _analyze_control_flow()
              -> build_cfg()
              -> _analyze_register_splits()
              -> compute_dominators()
              -> compute_postdominators()
              -> detect_loops()
              -> build_region_tree()
              -> generate_code()
     -> format_source()
```

## 4. 各层数据结构

### 4.1 文件/字节码层

`BytecodeLoader` 会把原始字节流解析为：

- 若干常量池数组：`byte_array`、`short_array`、`long_array`、`double_array`、`string_array`、`octet_array`
- `objects`：多个 `CodeObject`
- `toplevel`：顶层对象索引

### 4.2 代码对象层

`CodeObject` 是本项目最关键的“字节码对象”抽象。一个对象可以对应：

- 顶层脚本
- 普通函数
- lambda/表达式函数
- 类
- 属性对象
- getter/setter

一个 `CodeObject` 里最重要的字段：

- `context_type`：对象类型
- `code`：原始 16 位代码流
- `data`：该对象使用的数据表
- `properties`：属性列表
- `source_positions`：字节码地址到源位置的映射

### 4.3 指令层

`Instruction` 是解码后的单条指令：

- `addr`：指令地址
- `op`：opcode
- `operands`：操作数列表
- `size`：指令长度

注意这里的 `addr` 不是字节偏移，而是 `code` 数组中的“16 位单元地址”。

### 4.4 AST 层

`tjs2_decompiler.py` 自己定义了一套轻量 AST：

- 表达式：`ConstExpr`、`VarExpr`、`PropertyExpr`、`CallExpr`、`AssignExpr`、`UnaryExpr`、`BinaryExpr`、`TernaryExpr` 等
- 语句：`VarDeclStmt`、`ExprStmt`、`IfStmt`、`WhileStmt`、`ForStmt`、`TryStmt`、`SwitchStmt`、`ReturnStmt` 等

这套 AST 是整个项目的“内部语言”。前面的解析、CFG 和结构化恢复，最后都要落到这些类上。

### 4.5 CFG/Region 层

当项目进入结构化控制流恢复阶段后，会额外引入两层表示：

- `BasicBlock` / `CFG`：控制流图
- `Region`：从 CFG 识别出的结构化区域

可以把它理解成：

```text
Instruction -> BasicBlock -> CFG -> Region tree -> AST -> Source
```

## 5. 为什么主模块这么大

`tjs2_decompiler.py` 超过 30 万字符，看起来很“重”，但它其实承载了多个逻辑层：

1. VM 指令定义。
2. 源码 AST 定义。
3. 字节码文件加载器。
4. 线性指令翻译器。
5. 对对象、函数、类、属性的源码生成。
6. CLI 和批量处理入口。

这不是最理想的模块划分，但对学习者反而有个好处：很多关键概念都在同一文件里，跳转成本低。

## 6. 反编译器的几个核心思想

### 6.1 “先恢复语义，再恢复结构”

本项目不是一上来就试图把跳转图直接还原成高级语法，而是分两步：

1. 先把寄存器级操作恢复成表达式和语句语义。
2. 再把控制流结构恢复成 `if/while/switch/try`。

这也是为什么 `Decompiler` 仍然很重要，即使最终输出走的是 `CFGDecompiler`。

### 6.2 “寄存器值不是变量名”

TJS2 字节码里大量使用寄存器，尤其是负寄存器作为局部槽位。项目不会简单把每个寄存器直接翻译成固定变量，而是会做：

- 寄存器到表达式的映射 `regs`
- 局部槽位到变量名的映射 `local_vars`
- register split 分析，避免同一个槽位在不同生命周期里被误当成同一个高层变量

这是可读性提升最关键的地方之一。

### 6.3 “容器字面量是延迟恢复的”

像 `new Array()`、`new Dictionary()`，项目不会立刻输出，而是先进入：

- `pending_arrays`
- `pending_dicts`

等后面的 `SPI` 连续填充完成，再一次性恢复成数组/字典字面量。

### 6.4 “赋值有时要先挂起”

对属性赋值和索引赋值，项目经常不会立即吐出 `obj.x = value;`，而是把它放进 `_pending_spie`，等待判断后续是否会把这个赋值结果继续当表达式使用，例如：

```tjs
foo(obj.x = value)
local = (obj.x = value)
```

这让输出比“每条指令一条语句”自然很多。

### 6.5 “结构化控制流恢复依赖 CFG”

高质量反编译的关键不在单条指令，而在 CFG：

- `build_cfg()` 先切基本块、连边
- `compute_dominators()` / `compute_postdominators()` 提供理论基础
- `detect_loops()` 恢复循环边界
- `build_region_tree()` 识别 `if/switch/try`
- `generate_code()` 最终落成 AST 语句

## 7. TJS2 相关的几个项目内约定

为了看懂本项目，需要先接受几个“内部约定”：

### 7.1 特殊寄存器

- `0`：常被当成 `void` 或“无返回值/无结果寄存器”
- `-1`：`this`
- `-2`：`this` 代理，进入 `with` 场景时可能解释成 `WithThisExpr`
- `< -2`：局部变量槽位，最终会命名成 `arg0`、`local0` 之类

### 7.2 条件不是直接写回寄存器

比较类指令和 `TT/TF/NF` 主要维护的是 `flag` 与 `flag_negated` 状态，后面的 `JF/JNF/SETF/SETNF` 会消耗这个条件状态。

### 7.3 opcode 命名有规律

- 基础运算：`ADD`、`SUB`、`LOR`
- 点属性版本：`...PD`
- 索引属性版本：`...PI`
- 属性引用/属性句柄版本：`...P`

例如 `ADD`、`ADDPD`、`ADDPI`、`ADDP` 都围绕“加法/复合赋值”展开，只是作用目标不同。

## 8. 输出质量来自哪些地方

项目最终输出可读，不是因为某一个函数很强，而是多个细节叠加：

- 识别匿名函数、类、属性对象的归属关系
- 把子函数/子类尽量放回父对象附近
- 识别 `with` 范围
- 识别 `swap`
- 识别短路表达式 `&&` / `||`
- 识别三元表达式
- 识别 `for` 初始化和更新语句
- 恢复默认参数
- 恢复 `extends` / `super`
- 整理 `else if`、空行和长行

## 9. 你应该带着什么问题去读源码

如果你是第一次学这个项目，建议优先关注下面几个问题：

1. TJS2 字节码文件被读进来后，第一层抽象是什么？
2. 一条 opcode 是如何被翻译成表达式/语句的？
3. 为什么项目要维护 `flag`、`pending_arrays`、`pending_dicts`、`_pending_spie` 这些状态？
4. 为什么仅靠线性扫描很难恢复高质量 `if/while/switch/try`？
5. CFG、支配关系、Region tree 分别解决了什么问题？

当你能回答这 5 个问题时，基本就已经真正看懂这个仓库的主干了。
