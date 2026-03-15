# TJS2 虚拟机操作码手册

## 1. 这份文档怎么读

这不是一份“官方 TJS2 VM 规范翻译”，而是一份“基于本仓库实现方式整理的 opcode 学习手册”。

也就是说：

- 它优先回答“这个项目如何解释这些指令”
- 对源码阅读最重要
- 对反编译结果的理解最直接

如果你正在读 `tjs2_decompiler.py`，建议把这份文档当成旁边的注释本。

---

## 2. 入门前必须知道的 8 个约定

### 2.1 指令地址

这里的指令地址 `addr` 是 `code` 数组中的 16 位单元地址，不是字节偏移。

### 2.2 特殊寄存器

| 寄存器 | 项目中的常见解释 |
| --- | --- |
| `0` | `void`、无结果、空返回、被省略的临时目标 |
| `-1` | `this` |
| `-2` | `this` 代理；在 `with` 作用域里会特殊处理 |
| `< -2` | 局部变量/参数槽位 |
| `> 0` | 普通临时寄存器 |

### 2.3 条件系统不是“布尔寄存器”模型

很多条件类 opcode 并不直接写一个布尔寄存器，而是维护：

- `flag`
- `flag_negated`

后续由 `JF`、`JNF`、`SETF`、`SETNF` 消耗。

### 2.4 跳转偏移是相对地址

`JF/JNF/JMP/ENTRY` 的跳转目标，在本项目里都按：

```text
target_addr = instr.addr + instr.operands[0]
```

来计算。

### 2.5 opcode 命名规则

| 后缀 | 含义 |
| --- | --- |
| 无后缀 | 直接对寄存器/值操作 |
| `PD` | property dot，点属性版本，如 `obj.prop` |
| `PI` | property index，索引属性版本，如 `obj[idx]` |
| `P` | 属性引用/属性句柄版本 |
| `S` | 在属性访问类指令中通常代表“引用风格/地址风格” |
| `E` / `EH` | setter 相关变体，本项目主要把它们并入“属性写入”语义 |

### 2.6 调用参数的特殊编码

调用类 opcode 里：

- `argc >= 0`：后面直接跟 `argc` 个参数寄存器
- `argc == -1`：整体转发，可理解为 `...`
- `argc == -2`：参数按“类型 + 寄存器”二元组编码

### 2.7 `RET` 和 `SRV` 不是一回事

在本项目里：

- `SRV r` 更像真正的 `return expr`
- `RET` 更像函数尾部控制流结束标记

### 2.8 不是所有 opcode 都直接产出源码

很多指令只更新状态，不立刻生成语句，比如：

- `TT/TF/NF`
- `CONST`
- `CP`
- `SETF/SETNF`
- `GLOBAL`

---

## 3. 指令编码长度规律

`get_instruction_size()` 是理解整个 VM 的入口。

### 3.1 长度为 1 的典型指令

- `NOP`
- `NF`
- `RET`
- `EXTRY`
- `REGMEMBER`
- `DEBUGGER`

### 3.2 长度为 2 的典型指令

- `TT`
- `TF`
- `SETF`
- `SETNF`
- `LNOT`
- `BNOT`
- `ASC`
- `CHR`
- `NUM`
- `CHS`
- `INV`
- `CHKINV`
- `TYPEOF`
- `EVAL`
- `EEXP`
- `INT`
- `REAL`
- `STR`
- `OCTET`
- `JF`
- `JNF`
- `JMP`
- `SRV`
- `THROW`
- `GLOBAL`
- `INC`
- `DEC`

### 3.3 长度为 3 的典型指令

- `CONST`
- `CP`
- `CEQ`
- `CDEQ`
- `CLT`
- `CGT`
- `CHKINS`
- `CHGTHIS`
- `ADDCI`
- `CCL`
- `ENTRY`
- `SETP`
- `GETP`
- `INCP`
- `DECP`

### 3.4 二元运算家族长度

对 `LOR/LAND/BOR/BXOR/BAND/SAR/SAL/SR/ADD/SUB/MOD/DIV/IDIV/MUL` 这 14 个基础运算：

- 基础版：长度 3
- `PD` 版：长度 5
- `PI` 版：长度 5
- `P` 版：长度 4

### 3.5 属性访问类长度

长度 4 的典型指令：

- `INCPD` / `DECPD`
- `INCPI` / `DECPI`
- `GPD` / `GPDS`
- `GPI` / `GPIS`
- `SPD` / `SPDE` / `SPDEH` / `SPDS`
- `SPI` / `SPIE` / `SPIS`
- `DELD` / `DELI`
- `TYPEOFD` / `TYPEOFI`

### 3.6 调用类长度

- `CALL` / `NEW`
- `CALLD` / `CALLI`

长度都依赖 `argc`，所以是可变长。

---

## 4. 操作码家族总览

按功能分，当前项目里的 opcode 可以粗分成 10 组：

1. 基础与常量装载
2. 条件与跳转
3. 一元运算与类型转换
4. 自增自减
5. 二元运算与复合赋值
6. 属性访问与属性引用
7. 调用与构造
8. 类型/对象/上下文操作
9. 返回、异常、作用域控制
10. 其他保留或弱语义指令

---

## 5. 基础与常量装载

### `NOP = 0`

- 含义：空操作
- 项目处理：忽略，不生成源码

### `CONST = 1`

- 形式：`CONST dst, data_idx`
- 含义：从对象数据表装载常量
- 项目处理：把 `obj.data[data_idx]` 转为 `ConstExpr` 或对象引用表达式

### `CP = 2`

- 形式：`CP dst, src`
- 含义：复制寄存器/槽位值
- 项目处理：
  - 写入负寄存器时通常变成局部变量声明或赋值
  - 特殊模式下可触发 `with` 进入
  - 也可能把子函数对象恢复成具名函数声明

### `CL = 3`

- 形式：`CL r`
- 含义：清空寄存器/槽位
- 项目处理：恢复成 `void`，对局部槽位可能输出 `var x;`

### `CCL = 4`

- 形式：`CCL r, count`
- 含义：从 `r` 起连续清空多个槽位
- 项目处理：批量写入 `VoidExpr`

---

## 6. 条件、比较与跳转

### `TT = 5`

- 形式：`TT r`
- 含义：将寄存器内容作为“当前条件”
- 项目处理：`flag = get_reg(r)`，不直接输出

### `TF = 6`

- 形式：`TF r`
- 含义：将寄存器内容作为“反向条件”
- 项目处理：设置 `flag` 并标记 `flag_negated = True`

### 比较类

| opcode | 值 | 项目中的高层语义 |
| --- | --- | --- |
| `CEQ` | 7 | `left == right` |
| `CDEQ` | 8 | `left === right` |
| `CLT` | 9 | `left < right` |
| `CGT` | 10 | `left > right` |

这几个指令主要更新 `flag`，由后续跳转或 `SETF/SETNF` 消费。

### `SETF = 11`

- 形式：`SETF dst`
- 含义：把当前条件按正向读成表达式值
- 项目处理：`dst = current_condition`

### `SETNF = 12`

- 形式：`SETNF dst`
- 含义：把当前条件按取反后的方向读成表达式值

### `LNOT = 13`

- 形式：`LNOT r`
- 含义：逻辑非
- 项目处理：`r = !r`

### `NF = 14`

- 形式：无操作数
- 含义：翻转当前条件极性
- 项目处理：切换 `flag_negated`

### 跳转类

| opcode | 值 | 说明 |
| --- | --- | --- |
| `JF` | 15 | 条件跳转 |
| `JNF` | 16 | 条件跳转的另一变体 |
| `JMP` | 17 | 无条件跳转 |

项目里：

- `JF/JNF` 会进入 `if`、短路、`switch`、循环识别
- `JMP` 会参与 `break`、`continue`、循环回边和普通 CFG 连边分析

> 对初学者最实用的理解方式：不要死背 `JF/JNF` 的“官方真假语义”，而要看项目如何把它们映射到 `cond_true/cond_false` 和结构化恢复逻辑里。

---

## 7. 自增自减家族

### 直接寄存器版

| opcode | 值 | 语义 |
| --- | --- | --- |
| `INC` | 18 | `++x` / `x++` |
| `DEC` | 22 | `--x` / `x--` |

项目会结合前一条 `CP`/`GPD`/`GPI`/`GETP` 推断前缀还是后缀形式。

### 点属性版

| opcode | 值 | 语义 |
| --- | --- | --- |
| `INCPD` | 19 | `++obj.prop` / `obj.prop++` |
| `DECPD` | 23 | `--obj.prop` / `obj.prop--` |

### 索引属性版

| opcode | 值 | 语义 |
| --- | --- | --- |
| `INCPI` | 20 | `++obj[idx]` / `obj[idx]++` |
| `DECPI` | 24 | `--obj[idx]` / `obj[idx]--` |

### 属性引用版

| opcode | 值 | 语义 |
| --- | --- | --- |
| `INCP` | 21 | `++(*propRef)` |
| `DECP` | 25 | `--(*propRef)` |

---

## 8. 二元运算与复合赋值家族

这一组是 TJS2 VM 中最整齐的一批 opcode。

### 8.1 基础家族与符号

| 基础 opcode | 值 | 运算符 |
| --- | --- | --- |
| `LOR` | 26 | `||` |
| `LAND` | 30 | `&&` |
| `BOR` | 34 | `|` |
| `BXOR` | 38 | `^` |
| `BAND` | 42 | `&` |
| `SAR` | 46 | `>>` |
| `SAL` | 50 | `<<` |
| `SR` | 54 | `>>>` |
| `ADD` | 58 | `+` |
| `SUB` | 62 | `-` |
| `MOD` | 66 | `%` |
| `DIV` | 70 | `/` |
| `IDIV` | 74 | `\\` |
| `MUL` | 78 | `*` |

### 8.2 家族变体规律

对任意一个基础 opcode `OP`：

- `OP`：寄存器/局部变量版
- `OP + 1`：`OPPD`，点属性复合赋值
- `OP + 2`：`OPPI`，索引属性复合赋值
- `OP + 3`：`OPP`，属性引用复合赋值

### 8.3 项目中的解释方式

#### 基础版 `OP`

- 如果目标是负寄存器局部槽位，通常恢复成：

```tjs
local += rhs
```

- 如果目标是普通临时寄存器，则更像：

```tjs
r = r + rhs
```

#### `OPPD`

恢复成：

```tjs
obj.prop <op>= value
```

#### `OPPI`

恢复成：

```tjs
obj[idx] <op>= value
```

#### `OPP`

恢复成：

```tjs
(*propRef) <op>= value
```

### 8.4 全家族一览

| 基础 | 点属性 | 索引属性 | 属性引用 |
| --- | --- | --- | --- |
| `LOR` | `LORPD` | `LORPI` | `LORP` |
| `LAND` | `LANDPD` | `LANDPI` | `LANDP` |
| `BOR` | `BORPD` | `BORPI` | `BORP` |
| `BXOR` | `BXORPD` | `BXORPI` | `BXORP` |
| `BAND` | `BANDPD` | `BANDPI` | `BANDP` |
| `SAR` | `SARPD` | `SARPI` | `SARP` |
| `SAL` | `SALPD` | `SALPI` | `SALP` |
| `SR` | `SRPD` | `SRPI` | `SRP` |
| `ADD` | `ADDPD` | `ADDPI` | `ADDP` |
| `SUB` | `SUBPD` | `SUBPI` | `SUBP` |
| `MOD` | `MODPD` | `MODPI` | `MODP` |
| `DIV` | `DIVPD` | `DIVPI` | `DIVP` |
| `IDIV` | `IDIVPD` | `IDIVPI` | `IDIVP` |
| `MUL` | `MULPD` | `MULPI` | `MULP` |

---

## 9. 位运算、类型、值变换与特殊一元运算

### `BNOT = 82`

- 语义：按位非
- 恢复：`~expr`

### `TYPEOF = 83`

- 语义：`typeof expr`

### `TYPEOFD = 84`

- 语义：`typeof obj.prop`

### `TYPEOFI = 85`

- 语义：`typeof obj[idx]`

### `EVAL = 86`

- 项目中的恢复：后缀 `!`
- 更像某种“求值展开”语义，作为表达式保留

### `EEXP = 87`

- 和 `EVAL` 类似，但直接作为语句输出

### `CHKINS = 88`

- 项目中恢复为：`instanceof`

### `ASC = 89`

- 恢复：`#expr`

### `CHR = 90`

- 恢复：`$expr`

### `NUM = 91`

- 恢复：一元正号 `+expr`

### `CHS = 92`

- 恢复：一元负号 `-expr`

### `INV = 93`

- 项目中恢复成：`invalidate expr`

### `CHKINV = 94`

- 恢复成：`isvalid expr`

### 类型转换类

| opcode | 值 | 恢复 |
| --- | --- | --- |
| `INT` | 95 | `int(expr)` 风格 |
| `REAL` | 96 | `real(expr)` |
| `STR` | 97 | `string(expr)` |
| `OCTET` | 98 | `octet(expr)` |

---

## 10. 调用与构造

### `CALL = 99`

- 形式：`CALL dst, func, argc, ...`
- 含义：普通函数调用
- 项目恢复：
  - `dst == 0`：直接输出 `func(args);`
  - `dst > 0`：把结果放进寄存器表达式
  - 某些有副作用且多次读取的场景，会先提升成 `_tempN`

### `CALLD = 100`

- 形式：`CALLD dst, obj, method_idx, argc, ...`
- 含义：点方法调用
- 恢复：`obj.method(args...)`

项目里还有一个特殊优化：

- `(new RegExp())._compile("...")` 会尽量折叠回正则字面量

### `CALLI = 101`

- 形式：`CALLI dst, obj, method_expr, argc, ...`
- 含义：索引方法调用
- 恢复：`obj[methodExpr](args...)`

### `NEW = 102`

- 形式：`NEW dst, ctor, argc, ...`
- 恢复：`new ctor(args...)`

项目中特别处理：

- `new Dictionary()` 后跟多条 `SPI`，会恢复成字典字面量
- `new Array()` 后跟多条 `SPI`，会恢复成数组字面量

---

## 11. 属性访问、属性写入与属性引用

这是 TJS2 VM 最核心的一组。

### 11.1 读取属性

| opcode | 值 | 恢复 |
| --- | --- | --- |
| `GPD` | 103 | `obj.prop` |
| `GPI` | 107 | `obj[idx]` |
| `GPDS` | 110 | `&obj.prop` |
| `GPIS` | 112 | `&obj[idx]` |

项目说明：

- `S` 版本在这里被当成“引用/地址风格”
- 某些 `GPD` 会被识别成“死读取”，直接单独输出表达式

### 11.2 写入点属性

| opcode | 值 | 项目中的主要恢复 |
| --- | --- | --- |
| `SPD` | 104 | `obj.prop = value` |
| `SPDE` | 105 | `obj.prop = value` 的变体 |
| `SPDEH` | 106 | `obj.prop = value` 的变体 |
| `SPDS` | 111 | `&obj.prop = value` 风格，或类成员定义相关变体 |

本项目对 `SPD*` 的几个重要处理：

1. 普通情况下恢复成属性赋值。
2. 如果右值还是后续表达式的一部分，先挂进 `_pending_spie`。
3. 在 `class` 中，`SPDS this, name, value` 可能被解释成成员声明。
4. 与函数/属性对象组合时，可能不直接输出普通赋值。

### 11.3 写入索引属性

| opcode | 值 | 恢复 |
| --- | --- | --- |
| `SPI` | 108 | `obj[idx] = value` |
| `SPIE` | 109 | 同类变体 |
| `SPIS` | 113 | `&obj[idx] = value` 风格 |

项目对 `SPI*` 的特殊处理：

- 如果目标对象是 pending Dictionary，则记录成字典条目
- 如果目标对象是 pending Array，则记录成数组元素
- 否则恢复成索引赋值

### 11.4 属性引用读写

| opcode | 值 | 恢复 |
| --- | --- | --- |
| `SETP` | 114 | `*propRef = value` |
| `GETP` | 115 | `*propRef` |

这是理解 `...P` 家族的基础。

### 11.5 删除属性

| opcode | 值 | 恢复 |
| --- | --- | --- |
| `DELD` | 116 | `delete obj.prop` |
| `DELI` | 117 | `delete obj[idx]` |

---

## 12. 返回、异常、作用域与对象上下文

### `SRV = 118`

- 恢复：`return expr`
- `r == 0` 时恢复为 `return;`

### `RET = 119`

- 项目处理：通常忽略
- 作用更像控制流结束标记

### `ENTRY = 120`

- 用于 `try` 区域入口
- 操作数中带 catch 偏移
- 是 CFG 和 `try/catch` 识别的关键指令

### `EXTRY = 121`

- `try` 相关边界标记
- 本项目一般不直接输出

### `THROW = 122`

- 恢复：`throw expr`

### `CHGTHIS = 123`

- 项目中恢复为：`func incontextof obj`
- 是 TJS 上下文绑定的重要指令

### `GLOBAL = 124`

- 恢复：`global`
- 在某些 `with` 上下文中，项目会把它解释成 `WithDotProxy`

---

## 13. 其他指令

### `ADDCI = 125`

- 本项目保留但未直接生成明显源码
- 在 def/use 分析中会记录输入

### `REGMEMBER = 126`

- 项目中不直接产出源码

### `DEBUGGER = 127`

- 项目中忽略

---

## 14. opcode 数值全表

这张表方便你在反汇编输出里快速对照数值与名字。

| 值 | 名称 | 值 | 名称 | 值 | 名称 | 值 | 名称 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `NOP` | 32 | `LANDPI` | 64 | `SUBPI` | 96 | `REAL` |
| 1 | `CONST` | 33 | `LANDP` | 65 | `SUBP` | 97 | `STR` |
| 2 | `CP` | 34 | `BOR` | 66 | `MOD` | 98 | `OCTET` |
| 3 | `CL` | 35 | `BORPD` | 67 | `MODPD` | 99 | `CALL` |
| 4 | `CCL` | 36 | `BORPI` | 68 | `MODPI` | 100 | `CALLD` |
| 5 | `TT` | 37 | `BORP` | 69 | `MODP` | 101 | `CALLI` |
| 6 | `TF` | 38 | `BXOR` | 70 | `DIV` | 102 | `NEW` |
| 7 | `CEQ` | 39 | `BXORPD` | 71 | `DIVPD` | 103 | `GPD` |
| 8 | `CDEQ` | 40 | `BXORPI` | 72 | `DIVPI` | 104 | `SPD` |
| 9 | `CLT` | 41 | `BXORP` | 73 | `DIVP` | 105 | `SPDE` |
| 10 | `CGT` | 42 | `BAND` | 74 | `IDIV` | 106 | `SPDEH` |
| 11 | `SETF` | 43 | `BANDPD` | 75 | `IDIVPD` | 107 | `GPI` |
| 12 | `SETNF` | 44 | `BANDPI` | 76 | `IDIVPI` | 108 | `SPI` |
| 13 | `LNOT` | 45 | `BANDP` | 77 | `IDIVP` | 109 | `SPIE` |
| 14 | `NF` | 46 | `SAR` | 78 | `MUL` | 110 | `GPDS` |
| 15 | `JF` | 47 | `SARPD` | 79 | `MULPD` | 111 | `SPDS` |
| 16 | `JNF` | 48 | `SARPI` | 80 | `MULPI` | 112 | `GPIS` |
| 17 | `JMP` | 49 | `SARP` | 81 | `MULP` | 113 | `SPIS` |
| 18 | `INC` | 50 | `SAL` | 82 | `BNOT` | 114 | `SETP` |
| 19 | `INCPD` | 51 | `SALPD` | 83 | `TYPEOF` | 115 | `GETP` |
| 20 | `INCPI` | 52 | `SALPI` | 84 | `TYPEOFD` | 116 | `DELD` |
| 21 | `INCP` | 53 | `SALP` | 85 | `TYPEOFI` | 117 | `DELI` |
| 22 | `DEC` | 54 | `SR` | 86 | `EVAL` | 118 | `SRV` |
| 23 | `DECPD` | 55 | `SRPD` | 87 | `EEXP` | 119 | `RET` |
| 24 | `DECPI` | 56 | `SRPI` | 88 | `CHKINS` | 120 | `ENTRY` |
| 25 | `DECP` | 57 | `SRP` | 89 | `ASC` | 121 | `EXTRY` |
| 26 | `LOR` | 58 | `ADD` | 90 | `CHR` | 122 | `THROW` |
| 27 | `LORPD` | 59 | `ADDPD` | 91 | `NUM` | 123 | `CHGTHIS` |
| 28 | `LORPI` | 60 | `ADDPI` | 92 | `CHS` | 124 | `GLOBAL` |
| 29 | `LORP` | 61 | `ADDP` | 93 | `INV` | 125 | `ADDCI` |
| 30 | `LAND` | 62 | `SUB` | 94 | `CHKINV` | 126 | `REGMEMBER` |
| 31 | `LANDPD` | 63 | `SUBPD` | 95 | `INT` | 127 | `DEBUGGER` |

---

## 15. 对源码阅读最有帮助的 5 条经验

### 15.1 先按家族看，不要逐个孤立地看

例如看懂 `ADD` 之后，再看 `ADDPD`、`ADDPI`、`ADDP` 会轻松很多。

### 15.2 把 opcode 和“内部状态”一起看

很多指令只有结合下面这些状态才看得懂：

- `regs`
- `flag`
- `pending_arrays`
- `pending_dicts`
- `_pending_spie`

### 15.3 不要把 `CALL` 当成总会直接输出语句

它可能只是更新寄存器表达式，后面再被嵌套进更大的表达式树。

### 15.4 `ENTRY` 不是普通跳转

它是 `try/catch` 结构识别的入口线索。

### 15.5 `RET` 很“弱”，`SRV` 很“强”

读到函数结束时，优先关注 `SRV`，再看 `RET` 是否只是编译器留下的尾标记。

---

## 16. 一句话版理解

如果只用一句话概括这套 TJS2 opcode：

> 它本质上是一套“寄存器 + 条件标志 + 属性访问 + 相对跳转”的字节码，而本项目的工作，就是把这些低层操作重新拼回高层 TJS2 语法。
