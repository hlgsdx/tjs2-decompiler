# tjs2-decompiler 学习文档

这套文档面向第一次接触 Kirikiri/TJS2 和本仓库的读者，重点不是“怎样运行一下工具”，而是“这个反编译器是如何工作的”。

## 文档目录

- [项目架构总览](./project-architecture.md)
- [模块职责与源码阅读顺序](./module-guide.md)
- [TJS2 虚拟机操作码手册](./tjs2-vm-opcodes.md)

## 建议阅读路径

如果你是 TJS2 小白，推荐按下面的顺序读：

1. 先看 [项目架构总览](./project-architecture.md)，建立整体心智模型。
2. 再看 [模块职责与源码阅读顺序](./module-guide.md)，带着问题去读源码。
3. 最后对照 [TJS2 虚拟机操作码手册](./tjs2-vm-opcodes.md) 回看 `tjs2_decompiler.py` 中的指令翻译逻辑。

## 这套文档的边界

- 文档内容基于当前仓库中的 Python 实现来总结。
- 对 TJS2 bytecode/opcode 的解释，以“本项目如何解析和恢复语义”为准。
- 对少数官方/历史实现细节不完全可见的地方，文档会明确写成“项目内推断/命名约定”。
