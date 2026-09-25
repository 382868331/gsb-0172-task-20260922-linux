# OFDM 循环前缀边界与子载波恢复（本题契约子集）

离线数字通信练习库：在含**时偏、频偏、复增益与前置噪声**的接收信号中定位
N=32 / CP=8 的合成帧，估计载波频偏并补偿，去循环前缀后自行做 DFT，
用已知导频估计单一复增益，最终判决恢复 QPSK 符号与位串。

两个接收入口：`decode` 恢复**单帧**（原接口）；`scan_frames` 在有限缓冲区内
**一次扫描多帧**——枚举候选、逐候选独立评估、区间 DP 选出互不重叠的一组，
被拒绝候选与尾部不完整帧分别另列。

**只实现题目定义的有限子集，不是、也不宣称兼容任何完整行业标准（如 802.11/LTE）。**

## 环境与依赖

* Python **3.14.7**（3.10+ 语法即可运行，开发与验收在 3.14.7）。
* **仅标准库**（`math`、`cmath`、`bisect`、`dataclasses`、`unittest`）。安装 Python 后
  无需安装任何其他依赖；核心、测试、演示全部离线，无网络/账号/硬件/数据库。
* 核心算法（相关、频偏估计、DFT/IDFT、导频最小二乘、QPSK 判决、区间 DP）
  全部在 `ofdm_recovery/core.py` 内手写，没有用第三方同类实现代替。

## 运行命令

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 必要测试，约 0.2 s
./.venv/bin/python demo.py                            # 单帧演示，实测 < 0.1 s
./.venv/bin/python demo_scan.py                       # 多帧扫描演示，实测 < 0.1 s（远低于 90 s 预算）
```

## 仓库结构

```
ofdm_recovery/__init__.py   公开接口导出
ofdm_recovery/core.py       核心库（DFT、候选搜索、单帧解码、多帧扫描、区间 DP、错误类型）
tests/reference.py          独立直接合成器（不导入被测核心；期望值的唯一来源）
tests/test_ofdm.py          单帧必要测试（24 个）
tests/test_scan.py          多帧扫描必要测试（18 个）
demo.py                     单帧 demo：1 个正常恢复 + 1 个实际触发的拒绝
demo_scan.py                多帧 demo：两帧 + 假峰 + 尾帧 + 最优性暴力核对
```

`tests/reference.py` 是与核心互相独立的发射端实现：朴素 IDFT、独立前导序列、
独立 QPSK 映射；测试期望值（真实时偏、频偏、增益、位串）全部来自合成器的
**输入参数**，不调用被测核心反推。往返一致测试仅作补充。

## 帧结构（固定常量）

| 项 | 值 |
|---|---|
| FFT 点数 N / 循环前缀 CP | 32 / 8 |
| 前导 | 两个相同的 16 样本复块（32 样本） |
| OFDM 符号 | 2 个，每个 40 样本（CP + 32） |
| 帧长 | 112 样本 |
| 数据载波 | ±1..±6（DFT 索引 1..6、26..31），QPSK，共 24 符号/帧 |
| 导频载波 | +7（索引 7，已知 +1）、−7（索引 25，已知 −1） |
| DC 与其余载波 | 0 |
| 输入信号长度上限 | **512 样本**（含帧前后噪声） |

QPSK 星座为 `±(1+j)/√2`。位映射：每位对应一个坐标，实部在前、虚部在后，
正→`0`、负→`1`。符号输出顺序：符号 0→1，每符号载波 −6..−1、+1..+6；
共 24 符号、48 位。

## 公开接口

```python
from ofdm_recovery import (
    decode, find_candidates, scan_frames, dft, symbols_to_bits,
    DecodeResult, Candidate, ScanResult, ScannedFrame,
    RejectedCandidate, TruncatedCandidate,
)
```

### `decode(signal, *, rho_threshold=0.85, residual_threshold=0.35, max_candidates=1024) -> DecodeResult`

离线恢复一帧。`signal` 为 `list/tuple`，元素为 `int/float/complex`（长度 1..512）。
返回不可变 `DecodeResult`：

| 字段 | 含义 |
|---|---|
| `start` | 帧起点（前导第一样本）下标 |
| `frequency_offset` | 频偏估计，单位载波间隔，满足 `|f| < 0.5` |
| `residual` | 前导补偿归一化失配能量 `1 − 2|C|/(E1+E2)`，∈[0,1]，越小越好 |
| `gain` | 四个导频观测最小二乘得到的单一复增益 |
| `symbols` | 24 个 QPSK 判决符号（顺序见上） |
| `bits` | 48 位恢复位串 |

### `find_candidates(signal, *, rho_threshold=0.85, max_candidates=1024) -> list[Candidate]`

穷举**全部**可行起点（前导 32 样本能完整放入的位置），计算延迟 16 的自相关
`C(p)=Σ x[p+m+16]·conj(x[p+m])` 及两块能量 E1/E2。
保留 `coherence = |C|²/(E1·E2) ≥ rho_threshold` 的全部候选
（零能量静默位置跳过），按 **(残差升序, 起点升序)** 排序 —— 以前导残差择优、
最早位置确定性平局。`Candidate` 字段：`start / frequency_offset / residual / coherence`。

其余接口：`dft(x)`（任意长度朴素 DFT，无归一化）、`symbols_to_bits(symbols)`。

### `scan_frames(signal, *, rho_threshold=0.85, residual_threshold=0.35, max_candidates=1024) -> ScanResult`

在有限缓冲区（≤512 样本）内**一次扫描并恢复多帧**。流程：

1. 用与 `find_candidates` 相同的前导相关枚举**全部**候选起点；
2. 对每个候选**独立**求频偏、前导残差、导频残差与完整性，按顺序判定：
   * `|频偏| ≥ 0.5` → 拒绝，原因 `REASON_CFO_OUT_OF_RANGE`；
   * 前导残差 > `residual_threshold` → 拒绝，原因 `REASON_PREAMBLE_RESIDUAL`；
   * 帧在缓冲区末尾前不完整 → 列入 `truncated`（**只登记，不补零、不当完整帧**）；
   * 导频能量低于相对底限 → 拒绝，原因 `REASON_ZERO_PILOT_ENERGY`；
   * 导频残差 > `residual_threshold` → 拒绝，原因 `REASON_PILOT_RESIDUAL`；
3. 对合格候选用**区间动态规划**选出互不重叠（帧区间 `[start, start+112)`
   不相交）的一组，选择顺序：完整可解**帧数最多** → **前导残差总和最小**
   → **起点序列字典序最小**。不是"见第一个峰就跳过整段"的贪心。

返回不可变 `ScanResult`（各列表均按起点升序）：

| 字段 | 含义 |
|---|---|
| `frames` | 选中的帧，`ScannedFrame`：`start / frequency_offset / residual / pilot_residual / gain / symbols / bits`（字段含义同 `DecodeResult`，`pilot_residual` 为导频最小二乘归一化剩余能量，∈[0,1]） |
| `rejected` | `RejectedCandidate`：`start / frequency_offset / residual / coherence / reason`，`reason` 取上面四个 `REASON_*` 字符串常量之一 |
| `truncated` | `TruncatedCandidate`：`start / frequency_offset / residual / coherence / available`（`available` = 缓冲区中自 `start` 起实际可用样本数，< 112） |
| `not_selected` | 质量合格但因重叠竞争落败的 `Candidate` 列表 |

与 `decode` 的差异：扫描**不抛** `NoFrameError`——缓冲区无任何候选时返回全空
`ScanResult`；输入非法仍抛 `SignalFormatError` / `ParameterError`，候选数超预算
仍抛 `SearchBudgetError`（预算耗尽与证明无帧仍然分开）。

**导频残差定义**：四个导频观测（2 符号 × ±7 载波）对单一复增益做最小二乘后的
归一化剩余能量 `1 − |ΣY·conj(p)|² / (E_pilot·Σ|p|²)`。前导形似但导频错误的假峰
（如前导正常、导频发错值）前导残差 ≈0 能通过相关门限，但导频残差 ≈1，被
`REASON_PILOT_RESIDUAL` 拒绝；其前导相关平台衍生的错位幽灵候选（±1..2 样本）
同样因导频残差或前导残差被拒，不会借错位混入输出帧。

### 错误语义（均可定位、不静默纠正）

所有错误继承 `OFDMError(ValueError)`：

| 异常 | 触发条件 |
|---|---|
| `SignalFormatError` | 信号类型错、空、长度 >512、某样本为 NaN/Infinity（消息带 `signal[i]`）、bool 样本 |
| `ParameterError` | 阈值不在开区间 (0,1)、NaN、`max_candidates` 非正整数、bool 当整数 |
| `NoFrameError` | **穷举全部可行起点后证明无候选**（消息含扫描数量），区别于预算耗尽 |
| `SearchBudgetError` | 候选数超过 `max_candidates`，消息明确声明“这是预算限制，不证明帧不存在” |
| `FrequencyOffsetError` | 最优候选频偏 `|f| ≥ 0.5`，超出契约区间 |
| `FrameQualityError` | 最优候选残差 > `residual_threshold`（含“相关很高但幅度失配”的情形） |
| `TruncatedFrameError` | 前导定位成功，但信号在 112 样本帧结束前截断 |
| `ZeroPilotEnergyError` | ±7 导频能量低于总能量的 `1e-9` 相对底限，或导频观测相互抵消使增益估计恰为 0，无法估计/归一化增益 |

整数参数（如 `max_candidates`）拒绝 `bool`；所有标量数值拒绝 NaN/Infinity。
操作无共享可变状态（纯函数），失败即抛异常，不产生部分结果。
例外：`scan_frames` 的逐候选失败**不抛异常**——被拒绝/截断的候选带原因
记录在 `ScanResult` 中返回；无候选时返回全空结果而非 `NoFrameError`。

## 算法与设计取舍

1. **定时同步**：延迟 M=16 的自相关找重复前导块；相干性门限枚举**所有**候选，
   不是只取 argmax。择优用残差（同时惩罚相位与幅度失配）而非仅相关强度 ——
   纯幅度缩放的两块 `coherence=1` 但残差很大，会被正确降级。
2. **频偏估计**：两块相隔 16 样本，相位差 `2πf·16/32 = πf`，
   故 `f = arg(C)/π`，无歧义区间恰为 `|f|<0.5`；用 `atan2` 主值。
   补偿乘子 `exp(−j2πf n/32)`，时间原点取帧起点（绝对相位由导频增益吸收）。
3. **去 CP + DFT**：每符号跳过前 8 点，对 32 点做手写 DFT。
4. **信道估计**：两符号 × 两导频 = 4 个观测，最小二乘单一复增益
   `h = Σ Y·conj(p)/Σ|p|²`（导频 ±1 时即平均），再逐数据载波 `Y/h` 后硬判决。
5. 无插值/均衡/跟踪：契约只有一个平坦复增益。
6. CP=8 足够吸收定时小幅滑动；起点判决以残差为准，残差相同选最早位置。
7. **多帧选择用区间 DP**：候选按起点排序后，`dp[i] = max(跳过 i, 选 i + dp[下一不重叠])`，
   比较键为 (帧数, 负残差总和)，完全并列时取起点序列字典序最小者；
   复杂度 O(候选数 log 候选数)。测试用短缓冲上候选子集的暴力枚举核对最优性。

## 浮点误差口径（明确，不用宽容差掩盖边界错误）

以下为确定性测试中实测/断言的口径，测试用固定种子噪声、真实计算：

* **无噪声、无截断**：起点**精确**；频偏、复增益绝对误差 `< 1e-9`
  （干净合成信号下实测机器精度量级 `~1e-15`）；48 位全对；残差 `=0` 到 `1e-12`。
* **前置+带内噪声**：合成器噪声为两个均匀分布之和，幅度参数 0.05 时
  每坐标标准差 ≈ 0.041（信号单位幅度量级）。测试断言：起点精确；
  频偏绝对误差 `< 0.02` 载波间隔；增益模值误差 `< 10%`；残差 `< 0.02`；
  48 位全对。噪声更大时位错误属物理现象，库不承诺 SNR 性能曲线。
* **边界**：`|cfo|=0.49` 可恢复；`|f|≥0.5` 必然被 `FrequencyOffsetError` 拒绝
  （估计区间本身在 ±0.5 处分界）。信号长度 = 512（时偏 400+帧 112）可正常解码，
  长度 = 513 抛 `SignalFormatError`。
* DFT 单元测试对已知变换对（冲激、直流、单频）误差 `< 1e-10`。

## 已知限制

* 仅支持本文件固定的 N=32/CP=8/2 符号/QPSK/±7 导频布局，不可配置成其他制式。
* 信道模型为单一平坦复增益 + 载波频偏；不处理多径、采样钟偏差、相位噪声、
  帧间增益变化；无纠错编码、无交织。
* 频偏估计范围严格限于 `|f|<0.5` 载波间隔（契约要求），不做整数倍频偏搜索。
* 无窗口/加权相关；强带内干扰下门限（0.85/0.35）可能需要调用方按场景调整。
* 朴素 O(N²) DFT（N 固定 32），未做 FFT 优化——本题规模无此必要。
* 不做 AGC、不输出软信息；DC 与空载波不参与判决。
* 前导的自相关在真实起点 ±1..2 样本处存在相干平台：这些错位候选会进入
  候选集，由逐候选的前导/导频残差检查与区间 DP 处理（真帧的错位候选被
  导频残差拒绝或在重叠竞争中落败；见 `scan_frames` 一节）。多帧扫描假定
  帧在样本意义上不重叠；缓冲区内帧数上界为 ⌊512/112⌋ = 4。

## demo 说明

`./.venv/bin/python demo.py` 用确定性独立合成器生成一帧（真实时偏 23、频偏 +0.137、
复增益 `1.7·exp(j0.6)`、固定种子噪声 0.05、帧尾噪声），实时打印恢复的
起点、频偏、残差、增益与 48 位位串；然后构造一个前导可定位但在第二符号内
截断的坏帧，实际抛出 `TruncatedFrameError`。无 sleep、无预录结果。

`./.venv/bin/python demo_scan.py` 构造一段 413 样本缓冲区：前置噪声 + 帧 A
（起点 17，频偏 +0.21）+ 相邻帧 B（起点 129，频偏 −0.33，位串与增益均不同）
+ 一个前导正常但导频错误的假峰（起点 241）+ 一个只保留 60 样本的尾部不完整帧
（起点 353）。演示实时打印扫描结果：两帧的起点/频偏/残差/增益/48 位位串并与
发送位串核对，假峰以 `pilot_residual` 拒绝（含 ±2 样本内的错位幽灵候选），
尾帧列入 `truncated` 不解码；最后对缓冲区内全部合格候选做子集暴力枚举，
核对区间 DP 选择的 (帧数, 残差总和, 字典序) 最优性。实测 < 0.1 s，远低于
90 s 预算；无 sleep、无预录结果。
