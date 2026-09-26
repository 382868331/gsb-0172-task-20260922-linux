# OFDM 循环前缀边界与子载波恢复（本题契约子集）

离线数字通信练习库：在含**时偏、频偏、复增益与前置噪声**的接收信号中定位
N=32 / CP=8 的合成帧，估计载波频偏并补偿，去循环前缀后自行做 DFT，
用已知导频估计单一复增益，最终判决恢复 QPSK 符号与位串。
本轮迭代在硬判决之后新增 **QPSK / 16QAM 软解调**：按题内显式 Gray 表
对每个数据比特输出对数似然比（LLR，**正值偏向比特 0**），支持擦除
子载波与信道补偿后的噪声方差调整。

**只实现题目定义的有限子集，不是、也不宣称兼容任何完整行业标准（如 802.11/LTE）。**

## 环境与依赖

* Python **3.14.7**（3.10+ 语法即可运行，开发与验收在 3.14.7）。
* **仅标准库**（`math`、`cmath`、`dataclasses`、`unittest`）。安装 Python 后
  无需安装任何其他依赖；核心、测试、演示全部离线，无网络/账号/硬件/数据库。
* venv 准备（可选，仓库不强制）：`python3 -m venv .venv && . .venv/bin/activate`，
  之后无需 `pip install` 任何东西。
* 核心算法（相关、频偏估计、DFT/IDFT、导频最小二乘、QPSK 判决）全部在
  `ofdm_recovery/core.py` 内手写，没有用第三方同类实现代替。

## 运行命令

```bash
python -m unittest discover -s tests -v   # 必要测试，约 0.05 s
python demo.py                            # 固定输入演示，实测 < 0.1 s（远低于约 8 s 预算）
python demo_iteration.py                  # 本轮软解调演示，实测 < 0.1 s（预算 90 s）
```

## 仓库结构

```
ofdm_recovery/__init__.py   公开接口导出
ofdm_recovery/core.py       核心库（DFT、候选搜索、解码、错误类型）
tests/reference.py          独立直接合成器（不导入被测核心；期望值的唯一来源）
tests/test_ofdm.py          必要测试（24 个）
demo.py                     固定输入 demo：1 个正常恢复 + 1 个实际触发的拒绝
tests/test_soft.py          软解调测试（19 个，含独立直接概率求和对照）
demo_iteration.py           软解调演示：正常帧 LLR + 判决边界/深衰落/擦除
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
from ofdm_recovery import decode, find_candidates, dft, symbols_to_bits, DecodeResult
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

### 软解调（本轮新增）

```python
from ofdm_recovery import (
    soft_demodulate, SoftDemodResult, QPSK_GRAY, QAM16_GRAY, decode,
)
```

#### `soft_demodulate(symbols, *, modulation="qpsk", noise_variance, gains=None, erased=None) -> SoftDemodResult`

对星座观测序列做软解调，返回不可变 `SoftDemodResult`：

| 字段 | 含义 |
|---|---|
| `modulation` | `"qpsk"` 或 `"16qam"` |
| `llrs` | 每比特 LLR 元组（每符号 2 或 4 位，组内位序与 Gray 表键一致） |
| `erasures` | 每符号擦除标志元组 |

* **星座映射**：题内显式 Gray 表常量 `QPSK_GRAY`（4 项）与 `QAM16_GRAY`
  （16 项，电平 ±1/±3 按 1/√10 归一化，沿每根轴 Gray 序 00→-3, 01→-1,
  11→+1, 10→+3）。QPSK 表与硬判决映射一致（b0=实部、b1=虚部，正→0）。
* **LLR 约定**：`LLR = log P(b=0) − log P(b=1)`，**正值偏向比特 0**；
  两组星座概率用 log-sum-exp 稳定累加。
* `noise_variance`：调用方给出的**正**复噪声方差 σ²=E|n|²；
  非正、NaN/Infinity、bool 一律 `ParameterError`。
* `gains`：可选的每符号复增益；给出时观测按 y/h 补偿，噪声方差同步
  调整为 σ²/|h|²（深衰落时 LLR 幅度随之缩小）。**零增益按擦除处理**。
* **擦除**：`erased` 显式标记或零增益触发的擦除符号，全部比特返回
  `LLR=0.0` 且 `erasures` 置位——不硬判成全 0。

#### `decode(..., noise_variance=None)`

给出正的 `noise_variance` 时，`DecodeResult` 追加两个字段（否则为 `None`）：
`llrs`（48 个数据比特的 QPSK LLR，位序与 `bits` 完全一致）与
`erasures`（24 个数据符号标志）。软解调只作用于 ±1..±6 数据载波——
**有效导频、DC 与保护带不进入数据位序列**；补偿用估计出的单一复增益，
方差按 σ²/|gain|² 调整。旧硬判决字段 `symbols`/`bits` 位序与取值不变。

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
| `ZeroPilotEnergyError` | ±7 导频能量低于总能量的 `1e-9` 相对底限，无法估计增益 |

整数参数（如 `max_candidates`）拒绝 `bool`；所有标量数值拒绝 NaN/Infinity。
操作无共享可变状态（纯函数），失败即抛异常，不产生部分结果。

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
* **软解调**：LLR 与测试内独立直接概率求和参考（逐点 `exp` 累加后取对数比）
  在星座中心、Gray 中点等样例上绝对误差 `< 5e-13`（`places=12`）；
  极低信噪比（σ²=1e6）时 |LLR| `< 1e-4` 趋近 0；深衰落 |h|=1e-3 时
  |LLR| 较无衰落缩小约 6 个数量级；QPSK 判决边界上 LLR 精确为 0。

## 已知限制

* 仅支持本文件固定的 N=32/CP=8/2 符号/QPSK/±7 导频布局，不可配置成其他制式。
* 信道模型为单一平坦复增益 + 载波频偏；不处理多径、采样钟偏差、相位噪声、
  帧间增益变化；无纠错编码、无交织。
* 频偏估计范围严格限于 `|f|<0.5` 载波间隔（契约要求），不做整数倍频偏搜索。
* 无窗口/加权相关；强带内干扰下门限（0.85/0.35）可能需要调用方按场景调整。
* 朴素 O(N²) DFT（N 固定 32），未做 FFT 优化——本题规模无此必要。
* 不做 AGC；DC 与空载波不参与判决与软解调。
* 软解调输出的是比特级 LLR，**不做纠错码解码**（无 Viterbi/BCJR/LDPC）；
  噪声模型为每符号独立同分布复高斯，调用方需自行给出正确的 σ²。

## demo 说明

`python demo.py` 用确定性独立合成器生成一帧（真实时偏 23、频偏 +0.137、
复增益 `1.7·exp(j0.6)`、固定种子噪声 0.05、帧尾噪声），实时打印恢复的
起点、频偏、残差、增益与 48 位位串；然后构造一个前导可定位但在第二符号内
截断的坏帧，实际抛出 `TruncatedFrameError`。无 sleep、无预录结果。

## demo_iteration 说明（本轮新增）

`python demo_iteration.py` 实时计算并打印：

1. **正常帧**：含噪信道下 `decode(..., noise_variance=0.02)` 同时给出硬判决
   位串与 48 个数据比特 LLR（导频/DC/保护带不在内），并验证硬判决位序不变；
2. **判决边界**：QPSK 观测沿实轴扫过边界，LLR 连续趋近 0 并过零变号；
3. **16QAM**：星座中心、Gray 中点（符号位 LLR 精确为 0）、深衰落
   （|h|=1e-3，LLR 趋近 0）与零增益擦除（LLR=0 且保留擦除标志）；
4. **显式擦除**：擦除位不硬判成全 0，而是 LLR=0 加擦除标志。

全程实测 < 0.1 s（预算 90 s），无 sleep、无预录输出。
