# OFDM 循环前缀边界与子载波恢复 + 保留子载波压峰（本题契约子集）

离线数字通信练习库，两部分：

* **接收**：在含**时偏、频偏、复增益与前置噪声**的接收信号中定位
  N=32 / CP=8 的合成帧，估计载波频偏并补偿，去循环前缀后自行做 DFT，
  用已知导频估计单一复增益，最终判决恢复 QPSK 符号与位串。
* **发射（可选，默认关闭）**：保留子载波（tone reservation）迭代投影压峰 ——
  只在调用方指定的保留频点上写入复值，数据/导频/DC 保持不动，
  降低时域峰均比（PAPR），并把保留音调总能量限制在预算内。

**只实现题目定义的有限子集，不是、也不宣称兼容任何完整行业标准（如 802.11/LTE）。**

## 环境与依赖

* Python **3.14.7**（3.10+ 语法即可运行，开发与验收在 3.14.7）。
* **仅标准库**（`math`、`cmath`、`dataclasses`、`unittest`）。安装 Python 后
  无需安装任何其他依赖；核心、测试、演示全部离线，无网络/账号/硬件/数据库。
* 核心算法（相关、频偏估计、DFT/IDFT、导频最小二乘、QPSK 判决）全部在
  `ofdm_recovery/core.py` 内手写，没有用第三方同类实现代替。

## 环境准备（venv）

无需任何第三方依赖，venv 可有可无；如需隔离环境：

```bash
python3.14 -m venv .venv          # 任意 Python 3.10+ 均可，验收版本 3.14.7
source .venv/bin/activate         # 可选；不激活也能直接跑
```

## 运行命令

```bash
python -m unittest discover -s tests -v   # 必要测试（39 个），实测 < 1 s
python demo.py                            # 接收链路演示，实测 < 0.1 s
python demo_iteration.py                  # 保留子载波压峰演示，实测 < 0.1 s（预算 90 s）
```

## 仓库结构

```
ofdm_recovery/__init__.py   公开接口导出
ofdm_recovery/core.py       核心库（DFT、候选搜索、解码、错误类型）
tests/reference.py          独立直接合成器（不导入被测核心；期望值的唯一来源）
tests/test_ofdm.py          接收链路必要测试（24 个）
tests/test_tone_reservation.py  压峰必要测试（15 个，自带独立直接 DFT 核对）
demo.py                     固定输入 demo：1 个正常恢复 + 1 个实际触发的拒绝
demo_iteration.py           压峰 demo：1 次实际压峰 + 1 个零预算边界
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

### `reduce_papr(spectrum, reserved_carriers, *, target_papr_db, energy_budget, max_iterations=64) -> ToneReservationResult`

发射侧保留子载波迭代投影压峰。**默认关闭**：原有接收链路与本库其它接口
完全不经过它，只有显式调用才启用；`tests/reference.py` 的原始发射合成器
保持原样，不使用该功能。

* `spectrum`：长度 32 的频域 OFDM 符号（list/tuple，元素 int/float/complex），
  数据/导频/DC 已就位。
* `reserved_carriers`：保留频点 DFT 索引集合（list/tuple/set）。
  合法频点是 `RESERVABLE_CARRIERS`（k=8..24，即非数据、非导频、非 DC 的零载波）；
  与数据/导频重叠、含 DC、越界、重复、非整数一律抛 `ReservedCarrierError`。
  空集合合法（无自由度，原样返回）。
* `target_papr_db`：目标峰均比（dB），任意有限实数。
* `energy_budget`：保留音调总能量上限 Σ|X[k]|²，有限实数且 ≥ 0；
  `0` 合法（无自由度，原样返回）。
* `max_iterations`：正整数迭代上限（默认 64），拒绝 bool。

算法：每轮按目标（内部门限额外留 1.5 dB 余量防止收敛停滞）计算超门限
时域残差，残差 DFT 后**只保留保留频点**（投影到保留子空间）并从频谱中
减去，再把保留频点整体缩放回能量预算球内。裁剪只用于计算修正量 ——
**发射波形始终是 `IDFT(原数据频谱 + 保留频点修正)`，从不直接裁剪时域
波形**。停止（达标或迭代耗尽）后用直接 DFT 重新变换最终时域符号，
强制核对数据/导频/DC 频点与输入一致（容差 1e-9 相对量级），不一致抛
`OFDMError`（实现 bug），绝不静默放行。

返回不可变 `ToneReservationResult`：

| 字段 | 含义 |
|---|---|
| `spectrum` | 最终频域符号；保护频点与输入一致，只有保留频点被写入 |
| `time_domain` | 最终时域符号（`spectrum` 的 IDFT） |
| `papr_db` | 最终峰均比（dB），取自历史最优一轮 |
| `target_papr_db` / `target_reached` | 目标与是否达成；**达不到只报告 `False`，不抛异常、不承诺每组数据都有足够自由度** |
| `reserved_energy` / `energy_budget` | 保留音调实际总能量与预算；保证 `reserved_energy ≤ budget`（1e-9 相对浮点容差内） |
| `iterations` | 每轮 `PaprIteration(iteration, papr_db, reserved_energy)`，轮 0 为未调整的初始符号 |

错误语义（均继承 `OFDMError(ValueError)`）：

| 异常 | 触发条件 |
|---|---|
| `SignalFormatError` | `spectrum` 类型错、长度非 32、某频点为 NaN/Infinity/bool（消息带 `spectrum[i]`） |
| `ReservedCarrierError` | 保留集合与数据/导频/DC 重叠、越界、重复、非整数、容器类型错 |
| `ParameterError` | `target_papr_db`/`energy_budget` 非有限实数、预算为负、`max_iterations` 非正整数或 bool |

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
* **压峰**：测试用测试文件自带的独立朴素 DFT/IDFT 核对（不调用被测核心）：
  保护频点（数据/导频/DC/其余零载波）偏差 `< 1e-9`（相对频谱最大幅度）；
  报告 PAPR 与独立实测 PAPR 偏差 `< 1e-9` dB；保留能量每轮 `≤ 预算`
  （1e-9 相对容差内）。压峰幅度本身取决于保留频点数与能量预算，
  库不承诺固定的 dB 改善量。

## 已知限制

* 仅支持本文件固定的 N=32/CP=8/2 符号/QPSK/±7 导频布局，不可配置成其他制式。
* 信道模型为单一平坦复增益 + 载波频偏；不处理多径、采样钟偏差、相位噪声、
  帧间增益变化；无纠错编码、无交织。
* 频偏估计范围严格限于 `|f|<0.5` 载波间隔（契约要求），不做整数倍频偏搜索。
* 无窗口/加权相关；强带内干扰下门限（0.85/0.35）可能需要调用方按场景调整。
* 朴素 O(N²) DFT（N 固定 32），未做 FFT 优化——本题规模无此必要。
* 不做 AGC、不输出软信息；DC 与空载波不参与判决。
* 压峰是简单的迭代投影（裁剪残差投影到保留子空间 + 能量球投影），
  不是凸优化求解器：对给定目标可能收敛缓慢或达不到，此时只通过
  `target_reached=False` 如实报告。内部门限留 1.5 dB 余量是固定启发式。
* 保留频点上的额外能量会落在带内零载波区（k=8..24），占用少量带内频谱；
  预算由调用方权衡。接收端无需任何改动：保留频点不参与判决，
  导频增益估计不受影响。

## demo 说明

`python demo.py` 用确定性独立合成器生成一帧（真实时偏 23、频偏 +0.137、
复增益 `1.7·exp(j0.6)`、固定种子噪声 0.05、帧尾噪声），实时打印恢复的
起点、频偏、残差、增益与 48 位位串；然后构造一个前导可定位但在第二符号内
截断的坏帧，实际抛出 `TruncatedFrameError`。无 sleep、无预录结果。

`python demo_iteration.py` 构造一个 12 个数据载波全同相的确定性高 PAPR
频域符号（初始约 10.1 dB），用 10 个保留音调（k=8..12、20..24）在能量
预算 6.0 内迭代压到 6.0 dB 目标以下，逐轮打印 PAPR 与保留能量，并核对
数据/导频/DC 频点未变；随后演示零能量预算边界：无自由度时频谱原样返回、
如实报告未达到目标。全部实时计算，实测远低于 90 秒预算。
