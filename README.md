# OFDM 循环前缀边界与子载波恢复（离线模块）

纯 Python 3.14.7 标准库实现的可导入模块 `ofdm_recovery.py`：从一段复基带
信号中定位一个合成 OFDM 帧、估计并补偿频偏、用自实现的 radix-2 FFT 解调
子载波、由导频估计单一复增益并判决 QPSK。无第三方依赖，全部离线运行。

## 帧契约（本模块只实现此有限子集，不声称兼容任何行业标准）

- N = 32 子载波，CP = 8 样本。
- 前导 = 两个相同的 16 样本块（共 32 样本，前导无 CP）。
- 两个 OFDM 符号，每个 CP+N = 40 样本；整帧 112 样本。
- 数据子载波 k = ±1..±6（每符号 12 个），QPSK 星座 (±1±1j)/√2；
  每帧 24 个 QPSK 符号 = 48 bit。
- 导频子载波 k = −7、+7，取已知实数 ±1（默认均为 +1）。
- DC（k=0）及其余子载波为 0。
- 比特映射：每符号两个比特，第 1 比特定实部符号（0→+1），第 2 比特定
  虚部符号，再除以 √2。

## 接口

```python
import ofdm_recovery as ofdm

result = ofdm.recover(signal, *, pilots=((-7, 1), (7, 1)),
                      threshold=0.8, pilot_min_ratio=1e-12,
                      node_budget=None)
```

- `signal`：实数或复数序列，非空，长度 ≤ 512（接口上限）。元素为
  NaN/Infinity、bool 或非数值时抛 `FrameValidationError`。
- `pilots`：两个 `(子载波, ±1)` 对，子载波必须是 −7 和 +7，值必须为实数
  ±1；否则 `FrameValidationError`。
- `threshold`：候选相干度门限，须满足 0 < threshold < 1。
- `pilot_min_ratio`：导频能量占已用子载波能量的最低比例，低于则判
  “零导频能量”。
- `node_budget`：候选扫描的位置预算（正整数，拒绝 bool）。耗尽时抛
  `SearchBudgetExhaustedError`，与“证明无帧”严格区分。

返回 `RecoveryResult`（frozen dataclass）：

| 字段 | 含义 |
|---|---|
| `symbols` | 24 个 QPSK 判决星座点（按符号、再按载波 −6..−1,1..6 排序） |
| `bits` | 48 字符 bit 串 |
| `start` | 帧起点下标 |
| `cfo` | 频偏估计，单位：子载波间隔（契约保证 \|cfo\| < 0.5 时无歧义） |
| `residual` | 胜出候选的前导残差（0 = 完美重复） |
| `gain` | 导频 LS 估计的单一复增益 |
| `candidates` | 全部候选，按 (residual, start) 排序 |

辅助接口：`find_candidates(signal, threshold, node_budget)` 返回全部候选；
`make_frame(bits, *, pilots, preamble)` 由 48 bit 合成一帧（发射侧便利函
数）；`fft/ifft` 为模块自用的 radix-2 DFT。

## 错误语义

- `FrameValidationError`（继承 `ValueError`）：参数违反契约，错误信息可
  定位到具体参数/下标；不做静默纠正。
- `FrameNotFoundError`：穷举扫描完成，无任何候选达到门限（证明无帧）。
- `TruncatedFrameError`（`FrameNotFoundError` 子类）：最佳候选是有效前导
  但信号在帧中途结束，携带 `start/available/required`。
- `PilotEnergyError`：导频 bin 能量比例低于下限，复增益无定义，携带
  `start/ratio`。
- `SearchBudgetExhaustedError`：搜索预算耗尽，帧存在性未决——与
  `FrameNotFoundError` 是不同结论，互不混用。

模块无全局状态，失败调用不留下任何部分变更。

## 算法与误差口径

1. 滞后 16 的滑动延迟相关（O(L) 递推）计算每个合法起点的相干度
   `|C|/√(E0·E1)` 与频偏校正残差 `(E0+E1−2|C|)/(E0+E1)`。
2. 相干度 ≥ threshold 的全部位置为候选；按残差升序择优，残差完全相等
   时取最早起点（并列候选语义，测试覆盖）。
3. CFO = ∠C/π（滞后 16、N=32，故无歧义范围为 |ε|<0.5 个子载波间隔，
   与题面契约一致；超出该范围的输入不在契约内）。
4. 逐样本补偿 CFO、去 CP、自实现 radix-2 FFT（前向约定
   `X[k]=Σ x[n] e^{−j2πkn/N}`）。
5. 两个导频 LS 合并估计单一复增益 `g=Σ y·x*/Σ|x|²`，数据载波除以 g
   后按符号判决 QPSK。

浮点口径（双精度）：无噪声合成帧上，start 精确，cfo 与 gain 误差在
1e-9 量级，residual 在 1e-15 量级，bit 精确恢复；测试断言即按此口径
（delta=1e-9 / 1e-12），不以宽容差掩盖边界。含噪输入的精度随信噪比退
化，模块不承诺具体数值。

## 设计取舍与已知限制

- 只支持上述固定帧结构（N=32/CP=8/两符号/固定导频位），不是通用 OFDM
  接收机，不声称兼容 802.11 等标准。
- CFO 无歧义范围 |ε|<0.5 子载波间隔（滞后 16 相关的固有性质），更大频
  偏不在契约内。
- 单帧、单复增益（频率选择性信道不建模）；判决为硬判决，无信道编码。
- 门限默认 0.8：纯噪声下误报概率极低，同时容忍前导上的中等噪声。
- 信号长度上限 512 为接口边界，超限直接拒绝而非截断。

## 运行

```bash
python -m unittest discover -s tests -v   # 21 个测试
python demo.py                            # 正常恢复 + 截断拒绝，约 1 秒内
```

测试中的参考发射机（`tests/test_ofdm_recovery.py` 内的直接求和 DFT 合成
器）与 demo 中的合成器都独立于被测模块实现，不调用 `make_frame` 产生期
望值；模块自身 `make_frame` 的往返一致性仅作补充测试。
