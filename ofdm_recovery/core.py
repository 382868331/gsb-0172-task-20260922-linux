"""OFDM 循环前缀边界与子载波恢复（本题契约子集）。

固定参数：N=32 点 FFT，CP=8，前导为两个相同的 16 样本块，随后两个 OFDM 符号。
子载波布局（DFT 索引 k=0..31）：

* 数据载波：k = 1..6 与 26..31（即 ±1..±6），QPSK；
* 导频载波：k=7（已知 +1）、k=25 即 -7（已知 -1）；
* DC（k=0）与 k=8..24 为 0。

信道模型：帧边界已知、无时钟漂移；除载波频偏与单一复增益（L=1）外，
支持最多 ``MAX_CHANNEL_TAPS``（=3）抽头的固定复数因果 FIR
``y[n] = Σ_{l=0}^{L-1} h_l x[n-l]``。用已知训练前导（两个相同 16 点块）
联合 ±7 导频做最小二乘抽头估计，再逐数据载波频域均衡；
信道频率响应的深零点（相对幅度低于门限）标记为**擦除**而不是强行判决。

本模块只实现题目定义的离线合成帧子集，不是完整行业标准接收机。
核心算法（相关、频偏估计、DFT、训练/导频信道估计、均衡判决）均在本文件内
实现，不调用第三方同类实现。
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

# ---------------------------------------------------------------- 协议常量

N_FFT = 32
CP_LEN = 8
BLOCK_LEN = 16
PREAMBLE_LEN = 2 * BLOCK_LEN            # 32
OFDM_SYMBOL_LEN = N_FFT + CP_LEN        # 40
NUM_OFDM_SYMBOLS = 2
FRAME_LEN = PREAMBLE_LEN + NUM_OFDM_SYMBOLS * OFDM_SYMBOL_LEN  # 112
MAX_SIGNAL_LEN = 512

# DFT 索引
DATA_CARRIERS = (1, 2, 3, 4, 5, 6, 26, 27, 28, 29, 30, 31)
# 对外使用的“有符号”载波顺序：-6..-1, +1..+6
SIGNED_DATA_CARRIERS = (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)
PILOT_CARRIERS = (7, 25)
# 已知导频值：+7 载波为 +1，-7 载波（索引 25）为 -1
PILOT_VALUES = {7: 1 + 0j, 25: -1 + 0j}

DEFAULT_RHO_THRESHOLD = 0.85
DEFAULT_RESIDUAL_THRESHOLD = 0.35
# 信号上限 512 时可行起点最多 512-32+1=481，默认预算不会在接口上限处耗尽。
DEFAULT_MAX_CANDIDATES = 1024

# 固定复数 FIR 信道的抽头数上限（题面：最多 3 抽头）。
MAX_CHANNEL_TAPS = 3
DEFAULT_CHANNEL_TAPS = 3
# 深零点擦除门限：数据载波 |H(k)|/sqrt(Σ|h_l|²) < 该值即擦除（不强行判决）。
DEFAULT_ERASURE_THRESHOLD = 0.25

CFO_LIMIT = 0.5  # 合法频偏估计区间 |f| < 0.5 个载波间隔
_PILOT_REL_FLOOR = 1e-9
_TAP_ENERGY_FLOOR = 1e-12
_QPSK = 1.0 / math.sqrt(2.0)

# 与 tests/reference.py 中独立前导块一致的已知训练序列
# （0.8 幅的 16 点 chirp-like 序列），信道估计按题面“已知训练符号”使用。
_TRAINING_AMPLITUDE = 0.8
_TRAINING_BLOCK = [
    _TRAINING_AMPLITUDE
    * cmath.exp(2j * math.pi * (m * (m + 1) // 2) / BLOCK_LEN)
    for m in range(BLOCK_LEN)
]


# ---------------------------------------------------------------- 错误语义


class OFDMError(ValueError):
    """本模块所有错误的基类（同时是 ValueError）。"""


class SignalFormatError(OFDMError):
    """信号本身非法：类型、NaN/Infinity、长度上限等可定位的输入错误。"""


class ParameterError(OFDMError):
    """标量参数非法（阈值、预算等）。"""


class NoFrameError(OFDMError):
    """穷举全部可行起点后，证明不存在满足阈值的帧（与搜索预算耗尽不同）。"""


class SearchBudgetError(OFDMError):
    """候选起点数量超过调用方给定的搜索预算（不代表帧不存在）。"""


class FrequencyOffsetError(OFDMError):
    """最优候选的频偏估计超出契约区间 |f| < 0.5 个载波间隔。"""


class FrameQualityError(OFDMError):
    """最优候选的前导补偿残差超过残差门限，帧不可信。"""


class TruncatedFrameError(OFDMError):
    """找到了前导起点，但信号在两个 OFDM 符号结束前被截断。"""


class ZeroPilotEnergyError(OFDMError):
    """导频载波能量为零（或低于相对底限），无法估计复增益。"""


class ChannelEstimateError(OFDMError):
    """训练+导频方程无法得到可用的 FIR 信道估计（如全零抽头导致矩阵奇异）。"""


# ---------------------------------------------------------------- 输入校验


def _reject_bool(value, name: str) -> None:
    if isinstance(value, bool):
        raise ParameterError(f"{name} must be an int/float, not bool")


def _finite_complex(value, where: str) -> complex:
    """把单个样本转为有限复数；bool、非数值、NaN、Infinity 一律拒绝。"""
    if isinstance(value, bool):
        raise SignalFormatError(f"{where}: bool is not accepted as a sample")
    if isinstance(value, complex):
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            raise SignalFormatError(f"{where}: complex sample is NaN/Infinity")
        return value
    if isinstance(value, int):
        return complex(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SignalFormatError(f"{where}: sample is NaN/Infinity")
        return complex(value)
    raise SignalFormatError(
        f"{where}: expected int/float/complex sample, got {type(value).__name__}"
    )


def _validate_signal(signal) -> list[complex]:
    if not isinstance(signal, (list, tuple)):
        raise SignalFormatError(
            "signal must be a list/tuple of numeric samples, got "
            f"{type(signal).__name__}"
        )
    if len(signal) == 0:
        raise SignalFormatError("signal must contain at least one sample")
    if len(signal) > MAX_SIGNAL_LEN:
        raise SignalFormatError(
            f"signal length {len(signal)} exceeds interface limit "
            f"{MAX_SIGNAL_LEN}"
        )
    return [_finite_complex(v, f"signal[{i}]") for i, v in enumerate(signal)]


def _validate_threshold(value, name: str) -> float:
    _reject_bool(value, name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ParameterError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ParameterError(f"{name} must be finite, got {value}")
    if not 0.0 < value < 1.0:
        raise ParameterError(f"{name} must lie in the open interval (0, 1)")
    return value


def _validate_budget(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterError("max_candidates must be an int, not bool/float")
    if value < 1:
        raise ParameterError("max_candidates must be >= 1")
    return value


def _validate_tap_count(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterError("channel_taps must be an int, not bool/float")
    if not 1 <= value <= MAX_CHANNEL_TAPS:
        raise ParameterError(
            f"channel_taps must lie in [1, {MAX_CHANNEL_TAPS}], got {value}"
        )
    return value


def _validate_unit_interval(value, name: str) -> float:
    """擦除门限等取值于闭区间 [0,1] 的有限实数参数。"""
    _reject_bool(value, name)
    if not isinstance(value, (int, float)):
        raise ParameterError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ParameterError(f"{name} must be finite, got {value}")
    if not 0.0 <= value <= 1.0:
        raise ParameterError(f"{name} must lie in the closed interval [0, 1]")
    return value


# ---------------------------------------------------------------- 自行 DFT


def dft(x) -> list[complex]:
    """长度任意的朴素 DFT：``X[k] = Σ x[n] exp(-j2πkn/N)``（无归一化）。"""
    if not isinstance(x, (list, tuple)) or len(x) == 0:
        raise SignalFormatError("dft input must be a non-empty list/tuple")
    n = len(x)
    xv = [_finite_complex(v, f"dft input[{i}]") for i, v in enumerate(x)]
    return [
        sum(
            (xv[m] * cmath.exp(-2j * math.pi * k * m / n) for m in range(n)),
            0j,
        )
        for k in range(n)
    ]


# ---------------------------------------------------------------- 候选搜索


@dataclass(frozen=True)
class Candidate:
    """一个相关候选起点。

    Attributes:
        start: 前导第一个样本在输入信号中的下标。
        frequency_offset: 由两块相关相位得到的频偏估计（载波间隔为单位）。
        residual: 频偏补偿后的前导归一化失配能量，``1 - 2|C|/(E1+E2)``，
            取值 [0, 1]，越小越好。
        coherence: 归一化相关强度 ``|C|²/(E1·E2)``，取值 [0, 1]。
    """

    start: int
    frequency_offset: float
    residual: float
    coherence: float


def find_candidates(
    signal,
    *,
    rho_threshold: float = DEFAULT_RHO_THRESHOLD,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[Candidate]:
    """穷举全部可行起点，返回通过相关门限的**所有**候选。

    对每个起点 p 计算延迟 M=16 的自相关
    ``C(p) = Σ_{m=0}^{15} x[p+m+16]·conj(x[p+m])`` 及两块能量 E1/E2。
    ``coherence = |C|²/(E1·E2) >= rho_threshold`` 的位置全部保留
    （能量为零的静漠位置跳过），按 (残差, 起点) 升序排列：
    残差为主、最早位置为确定性平局裁决。
    """
    sig = _validate_signal(signal)
    thr = _validate_threshold(rho_threshold, "rho_threshold")
    budget = _validate_budget(max_candidates)

    candidates: list[Candidate] = []
    last = len(sig) - PREAMBLE_LEN
    for p in range(last + 1):
        corr = 0j
        e1 = 0.0
        e2 = 0.0
        for m in range(BLOCK_LEN):
            a = sig[p + m]
            b = sig[p + m + BLOCK_LEN]
            corr += b * a.conjugate()
            e1 += a.real * a.real + a.imag * a.imag
            e2 += b.real * b.real + b.imag * b.imag
        if e1 <= 0.0 or e2 <= 0.0:
            continue
        mag = abs(corr)
        coherence = (mag * mag) / (e1 * e2)
        if coherence < thr:
            continue
        # 两块相隔 16 个样本：相位差 = 2πf·16/32 = πf，故 f = angle(C)/π。
        f_offset = math.atan2(corr.imag, corr.real) / math.pi
        residual = 1.0 - 2.0 * mag / (e1 + e2)
        candidates.append(
            Candidate(p, f_offset, residual, coherence)
        )
        if len(candidates) > budget:
            raise SearchBudgetError(
                f"candidate count exceeded max_candidates={budget} while "
                f"scanning start position p={p}; this is a budget limit, "
                "not a proof that no frame exists"
            )
    candidates.sort(key=lambda c: (c.residual, c.start))
    return candidates


# ---------------------------------------------------------------- 判决辅助


def _slice_qpsk(q: complex) -> complex:
    return complex(
        _QPSK if q.real >= 0.0 else -_QPSK,
        _QPSK if q.imag >= 0.0 else -_QPSK,
    )


def symbols_to_bits(symbols) -> str:
    """按固定映射把 QPSK 符号序列转为位串：实部正→'0' 负→'1'，虚部同理。"""
    if not isinstance(symbols, (list, tuple)):
        raise SignalFormatError("symbols must be a list/tuple")
    out = []
    for i, s in enumerate(symbols):
        if isinstance(s, bool) or not isinstance(s, complex):
            raise SignalFormatError(f"symbols[{i}] must be complex")
        out.append("0" if s.real >= 0.0 else "1")
        out.append("0" if s.imag >= 0.0 else "1")
    return "".join(out)


# ---------------------------------------------------------------- 主解码入口


@dataclass(frozen=True)
class DecodeResult:
    """解码结果。

    Attributes:
        start: 帧起点（前导第一样本）下标。
        frequency_offset: 频偏估计（瞬态安全窗精化后），单位载波间隔，|f| < 0.5。
        residual: 前导补偿归一化失配能量 ``1 - 2|C|/(E1+E2)``，[0,1]，越小越好。
            相关窗剔除了前 ``channel_taps-1`` 个受 FIR 拖尾瞬态污染的样本对。
        gain: 信道第 0 抽头（``channel[0]``）；L=1 时即旧契约的单一复增益。
        channel: 估计出的固定复数 FIR 抽头 ``(h_0, ..., h_{L-1})``，
            L 由 ``channel_taps`` 指定（默认 3）。
        symbols: 24 个判决符号，顺序为符号 0→1，每符号载波 -6..-1、+1..+6；
            非擦除载波为 ±(1±j)/√2，**深零点擦除载波为 ``None``**。
        bits: 48 位恢复位串（先实部后虚部，正为 0）；擦除位以 ``'x'`` 标记。
        erased: 被擦除数据载波的有符号下标元组（取自 -6..-1、+1..+6）。
    """

    start: int
    frequency_offset: float
    residual: float
    gain: complex
    channel: tuple
    symbols: list
    bits: str
    erased: tuple


def _solve_square(a: list[list[complex]], b: list[complex]) -> list[complex]:
    """复线性方程组 a·x=b 的高斯消元（部分主元）；奇异时抛 ChannelEstimateError。"""
    n = len(b)
    m = [list(a[r]) + [b[r]] for r in range(n)]
    scale = max(
        (abs(m[r][c]) for r in range(n) for c in range(n)),
        default=0.0,
    )
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) <= _TAP_ENERGY_FLOOR * max(1.0, scale):
            raise ChannelEstimateError(
                "training/pilot normal equations are singular; cannot estimate "
                f"{n}-tap channel (insufficient training energy)"
            )
        m[col], m[pivot] = m[pivot], m[col]
        pivot_row = m[col]
        for r in range(col + 1, n):
            factor = m[r][col] / pivot_row[col]
            if factor == 0j:
                continue
            row = m[r]
            for j in range(col, n + 1):
                row[j] -= factor * pivot_row[j]
    x = [0j] * n
    for r in range(n - 1, -1, -1):
        x[r] = (
            m[r][n] - sum(m[r][j] * x[j] for j in range(r + 1, n))
        ) / m[r][r]
    return x


def _training_fit_residual(preamble: list[complex], n_taps: int) -> float:
    """补偿后的前导在 L 抽头因果训练模型下的归一化 LS 失配能量。

    块 1 的全部 16 个样本按 ``y[n]=Σ_l h_l p[n-l]``（n-l<0 项为 0）建模，
    块 2 按 16 点循环卷积建模。真起点的前 L-1 个瞬态样本（前置为零）是
    绝对定时标记，使该度量能区分“起点提前 δ 且抽头整体后移 δ”这一
    规范等价位置与真正晚于帧起点的位置。
    """
    rows: list[list[complex]] = []
    obs: list[complex] = []
    for n in range(BLOCK_LEN):
        rows.append([
            _TRAINING_BLOCK[n - l] if n - l >= 0 else 0j
            for l in range(n_taps)
        ])
        obs.append(preamble[n])
    for n in range(BLOCK_LEN, PREAMBLE_LEN):
        m = n - BLOCK_LEN
        rows.append([
            _TRAINING_BLOCK[(m - l) % BLOCK_LEN] for l in range(n_taps)
        ])
        obs.append(preamble[n])
    ata = [
        [
            sum(rows[r][i].conjugate() * rows[r][j] for r in range(len(obs)))
            for j in range(n_taps)
        ]
        for i in range(n_taps)
    ]
    atb = [
        sum(rows[r][i].conjugate() * obs[r] for r in range(len(obs)))
        for i in range(n_taps)
    ]
    taps = _solve_square(ata, atb)
    miss = 0.0
    total = 0.0
    for r, y in enumerate(obs):
        model = sum(rows[r][l] * taps[l] for l in range(n_taps))
        d = model - y
        miss += d.real * d.real + d.imag * d.imag
        total += y.real * y.real + y.imag * y.imag
    if total <= 0.0:
        return 1.0
    return miss / total


def _select_candidate(
    sig: list[complex],
    n_taps: int,
    rho_threshold: float,
    budget: int,
) -> Candidate:
    """确定帧起点（FIR 模式自带瞬态安全窗门控扫描）。

    L=1：直接用外部 ``find_candidates`` 的全窗结果（旧规则），调用方保证
    非空。L>1：全窗相干系数会被前 L-1 个瞬态样本对污染，弱首径信道在真
    起点可能跌破门限，故这里在瞬态安全窗（m∈[L-1,15]）上重新门控扫描
    全部可行起点；再对每个候选做 32 样本全因果训练模型 LS 拟合。

    起点提前 δ≤L-1 与“抽头整体后移 δ、前补零”在数学上规范等价（无噪声
    失配同为 0），故把失配 ≤3 倍最小值（干净时取 1e-12 数值底限）且位置
    相邻（间隔 ≤L-1）的候选归为同一规范簇：多簇并存取最早簇（保留旧版
    “最早帧”平局语义），簇内取**最晚**起点（h₀ 非零的物理规范位置）。
    """
    scored: list[tuple[float, int, float, float, float]] = []
    last = len(sig) - PREAMBLE_LEN
    for p in range(last + 1):
        corr = 0j
        e1 = 0.0
        e2 = 0.0
        for m in range(n_taps - 1, BLOCK_LEN):
            a = sig[p + m]
            b = sig[p + m + BLOCK_LEN]
            corr += b * a.conjugate()
            e1 += a.real * a.real + a.imag * a.imag
            e2 += b.real * b.real + b.imag * b.imag
        if e1 <= 0.0 or e2 <= 0.0:
            continue
        coherence = abs(corr) ** 2 / (e1 * e2)
        if coherence < rho_threshold:
            continue
        f_off = math.atan2(corr.imag, corr.real) / math.pi
        safe_res = 1.0 - 2.0 * abs(corr) / (e1 + e2)
        if safe_res < 0.0:
            safe_res = 0.0
        omega = -2.0 * math.pi * f_off / N_FFT
        comp = [
            sig[p + n] * cmath.exp(1j * omega * n)
            for n in range(PREAMBLE_LEN)
        ]
        fit_res = _training_fit_residual(comp, n_taps)
        scored.append((fit_res, p, f_off, safe_res, coherence))
        if len(scored) > budget:
            raise SearchBudgetError(
                f"candidate count exceeded max_candidates={budget} while "
                f"scanning start position p={p}; this is a budget limit, "
                "not a proof that no frame exists"
            )

    if not scored:
        raise NoFrameError(
            f"exhaustively scanned all {last + 1} feasible start positions "
            f"(transient-safe window for L={n_taps}); none reached "
            f"rho_threshold={rho_threshold} (proven absent under this "
            "contract, not a budget limit)"
        )

    scored.sort(key=lambda t: (t[0], t[1]))
    best_fit = scored[0][0]
    bound = 3.0 * best_fit if best_fit > 1e-13 else 1e-12
    in_cluster = sorted(
        (t for t in scored if t[0] <= bound), key=lambda t: t[1]
    )
    # 规范等价位置只可能出现在真起点的前 L-1 个邻域内，故按“相邻位置
    # 间隔 ≤ L-1”把界内候选切成连续簇；多帧并存时选最早的簇（保留旧版
    # “最早位置”平局语义），簇内取最晚起点（h_0 非零的物理规范位置）。
    runs = [[in_cluster[0]]]
    for t in in_cluster[1:]:
        if t[1] - runs[-1][-1][1] <= n_taps - 1:
            runs[-1].append(t)
        else:
            runs.append([t])
    _, start, f_off, safe_res, coherence = max(
        runs[0], key=lambda t: t[1]
    )
    return Candidate(start, f_off, safe_res, coherence)


def _estimate_taps(
    seg: list[complex],
    spectra: list[list[complex]],
    n_taps: int,
) -> list[complex]:
    """训练前导（时域）联合 ±7 导频（频域）的最小二乘抽头估计。

    起点已取规范位置（h₀ 对应帧首样本），故块 1 的全部 16 个样本按因果
    模型 ``y[n]=Σ_{l≤n} h_l p[n-l]``（n-l<0 项为 0）使用；块 2 两个相同
    训练块构成 16 点循环卷积，n∈[16,31] 全部可用。再叠加每符号两个导频
    观测 ``Y[k] = p[k]·Σ_l h_l exp(-j2πkc·l/N)``，按 A^H A 法方程求解。
    """
    train = _TRAINING_BLOCK
    rows: list[list[complex]] = []
    obs: list[complex] = []

    for n in range(BLOCK_LEN):
        rows.append([
            train[n - l] if n - l >= 0 else 0j for l in range(n_taps)
        ])
        obs.append(seg[n])
    for n in range(BLOCK_LEN, PREAMBLE_LEN):
        m = n - BLOCK_LEN
        rows.append([
            train[(m - l) % BLOCK_LEN] for l in range(n_taps)
        ])
        obs.append(seg[n])

    for yk in spectra:
        for k, pilot in PILOT_VALUES.items():
            kc = k if k < N_FFT // 2 else k - N_FFT
            rows.append([
                complex(pilot)
                * cmath.exp(-2j * math.pi * kc * l / N_FFT)
                for l in range(n_taps)
            ])
            obs.append(yk[k])

    ata = [
        [
            sum(rows[r][i].conjugate() * rows[r][j] for r in range(len(obs)))
            for j in range(n_taps)
        ]
        for i in range(n_taps)
    ]
    atb = [
        sum(rows[r][i].conjugate() * obs[r] for r in range(len(obs)))
        for i in range(n_taps)
    ]
    taps = _solve_square(ata, atb)
    tap_energy = sum(t.real * t.real + t.imag * t.imag for t in taps)
    if tap_energy <= _TAP_ENERGY_FLOOR:
        raise ChannelEstimateError(
            "estimated FIR channel has essentially zero tap energy; "
            "the frame is unusable"
        )
    return taps


def decode(
    signal,
    *,
    rho_threshold: float = DEFAULT_RHO_THRESHOLD,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    channel_taps: int = DEFAULT_CHANNEL_TAPS,
    erasure_threshold: float = DEFAULT_ERASURE_THRESHOLD,
) -> DecodeResult:
    """离线恢复一帧（支持最多 3 抽头固定复数 FIR 信道）。

    失败时抛出 ``OFDMError`` 的具体子类，不返回半结果。深零点数据载波
    （``|H(k)|/sqrt(Σ|h_l|²) < erasure_threshold``）标记擦除：
    ``symbols`` 对应位置为 ``None``，``bits`` 对应两位为 ``'x'``，
    载波下标列入 ``erased``；其余载波做 ZF 均衡后 QPSK 硬判决。
    """
    sig = _validate_signal(signal)
    rho_thr = _validate_threshold(rho_threshold, "rho_threshold")
    res_thr = _validate_threshold(residual_threshold, "residual_threshold")
    n_taps = _validate_tap_count(channel_taps)
    era_thr = _validate_unit_interval(
        erasure_threshold, "erasure_threshold"
    )
    budget = _validate_budget(max_candidates)
    # 定时：L=1 沿用 find_candidates 的全窗穷举（无帧/预算语义不变）；
    # L>1 在 FIR 瞬态安全窗上门控扫描，并在规范等价簇内取最晚物理起点。
    if n_taps == 1:
        candidates = find_candidates(
            sig, rho_threshold=rho_thr, max_candidates=budget
        )
        if not candidates:
            raise NoFrameError(
                f"exhaustively scanned all "
                f"{len(sig) - PREAMBLE_LEN + 1} feasible start positions; "
                f"none reached rho_threshold={rho_threshold} (proven absent "
                "under this contract, not a budget limit)"
            )
        best = candidates[0]
    else:
        best = _select_candidate(sig, n_taps, rho_thr, budget)

    # L=1 时为候选的全窗估计（旧口径）；L>1 时为瞬态安全窗精化值。
    f_offset = best.frequency_offset
    residual = best.residual

    if abs(f_offset) >= CFO_LIMIT:
        raise FrequencyOffsetError(
            f"estimated CFO at start {best.start} is "
            f"{f_offset:.6f} carrier spacings; contract requires "
            f"|f| < {CFO_LIMIT}"
        )
    if residual > res_thr:
        raise FrameQualityError(
            f"best candidate at start {best.start} has preamble residual "
            f"{residual:.6f} > residual_threshold={res_thr}"
        )

    # 择优只用了前导 32 样本，截断检查在频偏/残差之后（与 L=1 旧顺序一致）。
    end = best.start + FRAME_LEN
    if end > len(sig):
        raise TruncatedFrameError(
            f"frame starts at {best.start} and needs {FRAME_LEN} samples "
            f"(through index {end - 1}), but signal has only {len(sig)} "
            "samples"
        )

    raw_frame = sig[best.start:best.start + FRAME_LEN]

    # 全帧频偏补偿（时间原点取帧起点；绝对相位被信道第 0 抽头吸收）。
    omega = -2.0 * math.pi * f_offset / N_FFT
    seg = [
        raw_frame[n] * cmath.exp(1j * omega * n)
        for n in range(FRAME_LEN)
    ]

    # 去 CP 并自行 DFT。
    spectra = []
    for s in range(NUM_OFDM_SYMBOLS):
        base = PREAMBLE_LEN + s * OFDM_SYMBOL_LEN + CP_LEN
        spectra.append(dft(seg[base:base + N_FFT]))

    # 导频能量门限（零导频帧在信道估计前即可定位地拒绝）。
    pilot_energy = 0.0
    total_energy = 0.0
    for yk in spectra:
        for v in yk:
            total_energy += v.real * v.real + v.imag * v.imag
        for k in PILOT_VALUES:
            v = yk[k]
            pilot_energy += v.real * v.real + v.imag * v.imag
    if total_energy <= 0.0 or pilot_energy < _PILOT_REL_FLOOR * total_energy:
        raise ZeroPilotEnergyError(
            "pilot carriers (+/-7) carry no usable energy "
            f"(pilot/total energy ratio floor {_PILOT_REL_FLOOR:g}); "
            "FIR channel cannot be estimated"
        )

    taps = _estimate_taps(seg, spectra, n_taps)
    tap_rms = math.sqrt(sum(t.real * t.real + t.imag * t.imag for t in taps))

    def channel_response(kc: int) -> complex:
        return sum(
            taps[l] * cmath.exp(-2j * math.pi * kc * l / N_FFT)
            for l in range(n_taps)
        )

    erased = tuple(
        kc
        for kc in SIGNED_DATA_CARRIERS
        if abs(channel_response(kc)) / tap_rms < era_thr
    )
    erased_set = set(erased)

    symbols: list = []
    for yk in spectra:
        for kc in SIGNED_DATA_CARRIERS:
            if kc in erased_set:
                symbols.append(None)
            else:
                symbols.append(_slice_qpsk(yk[kc % N_FFT] / channel_response(kc)))

    bit_chars = []
    for s in symbols:
        if s is None:
            bit_chars.append("xx")
        else:
            bit_chars.append("0" if s.real >= 0.0 else "1")
            bit_chars.append("0" if s.imag >= 0.0 else "1")

    return DecodeResult(
        start=best.start,
        frequency_offset=f_offset,
        residual=residual,
        gain=taps[0],
        channel=tuple(taps),
        symbols=symbols,
        bits="".join(bit_chars),
        erased=erased,
    )
