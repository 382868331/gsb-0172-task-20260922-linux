"""OFDM 循环前缀边界与子载波恢复（本题契约子集）。

固定参数：N=32 点 FFT，CP=8，前导为两个相同的 16 样本块，随后两个 OFDM 符号。
子载波布局（DFT 索引 k=0..31）：

* 数据载波：k = 1..6 与 26..31（即 ±1..±6），QPSK；
* 导频载波：k=7（已知 +1）、k=25 即 -7（已知 -1）；
* DC（k=0）与 k=8..24 为 0。

除单一复增益外，还支持**最多 3 抽头的固定复数 FIR 信道**
``y[n] = h0 x[n] + h1 x[n-1] + h2 x[n-2]``：帧边界已知、无时钟漂移，
用前导第二块（训练符号的稳态段）在偶数载波上做最小二乘抽头估计，
再按解析式给出全部载波响应并均衡；响应深零点（相对 RMS 低于门限）的
数据载波标记为擦除，不伪造判决。无噪声时所有未擦除位精确恢复。

本模块只实现题目定义的离线合成帧子集，不是完整行业标准接收机。
核心算法（相关、频偏估计、DFT、训练符号信道估计、判决）均在本文件内实现，
不调用第三方同类实现。
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

CFO_LIMIT = 0.5  # 合法频偏估计区间 |f| < 0.5 个载波间隔
_PILOT_REL_FLOOR = 1e-9
_QPSK = 1.0 / math.sqrt(2.0)

# ---------------------------------------------------------------- FIR 信道
# 信道模型：y[n] = h0·x[n] + h1·x[n-1] + h2·x[n-2]，帧内固定复抽头。
MAX_CHANNEL_TAPS = 3
# 数据载波 k 被判为深零点（擦除）的相对门限：
# |H(k)| < erasure_threshold · rms(H)，rms(H) = sqrt(|h0|²+|h1|²+|h2|²)。
# 精确零点相邻载波的相对响应约为 0.139，0.1 只擦除真正的近零点。
DEFAULT_ERASURE_THRESHOLD = 0.1

# 训练块（契约常量）：16 点复序列，前导 = 该块重复两次。
# 其 16 点 DFT 在 q=0 处为精确零点（Σ 遍历全部 16 次单位根），
# 其余频点非零；信道估计只使用非零频点。
TRAINING_BLOCK = tuple(
    0.8 * cmath.exp(2j * math.pi * (m * (m + 1) // 2) / BLOCK_LEN)
    for m in range(BLOCK_LEN)
)


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
    """训练符号上的 FIR 抽头最小二乘方程组病态，信道无法估计。"""


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


# FIR 拖尾占用每块前 MAX_CHANNEL_TAPS-1 个样本；相关只在其后的稳态段
# 进行（第二块剩余样本是第一块的循环移位重复，见模块文档）。
CORR_TRANSIENT = MAX_CHANNEL_TAPS - 1
CORR_TERMS = BLOCK_LEN - CORR_TRANSIENT  # 14 个稳态相关项


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
    （能量为零的静默位置跳过），按 (残差, 起点) 升序排列：
    残差为主、最早位置为确定性平局裁决。多径下真起点的最终确认由
    ``decode`` 用训练符号的 3 抽头拟合残差完成，不在本函数内做。
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
    """按固定映射把 QPSK 符号序列转为位串：实部正→'0' 负→'1'，虚部同理。

    被擦除的符号（``None``）映射为两位 ``'xx'``，不伪造判决。
    """
    if not isinstance(symbols, (list, tuple)):
        raise SignalFormatError("symbols must be a list/tuple")
    out = []
    for i, s in enumerate(symbols):
        if s is None:
            out.append("xx")
            continue
        if isinstance(s, bool) or not isinstance(s, complex):
            raise SignalFormatError(f"symbols[{i}] must be complex or None")
        out.append("0" if s.real >= 0.0 else "1")
        out.append("0" if s.imag >= 0.0 else "1")
    return "".join(out)


# ---------------------------------------------------------------- FIR 信道估计


@dataclass(frozen=True)
class ChannelEstimate:
    """训练符号估计出的固定 FIR 信道及其频域结果。

    Attributes:
        taps: 恒为 3 个复抽头 ``(h0, h1, h2)``；真实抽头少于 3 个时，
            多余位置在无噪声下估计为 0（有数值误差）。
        rms_gain: 抽头 RMS 增益 ``sqrt(|h0|²+|h1|²+|h2|²)``，
            它也是全部载波频响模值的 RMS（Parseval）。
        frequency_response: 32 个载波的信道频响 ``H[k]``，
            ``H[k] = Σ_l h_l exp(-j2πkl/32)``。
        erased_carriers: 被判深零点的**有符号**数据载波（如 ``(-4, 3)``）；
            信道帧内固定，两个 OFDM 符号的擦除载波相同。
        fit_residual: 训练符号 3 抽头联合拟合的归一化残差 J（[0,1]，
            越小越好）；无噪声任意多径下 ≈0，是帧质量在多径下的
            有效度量（前导相关残差在多径下会因瞬态项非零）。
    """

    taps: tuple[complex, complex, complex]
    rms_gain: float
    frequency_response: tuple[complex, ...]
    erased_carriers: tuple[int, ...]
    fit_residual: float


# 训练符号联合设计矩阵（32 行 × 3 抽头），契约常量，模块加载时构建：
# * 第一块（帧内 n=0..15）：因果零前史，y[n]=h0·b[n]+h1·b[n-1]+h2·b[n-2]，
#   帧外样本为 0 —— 起始瞬态是区分 ±2 样本定时等价类的关键信息；
# * 第二块（n=16..31）：稳态循环卷积，y[16+m]=Σ_l h_l·b[(m-l) mod 16]。
def _build_design_rows():
    b = TRAINING_BLOCK
    rows = []
    for m in range(BLOCK_LEN):
        rows.append((
            b[m],
            b[m - 1] if m >= 1 else 0j,
            b[m - 2] if m >= 2 else 0j,
        ))
    for m in range(BLOCK_LEN):
        rows.append((b[m], b[(m - 1) % BLOCK_LEN], b[(m - 2) % BLOCK_LEN]))
    return tuple(rows)


_DESIGN_ROWS = _build_design_rows()
_DESIGN_NORMAL = tuple(
    tuple(
        sum(row[i].conjugate() * row[j] for row in _DESIGN_ROWS)
        for j in range(MAX_CHANNEL_TAPS)
    )
    for i in range(MAX_CHANNEL_TAPS)
)


def _solve3_normal(A, b):
    """带部分选主元的 3×3 复高斯消元；奇异时返回 None。"""
    mat = [
        [complex(A[r][0]), complex(A[r][1]), complex(A[r][2]), complex(b[r])]
        for r in range(3)
    ]
    scale = max(
        abs(mat[r][c]) for r in range(3) for c in range(3)
    ) or 1.0
    for col in range(3):
        pivot_row = max(range(col, 3), key=lambda r: abs(mat[r][col]))
        mat[col], mat[pivot_row] = mat[pivot_row], mat[col]
        pivot = mat[col][col]
        if abs(pivot) < 1e-12 * scale:
            return None
        for r in range(3):
            if r == col:
                continue
            factor = mat[r][col] / pivot
            if factor == 0j:
                continue
            for c in range(col, 4):
                mat[r][c] -= factor * mat[col][c]
    return [mat[i][3] / mat[i][i] for i in range(3)]


def _steady_cfo(sig, start) -> float:
    """稳态窗口（第二块 m=2..15）的延迟相关频偏：多径下无偏。"""
    corr = 0j
    for m in range(CORR_TRANSIENT, BLOCK_LEN):
        corr += sig[start + m + BLOCK_LEN] * sig[start + m].conjugate()
    if corr == 0j:
        return 0.0
    return math.atan2(corr.imag, corr.real) / math.pi


def _fit_channel(sig, start, frequency_offset):
    """在假设起点处做 3 抽头联合 LS，返回 (taps, 归一化训练拟合残差 J)。

    J 为 32 个训练样本的残差能量占接收训练能量的比例；
    真起点处模型精确（无噪声时 J≈0），偏移/伪起点处 J 显著增大。
    """
    seg = [
        sig[start + n] * cmath.exp(-2j * math.pi * frequency_offset * n / N_FFT)
        for n in range(PREAMBLE_LEN)
    ]
    target = [
        sum(row[i].conjugate() * seg[r] for r, row in enumerate(_DESIGN_ROWS))
        for i in range(MAX_CHANNEL_TAPS)
    ]
    taps = _solve3_normal(_DESIGN_NORMAL, target)
    if taps is None:
        raise ChannelEstimateError(
            f"3-tap LS normal equations are singular at start {start}; "
            "received training block is too distorted to estimate the channel"
        )
    residual = 0.0
    energy = 0.0
    for r, row in enumerate(_DESIGN_ROWS):
        fit = taps[0] * row[0] + taps[1] * row[1] + taps[2] * row[2]
        diff = seg[r] - fit
        residual += diff.real * diff.real + diff.imag * diff.imag
        energy += seg[r].real * seg[r].real + seg[r].imag * seg[r].imag
    j_fit = residual / energy if energy > 0.0 else 1.0
    return taps, j_fit


# ---------------------------------------------------------------- 主解码入口


@dataclass(frozen=True)
class DecodeResult:
    """解码结果。

    Attributes:
        start: 帧起点（前导第一样本）下标。
        frequency_offset: 频偏估计，单位为载波间隔，满足 |f| < 0.5。
        residual: 最优候选的前导补偿残差，[0, 1]，越小越好。
        gain: 四个导频观测最小二乘得到的单一复增益（历史字段）。
            平坦信道下即真实信道增益；多径信道下只是导频载波处的
            标量参考，真实均衡依据 ``channel``。
        symbols: 24 个判决结果，顺序为符号 0→1，每符号载波
            -6..-1、+1..+6；正常为 ±(1±j)/√2，深零点载波为 ``None``（擦除）。
        bits: 48 字符恢复位串（每符号 2 位，先实部后虚部，正为 0）；
            被擦除符号对应的两位为 ``'x'``。
        channel: 训练符号估计出的 3 抽头 FIR 信道（含频响与擦除清单）。
        erased_carriers: 被判深零点的有符号数据载波（来自 ``channel``）。
        erased_positions: 24 个数据符号中被擦除的位置索引（来自 ``channel``）。
    """

    start: int
    frequency_offset: float
    residual: float
    gain: complex
    symbols: list
    bits: str
    channel: ChannelEstimate
    erased_carriers: tuple = ()
    erased_positions: tuple = ()


def _validate_positive(value, name: str) -> float:
    _reject_bool(value, name)
    if not isinstance(value, (int, float)):
        raise ParameterError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ParameterError(f"{name} must be a finite positive number")
    return value


def decode(
    signal,
    *,
    rho_threshold: float = DEFAULT_RHO_THRESHOLD,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
    erasure_threshold: float = DEFAULT_ERASURE_THRESHOLD,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> DecodeResult:
    """离线恢复一帧（支持最多 3 抽头固定复数 FIR 信道）。

    失败时抛出 ``OFDMError`` 的具体子类，不返回半结果。
    ``erasure_threshold`` 为深零点相对门限：数据载波 k 满足
    ``|H(k)| < erasure_threshold·rms(H)`` 时该载波判擦除（``None``/``'x'``），
    不做硬判决；其余载波 ZF 均衡后判 QPSK。
    """
    sig = _validate_signal(signal)
    res_thr = _validate_threshold(residual_threshold, "residual_threshold")
    era_thr = _validate_positive(erasure_threshold, "erasure_threshold")
    # find_candidates 内部会校验其余参数与信号。
    candidates = find_candidates(
        sig, rho_threshold=rho_threshold, max_candidates=max_candidates
    )
    if not candidates:
        raise NoFrameError(
            f"exhaustively scanned all {len(sig) - PREAMBLE_LEN + 1} feasible "
            f"start positions; none reached rho_threshold={rho_threshold} "
            "(proven absent under this contract, not a budget limit)"
        )

# 候选终审的拟合残差等价带：J ≤ max(相对因子·J_min, 绝对下限)。
# 物理上同一波形存在定时等价类：
#   起点 p、抽头 (h0,h1,0)  ==  起点 p-1、抽头 (0,h0,h1)
#   起点 p、抽头 (h0,0,0)   ==  起点 p-1、(0,h0,0) == 起点 p-2、(0,0,h0)
# （3 抽头的 h2 项无法被移位吸收，故真起点对 3 抽头唯一。）
# 规范形式取“首抽头非零”的表示（等价类中最晚起点）：
# 带内按 |h0|/rms 降序，再以前导残差、起点兜底。
_FIT_BAND_REL = 1.1
_FIT_BAND_ABS = 1e-6


def decode(
    signal,
    *,
    rho_threshold: float = DEFAULT_RHO_THRESHOLD,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
    erasure_threshold: float = DEFAULT_ERASURE_THRESHOLD,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> DecodeResult:
    """离线恢复一帧（支持最多 3 抽头固定复数 FIR 信道）。

    失败时抛出 ``OFDMError`` 的具体子类，不返回半结果。
    ``erasure_threshold`` 为深零点相对门限：数据载波 k 满足
    ``|H(k)| < erasure_threshold·rms(H)`` 时该载波判擦除（``None``/``'x'``），
    不做硬判决；其余载波 ZF 均衡后判 QPSK。
    """
    sig = _validate_signal(signal)
    res_thr = _validate_threshold(residual_threshold, "residual_threshold")
    era_thr = _validate_positive(erasure_threshold, "erasure_threshold")
    # find_candidates 内部会校验其余参数与信号。
    candidates = find_candidates(
        sig, rho_threshold=rho_threshold, max_candidates=max_candidates
    )
    if not candidates:
        raise NoFrameError(
            f"exhaustively scanned all {len(sig) - PREAMBLE_LEN + 1} feasible "
            f"start positions; none reached rho_threshold={rho_threshold} "
            "(proven absent under this contract, not a budget limit)"
        )

    pool = [c for c in candidates if c.residual <= res_thr]
    if not pool:
        best = candidates[0]
        raise FrameQualityError(
            f"best candidate at start {best.start} has preamble residual "
            f"{best.residual:.6f} > residual_threshold={res_thr}"
        )

    # 终审：每个候选用稳态频偏补偿后做 3 抽头训练拟合，
    # J 等价带内按 (全窗口前导残差, 起点) 裁决。
    scored = []
    for cand in pool:
        f_steady = _steady_cfo(sig, cand.start)
        taps, j_fit = _fit_channel(sig, cand.start, f_steady)
        scored.append((cand, f_steady, taps, j_fit))
    j_min = min(item[3] for item in scored)
    band_limit = max(_FIT_BAND_REL * j_min, _FIT_BAND_ABS)
    band = [item for item in scored if item[3] <= band_limit]

    def _canonical_key(item):
        cand, _, taps, _ = item
        rms = math.sqrt(sum(abs(t) ** 2 for t in taps)) or 1.0
        # 规范形式：首抽头占比最大（等价类最晚起点），
        # 再按前导残差、起点兜底（与既有平局裁决一致）。
        return (-abs(taps[0]) / rms, cand.residual, cand.start)

    best_cand, f_offset, taps, j_fit = min(band, key=_canonical_key)

    if abs(f_offset) >= CFO_LIMIT:
        raise FrequencyOffsetError(
            f"estimated CFO at start {best_cand.start} is "
            f"{f_offset:.6f} carrier spacings; contract requires "
            f"|f| < {CFO_LIMIT}"
        )

    end = best_cand.start + FRAME_LEN
    if end > len(sig):
        raise TruncatedFrameError(
            f"frame starts at {best_cand.start} and needs {FRAME_LEN} samples "
            f"(through index {end - 1}), but signal has only {len(sig)} "
            "samples"
        )

    # 全帧频偏补偿（时间原点取帧起点；绝对相位被信道抽头吸收）。
    omega = -2.0 * math.pi * f_offset / N_FFT
    seg = [
        sig[best_cand.start + n] * cmath.exp(1j * omega * n)
        for n in range(FRAME_LEN)
    ]

    response = tuple(
        sum(taps[l] * cmath.exp(-2j * math.pi * k * l / N_FFT) for l in range(3))
        for k in range(N_FFT)
    )
    rms_gain = math.sqrt(sum(abs(t) ** 2 for t in taps))

    # 去 CP 并自行 DFT。
    spectra = []
    for s in range(NUM_OFDM_SYMBOLS):
        base = PREAMBLE_LEN + s * OFDM_SYMBOL_LEN + CP_LEN
        spectra.append(dft(seg[base:base + N_FFT]))

    # 历史字段：四个导频观测的最小二乘单一复增益 h = ΣY·conj(p)/Σ|p|²。
    gain_num = 0j
    gain_den = 0.0
    pilot_obs_energy = 0.0
    pilot_model_energy = 0.0
    for yk in spectra:
        for k, p in PILOT_VALUES.items():
            v = yk[k]
            gain_num += v * p.conjugate()
            gain_den += abs(p) ** 2
            pilot_obs_energy += abs(v) ** 2
            # 循环卷积下导频观测的模型值为 H(k)·p。
            pilot_model_energy += abs(response[k] * p) ** 2

    # 发射端把导频置零时（见参考合成器 zero_pilots），观测能量远小于
    # 信道模型预测；若模型本身在导频处也是零点，则属合法深零点，不拒绝。
    if (
        pilot_model_energy > 0.0
        and pilot_obs_energy < _PILOT_REL_FLOOR * pilot_model_energy
    ):
        raise ZeroPilotEnergyError(
            "pilot carriers (+/-7) carry no usable energy although the channel "
            f"model predicts energy (ratio floor {_PILOT_REL_FLOOR:g}); "
            "frame was transmitted without pilots"
        )
    gain = gain_num / gain_den if gain_den > 0.0 else 0j

    erased_carriers = tuple(
        kc
        for kc in SIGNED_DATA_CARRIERS
        if abs(response[kc % N_FFT]) < era_thr * rms_gain
    )
    erased_set = set(erased_carriers)
    # 固定信道下两个 OFDM 符号的擦除载波相同；展开为 24 个符号位置索引。
    erased_positions = tuple(
        s * len(SIGNED_DATA_CARRIERS) + j
        for s in range(NUM_OFDM_SYMBOLS)
        for j, kc in enumerate(SIGNED_DATA_CARRIERS)
        if kc in erased_set
    )

    symbols: list = []
    for yk in spectra:
        for kc in SIGNED_DATA_CARRIERS:
            k = kc % N_FFT
            if kc in erased_set:
                symbols.append(None)
            else:
                symbols.append(_slice_qpsk(yk[k] / response[k]))

    channel = ChannelEstimate(
        taps=tuple(taps),
        rms_gain=rms_gain,
        frequency_response=response,
        erased_carriers=erased_carriers,
        fit_residual=j_fit,
    )

    return DecodeResult(
        start=best_cand.start,
        frequency_offset=f_offset,
        residual=best_cand.residual,
        gain=gain,
        symbols=symbols,
        bits=symbols_to_bits(symbols),
        channel=channel,
        erased_carriers=erased_carriers,
        erased_positions=erased_positions,
    )
