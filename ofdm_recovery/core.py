"""OFDM 循环前缀边界与子载波恢复（本题契约子集）。

固定参数：N=32 点 FFT，CP=8，前导为两个相同的 16 样本块，随后两个 OFDM 符号。
子载波布局（DFT 索引 k=0..31）：

* 数据载波：k = 1..6 与 26..31（即 ±1..±6），QPSK；
* 导频载波：k=7（已知 +1）、k=25 即 -7（已知 -1）；
* DC（k=0）与 k=8..24 为 0。

本模块只实现题目定义的离线合成帧子集，不是完整行业标准接收机。
核心算法（相关、频偏估计、DFT、导频信道估计、判决）均在本文件内实现，
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

# ---------------------------------------------------------------- 星座 Gray 表
#
# 显式 Gray 映射表（题内约定，不声称兼容任何行业标准）：
# * QPSK：每符号 2 位 (b0, b1)，b0 对应实部、b1 对应虚部，位 0 -> 正坐标。
# * 16QAM：每符号 4 位 (b0, b1, b2, b3)，(b0, b1) 映射实部、(b2, b3) 映射
#   虚部，每轴 2 位按 Gray 00->+3, 01->+1, 11->-1, 10->-3，电平除以 sqrt(10)
#   使平均能量为 1。

QPSK_GRAY = (
    ((0, 0), complex(_QPSK, _QPSK)),
    ((0, 1), complex(_QPSK, -_QPSK)),
    ((1, 1), complex(-_QPSK, -_QPSK)),
    ((1, 0), complex(-_QPSK, _QPSK)),
)

_QAM16_AXIS_GRAY = (
    ((0, 0), 3.0),
    ((0, 1), 1.0),
    ((1, 1), -1.0),
    ((1, 0), -3.0),
)
_QAM16_NORM = 1.0 / math.sqrt(10.0)
QAM16_GRAY = tuple(
    (
        (b0, b1, b2, b3),
        complex(re_level, im_level) * _QAM16_NORM,
    )
    for (b0, b1), re_level in _QAM16_AXIS_GRAY
    for (b2, b3), im_level in _QAM16_AXIS_GRAY
)

_CONSTELLATIONS = {"qpsk": QPSK_GRAY, "qam16": QAM16_GRAY}


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

# ---------------------------------------------------------------- 软解调


def _validate_noise_var(value) -> float:
    """复噪声方差：必须是正的有限实数（拒绝 bool/0/负/NaN/Inf）。"""
    _reject_bool(value, "noise_var")
    if not isinstance(value, (int, float)):
        raise ParameterError("noise_var must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ParameterError(f"noise_var must be finite, got {value}")
    if value <= 0.0:
        raise ParameterError(f"noise_var must be positive, got {value}")
    return value


def _constellation(modulation) -> tuple:
    if isinstance(modulation, bool) or not isinstance(modulation, str):
        raise ParameterError("modulation must be 'qpsk' or 'qam16'")
    key = modulation.lower()
    if key not in _CONSTELLATIONS:
        raise ParameterError(
            f"unknown modulation {modulation!r}; expected 'qpsk' or 'qam16'"
        )
    return _CONSTELLATIONS[key]


def _log_sum_exp(values) -> float:
    """数值稳定的 log(Σ exp(v))，用 math.fsum 做高精度求和。"""
    vals = list(values)
    m = max(vals)
    return m + math.log(math.fsum(math.exp(v - m) for v in vals))


def soft_demod(symbol, noise_var, *, modulation: str = "qpsk") -> list[float]:
    """对单个已均衡符号做软解调，返回每位的对数似然比（LLR）。

    LLR 约定为 log P(bit=0 | y) - log P(bit=1 | y)，**正值偏向位 0**。
    两组星座点概率用 log-sum-exp 计算：
    LLR_i = lse_{s: b_i=0}(-|y-s|²/σ²) - lse_{s: b_i=1}(-|y-s|²/σ²)，
    其中 σ² = noise_var 为复噪声方差（必须为正）。
    位序与星座表一致：QPSK 2 位（先实部后虚部），16QAM 4 位。
    """
    table = _constellation(modulation)
    sigma2 = _validate_noise_var(noise_var)
    y = _finite_complex(symbol, "symbol")
    loglik = [
        -((y.real - s.real) ** 2 + (y.imag - s.imag) ** 2) / sigma2
        for _, s in table
    ]
    nbits = len(table[0][0])
    llrs = []
    for i in range(nbits):
        l0 = _log_sum_exp(
            ll for (bits, _), ll in zip(table, loglik) if bits[i] == 0
        )
        l1 = _log_sum_exp(
            ll for (bits, _), ll in zip(table, loglik) if bits[i] == 1
        )
        llrs.append(l0 - l1)
    return llrs


def soft_demod_equalized(
    symbol, gain, noise_var, *, modulation: str = "qpsk"
) -> tuple[list[float], bool]:
    """带信道增益的软解调：返回 (llrs, erased)。

    先用 gain 均衡符号，噪声方差按增益同步调整为
    noise_var / |gain|²（深衰落时等效噪声放大，LLR 自然趋近 0）。
    gain == 0 视为**擦除**：返回全 0 LLR 且 erased=True，
    不做硬判成全 0 位。
    """
    table = _constellation(modulation)
    sigma2 = _validate_noise_var(noise_var)
    g = _finite_complex(gain, "gain")
    y = _finite_complex(symbol, "symbol")
    nbits = len(table[0][0])
    if g == 0j:
        return [0.0] * nbits, True
    eq = y / g
    return soft_demod(eq, sigma2 / (g.real * g.real + g.imag * g.imag),
                      modulation=modulation), False



# ---------------------------------------------------------------- 主解码入口


@dataclass(frozen=True)
class DecodeResult:
    """解码结果。

    Attributes:
        start: 帧起点（前导第一样本）下标。
        frequency_offset: 频偏估计，单位为载波间隔，满足 |f| < 0.5。
        residual: 最优候选的前导补偿残差，[0, 1]，越小越好。
        gain: 四个导频观测最小二乘得到的单一复增益。
        symbols: 24 个 QPSK 判决符号，顺序为符号 0→1，
            每符号载波 -6..-1、+1..+6；取值 ±(1±j)/√2。
        bits: 48 位恢复位串（每符号 2 位，先实部后虚部，正为 0）。
    """

    start: int
    frequency_offset: float
    residual: float
    gain: complex
    symbols: list[complex]
    bits: str


def decode(
    signal,
    *,
    rho_threshold: float = DEFAULT_RHO_THRESHOLD,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> DecodeResult:
    """离线恢复一帧。失败时抛出 ``OFDMError`` 的具体子类，不返回半结果。"""
    best, gain, eq_symbols = _recover_frame(
        signal,
        rho_threshold=rho_threshold,
        residual_threshold=residual_threshold,
        max_candidates=max_candidates,
    )
    symbols = [_slice_qpsk(v) for v in eq_symbols]
    return DecodeResult(
        start=best.start,
        frequency_offset=best.frequency_offset,
        residual=best.residual,
        gain=gain,
        symbols=symbols,
        bits=symbols_to_bits(symbols),
    )


def _recover_frame(
    signal,
    *,
    rho_threshold: float,
    residual_threshold: float,
    max_candidates: int,
):
    """共享接收管线：定位、补偿、DFT、信道估计，返回
    ``(best_candidate, gain, 均衡后的 24 个数据载波观测)``。

    只保留数据载波（±1..±6）；导频、DC 与保护带不进入符号序列。
    """
    sig = _validate_signal(signal)
    res_thr = _validate_threshold(residual_threshold, "residual_threshold")
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

    best = candidates[0]

    if abs(best.frequency_offset) >= CFO_LIMIT:
        raise FrequencyOffsetError(
            f"estimated CFO at start {best.start} is "
            f"{best.frequency_offset:.6f} carrier spacings; contract requires "
            f"|f| < {CFO_LIMIT}"
        )
    if best.residual > res_thr:
        raise FrameQualityError(
            f"best candidate at start {best.start} has preamble residual "
            f"{best.residual:.6f} > residual_threshold={res_thr}"
        )

    end = best.start + FRAME_LEN
    if end > len(sig):
        raise TruncatedFrameError(
            f"frame starts at {best.start} and needs {FRAME_LEN} samples "
            f"(through index {end - 1}), but signal has only {len(sig)} "
            "samples"
        )

    # 全帧频偏补偿（时间原点取帧起点；绝对相位被导频增益吸收）。
    omega = -2.0 * math.pi * best.frequency_offset / N_FFT
    seg = [
        sig[best.start + n] * cmath.exp(1j * omega * n)
        for n in range(FRAME_LEN)
    ]

    # 去 CP 并自行 DFT。
    spectra = []
    for s in range(NUM_OFDM_SYMBOLS):
        base = PREAMBLE_LEN + s * OFDM_SYMBOL_LEN + CP_LEN
        spectra.append(dft(seg[base:base + N_FFT]))

    # 单一复增益：四个导频观测的最小二乘 h = ΣY·conj(p) / Σ|p|²。
    num = 0j
    den = 0.0
    pilot_energy = 0.0
    total_energy = 0.0
    for yk in spectra:
        for v in yk:
            total_energy += v.real * v.real + v.imag * v.imag
        for k, p in PILOT_VALUES.items():
            v = yk[k]
            num += v * p.conjugate()
            den += p.real * p.real + p.imag * p.imag
            pilot_energy += v.real * v.real + v.imag * v.imag

    if total_energy <= 0.0 or pilot_energy < _PILOT_REL_FLOOR * total_energy:
        raise ZeroPilotEnergyError(
            "pilot carriers (+/-7) carry no usable energy "
            f"(pilot/total energy ratio floor {_PILOT_REL_FLOOR:g}); "
            "complex gain cannot be estimated"
        )
    gain = num / den

    eq_symbols: list[complex] = []
    for yk in spectra:
        for kc in SIGNED_DATA_CARRIERS:
            k = kc % N_FFT
            eq_symbols.append(yk[k] / gain)

    return best, gain, eq_symbols


# ---------------------------------------------------------------- 软解调帧入口


@dataclass(frozen=True)
class SoftDecodeResult:
    """软解调结果（硬判决字段与 ``DecodeResult`` 完全同序，保持原位序）。

    Attributes:
        start / frequency_offset / residual / gain / symbols / bits:
            与 ``DecodeResult`` 相同的硬判决输出。
        modulation: 使用的星座（'qpsk' 或 'qam16'）。
        noise_var: 调用方给定的复噪声方差（正数）。
        llrs: 展平的每位 LLR（正值偏向位 0），QPSK 48 个、16QAM 96 个，
            顺序与 ``bits`` 的数据位序列一致（只含数据载波，
            导频/DC/保护带不混入）。
        erasures: 24 个布尔，标记对应符号是否为零增益擦除
            （擦除位的 LLR 恒为 0.0，不硬判成全 0）。
    """

    start: int
    frequency_offset: float
    residual: float
    gain: complex
    symbols: list[complex]
    bits: str
    modulation: str
    noise_var: float
    llrs: list[float]
    erasures: list[bool]


def decode_soft(
    signal,
    noise_var,
    *,
    modulation: str = "qpsk",
    rho_threshold: float = DEFAULT_RHO_THRESHOLD,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> SoftDecodeResult:
    """在原符号恢复管线之后追加软解调，返回每位 LLR 与擦除标志。

    信道补偿后的噪声方差按增益同步调整为 ``noise_var / |gain|²``；
    增益为零的载波按擦除处理（LLR=0 且置擦除标志）。
    """
    table = _constellation(modulation)
    sigma2 = _validate_noise_var(noise_var)
    best, gain, eq_symbols = _recover_frame(
        signal,
        rho_threshold=rho_threshold,
        residual_threshold=residual_threshold,
        max_candidates=max_candidates,
    )
    symbols = [_slice_qpsk(v) for v in eq_symbols]
    llrs: list[float] = []
    erasures: list[bool] = []
    for v in eq_symbols:
        # 与 soft_demod_equalized 同一定义：零增益 -> 擦除。
        if gain == 0j:  # 防御性：正常流程零增益已在导频检查处拒绝
            llrs.extend([0.0] * len(table[0][0]))
            erasures.append(True)
            continue
        llrs.extend(
            soft_demod(v, sigma2 / (abs(gain) ** 2), modulation=modulation)
        )
        erasures.append(False)
    return SoftDecodeResult(
        start=best.start,
        frequency_offset=best.frequency_offset,
        residual=best.residual,
        gain=gain,
        symbols=symbols,
        bits=symbols_to_bits(symbols),
        modulation=modulation.lower(),
        noise_var=sigma2,
        llrs=llrs,
        erasures=erasures,
    )
