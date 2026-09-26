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
        llrs: 调用方提供 noise_variance 时，48 个数据比特的软解调 LLR
            （正值偏向比特 0，位序与 bits 一致）；否则为 None。
        erasures: 与 llrs 对应的 24 个数据符号擦除标志；未软解调时为 None。
    """

    start: int
    frequency_offset: float
    residual: float
    gain: complex
    symbols: list[complex]
    bits: str
    llrs: tuple | None = None
    erasures: tuple | None = None


def decode(
    signal,
    *,
    rho_threshold: float = DEFAULT_RHO_THRESHOLD,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    noise_variance=None,
) -> DecodeResult:
    """离线恢复一帧。失败时抛出 ``OFDMError`` 的具体子类，不返回半结果。

    给出正的复噪声方差 ``noise_variance`` 时，在硬判决之外对每个数据
    比特追加 QPSK 软解调 LLR（正值偏向比特 0）；导频、DC 与保护带
    不进入数据位序列。信道补偿后噪声方差按 σ²/|gain|² 同步调整；
    增益为零时全部数据符号按擦除处理（LLR=0 并置擦除标志）。
    """
    sig = _validate_signal(signal)
    res_thr = _validate_threshold(residual_threshold, "residual_threshold")
    if noise_variance is not None:
        noise_variance = _validate_noise_variance(noise_variance)
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

    observations: list[complex] = []
    symbols: list[complex] = []
    for yk in spectra:
        for kc in SIGNED_DATA_CARRIERS:
            k = kc % N_FFT
            observations.append(yk[k])
            symbols.append(_slice_qpsk(yk[k] / gain))

    llrs = erasures = None
    if noise_variance is not None:
        soft = soft_demodulate(
            observations,
            modulation="qpsk",
            noise_variance=noise_variance,
            gains=[gain] * len(observations),
        )
        llrs, erasures = soft.llrs, soft.erasures

    return DecodeResult(
        start=best.start,
        frequency_offset=best.frequency_offset,
        residual=best.residual,
        gain=gain,
        symbols=symbols,
        bits=symbols_to_bits(symbols),
        llrs=llrs,
        erasures=erasures,
    )

# ---------------------------------------------------------------- 软解调

# 题内显式 Gray 星座表：位组 -> 星座点。LLR 约定正值偏向比特 0。
# QPSK：b0 对应实部、b1 对应虚部，正坐标 -> 0，负坐标 -> 1（与硬判决一致）。
QPSK_GRAY = {
    (0, 0): complex(_QPSK, _QPSK),
    (0, 1): complex(_QPSK, -_QPSK),
    (1, 0): complex(-_QPSK, _QPSK),
    (1, 1): complex(-_QPSK, -_QPSK),
}

# 16QAM：电平 ±1、±3 按 1/√10 归一化；(b0,b1) 映射实部、(b2,b3) 映射虚部，
# 沿每根轴为 Gray 序 00->-3, 01->-1, 11->+1, 10->+3。
_QAM16_SCALE = 1.0 / math.sqrt(10.0)
QAM16_GRAY = {
    (0, 0, 0, 0): complex(-3, -3) * _QAM16_SCALE,
    (0, 0, 0, 1): complex(-3, -1) * _QAM16_SCALE,
    (0, 0, 1, 1): complex(-3, 1) * _QAM16_SCALE,
    (0, 0, 1, 0): complex(-3, 3) * _QAM16_SCALE,
    (0, 1, 0, 0): complex(-1, -3) * _QAM16_SCALE,
    (0, 1, 0, 1): complex(-1, -1) * _QAM16_SCALE,
    (0, 1, 1, 1): complex(-1, 1) * _QAM16_SCALE,
    (0, 1, 1, 0): complex(-1, 3) * _QAM16_SCALE,
    (1, 1, 0, 0): complex(1, -3) * _QAM16_SCALE,
    (1, 1, 0, 1): complex(1, -1) * _QAM16_SCALE,
    (1, 1, 1, 1): complex(1, 1) * _QAM16_SCALE,
    (1, 1, 1, 0): complex(1, 3) * _QAM16_SCALE,
    (1, 0, 0, 0): complex(3, -3) * _QAM16_SCALE,
    (1, 0, 0, 1): complex(3, -1) * _QAM16_SCALE,
    (1, 0, 1, 1): complex(3, 1) * _QAM16_SCALE,
    (1, 0, 1, 0): complex(3, 3) * _QAM16_SCALE,
}

_CONSTELLATIONS = {"qpsk": QPSK_GRAY, "16qam": QAM16_GRAY}


@dataclass(frozen=True)
class SoftDemodResult:
    """软解调结果。

    Attributes:
        modulation: "qpsk" 或 "16qam"。
        llrs: 每个比特的对数似然比 log P(b=0)/P(b=1)，正值偏向比特 0；
            被擦除的符号对应位恒为 0.0。
        erasures: 每个符号是否被擦除（擦除符号的全部位 LLR 为 0）。
    """

    modulation: str
    llrs: tuple
    erasures: tuple


def _validate_noise_variance(value) -> float:
    _reject_bool(value, "noise_variance")
    if not isinstance(value, (int, float)):
        raise ParameterError("noise_variance must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ParameterError(f"noise_variance must be finite, got {value}")
    if value <= 0.0:
        raise ParameterError(
            f"noise_variance must be positive, got {value}"
        )
    return value


def _logsumexp(terms) -> float:
    """稳定的 log(Σ exp(t))；terms 为空时返回 -inf。"""
    peak = max(terms)
    if peak == -math.inf:
        return -math.inf
    return peak + math.log(sum(math.exp(t - peak) for t in terms))


def soft_demodulate(
    symbols,
    *,
    modulation: str = "qpsk",
    noise_variance,
    gains=None,
    erased=None,
) -> SoftDemodResult:
    """对星座观测序列做软解调，输出每比特 LLR（正值偏向比特 0）。

    symbols: 复数观测序列（若给出 gains，则为信道补偿**前**的观测 y=h·s+n）。
    modulation: "qpsk" 或 "16qam"，使用题内显式 Gray 表。
    noise_variance: 正的复噪声方差 σ² = E|n|²（调用方提供）。
    gains: 可选的每符号复增益序列；补偿后噪声方差按 σ²/|h|² 调整，
        零增益符号按擦除处理。
    erased: 可选的每符号布尔序列，True 表示该符号被擦除。

    擦除符号的全部比特返回 LLR=0.0 并在 erasures 中置位，不做硬判。
    每个比特用 log-sum-exp 分别累加两组星座点的对数概率。
    """
    if isinstance(modulation, bool) or not isinstance(modulation, str):
        raise ParameterError("modulation must be the string 'qpsk' or '16qam'")
    table = _CONSTELLATIONS.get(modulation.lower())
    if table is None:
        raise ParameterError(
            f"unknown modulation {modulation!r}; expected 'qpsk' or '16qam'"
        )
    sigma2 = _validate_noise_variance(noise_variance)
    if not isinstance(symbols, (list, tuple)):
        raise SignalFormatError("symbols must be a list/tuple")
    obs = [_finite_complex(v, f"symbols[{i}]") for i, v in enumerate(symbols)]
    count = len(obs)

    if gains is None:
        gain_list = [None] * count
    else:
        if not isinstance(gains, (list, tuple)) or len(gains) != count:
            raise ParameterError(
                "gains must be a list/tuple with one complex gain per symbol"
            )
        gain_list = [
            _finite_complex(g, f"gains[{i}]") for i, g in enumerate(gains)
        ]

    if erased is None:
        erased_list = [False] * count
    else:
        if not isinstance(erased, (list, tuple)) or len(erased) != count:
            raise ParameterError(
                "erased must be a list/tuple with one bool per symbol"
            )
        erased_list = []
        for i, e in enumerate(erased):
            if not isinstance(e, bool):
                raise ParameterError(f"erased[{i}] must be a bool")
            erased_list.append(e)

    bits_per_symbol = len(next(iter(table)))
    by_bit = [([], []) for _ in range(bits_per_symbol)]
    for bits, point in table.items():
        for b in range(bits_per_symbol):
            by_bit[b][bits[b]].append(point)

    llrs: list[float] = []
    erasures: list[bool] = []
    for i, y in enumerate(obs):
        h = gain_list[i]
        if h is None:
            z, var = y, sigma2
        elif h == 0j:
            z, var = None, None  # 零增益 -> 擦除
        else:
            z, var = y / h, sigma2 / (h.real * h.real + h.imag * h.imag)
        is_erased = erased_list[i] or z is None
        erasures.append(is_erased)
        if is_erased:
            llrs.extend([0.0] * bits_per_symbol)
            continue
        for b in range(bits_per_symbol):
            zeros, ones = by_bit[b]
            llr = _logsumexp(
                [-((z - s).real ** 2 + (z - s).imag ** 2) / var for s in zeros]
            ) - _logsumexp(
                [-((z - s).real ** 2 + (z - s).imag ** 2) / var for s in ones]
            )
            llrs.append(llr)
    return SoftDemodResult(modulation.lower(), tuple(llrs), tuple(erasures))
