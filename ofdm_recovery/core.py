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

import bisect
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
    """

    start: int
    frequency_offset: float
    residual: float
    gain: complex
    symbols: list[complex]
    bits: str


def _frame_pilot_metrics(
    sig: list[complex], start: int, f_offset: float
) -> tuple[complex, float, list[list[complex]]]:
    """对已知起点/频偏的**完整**帧做补偿、去 CP、DFT 与导频最小二乘。

    返回 ``(gain, pilot_residual, spectra)``。``pilot_residual`` 为四个导频
    观测的归一化最小二乘剩余能量 ``1 - |ΣY·conj(p)|² / (E_pilot·Σ|p|²)``，
    取值 [0, 1]，越小越好；导频观测相互抵消时 ``gain`` 可以为 0（此时
    ``pilot_residual`` 为 1），由调用方决定如何处理。导频能量低于相对
    底限时抛 ``ZeroPilotEnergyError``。
    调用方须保证 ``start + FRAME_LEN <= len(sig)``。
    """
    # 全帧频偏补偿（时间原点取帧起点；绝对相位被导频增益吸收）。
    omega = -2.0 * math.pi * f_offset / N_FFT
    seg = [
        sig[start + n] * cmath.exp(1j * omega * n)
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
    # 最小二乘剩余能量 = E_pilot - |num|²/den（理论非负，浮点下钳到 0）。
    pilot_residual = max(
        0.0, (pilot_energy - (num.real * num.real + num.imag * num.imag) / den)
        / pilot_energy
    )
    return gain, pilot_residual, spectra


def _symbols_from_spectra(
    spectra: list[list[complex]], gain: complex
) -> list[complex]:
    """逐数据载波 ``Y/gain`` 后硬判决（调用方保证 gain 非零）。"""
    symbols: list[complex] = []
    for yk in spectra:
        for kc in SIGNED_DATA_CARRIERS:
            k = kc % N_FFT
            symbols.append(_slice_qpsk(yk[k] / gain))
    return symbols


def _decode_frame_at(
    sig: list[complex], start: int, f_offset: float
) -> tuple[complex, list[complex], float]:
    """对已知起点/频偏的**完整**帧做补偿、去 CP、DFT、导频最小二乘与判决。

    返回 ``(gain, symbols, pilot_residual)``。导频能量不足、或导频观测
    相互抵消导致增益估计恰为 0（无法归一化数据载波）时，抛
    ``ZeroPilotEnergyError``。调用方须保证
    ``start + FRAME_LEN <= len(sig)``。
    """
    gain, pilot_residual, spectra = _frame_pilot_metrics(sig, start, f_offset)
    if gain == 0:
        raise ZeroPilotEnergyError(
            "pilot least-squares gain estimate is exactly zero "
            "(pilot observations cancel); data carriers cannot be normalized"
        )
    return gain, _symbols_from_spectra(spectra, gain), pilot_residual


def decode(
    signal,
    *,
    rho_threshold: float = DEFAULT_RHO_THRESHOLD,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> DecodeResult:
    """离线恢复一帧。失败时抛出 ``OFDMError`` 的具体子类，不返回半结果。"""
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

    gain, symbols, _pilot_residual = _decode_frame_at(
        sig, best.start, best.frequency_offset
    )

    return DecodeResult(
        start=best.start,
        frequency_offset=best.frequency_offset,
        residual=best.residual,
        gain=gain,
        symbols=symbols,
        bits=symbols_to_bits(symbols),
    )


# ---------------------------------------------------------------- 多帧扫描入口

# RejectedCandidate.reason 的取值（字符串常量，便于比较与打印）。
REASON_CFO_OUT_OF_RANGE = "cfo_out_of_range"
REASON_PREAMBLE_RESIDUAL = "preamble_residual"
REASON_ZERO_PILOT_ENERGY = "zero_pilot_energy"
REASON_PILOT_RESIDUAL = "pilot_residual"


@dataclass(frozen=True)
class ScannedFrame:
    """多帧扫描中一帧被选中并完整解码的帧。

    Attributes:
        start: 帧起点（前导第一样本）在输入缓冲区中的绝对下标。
        frequency_offset: 该帧独立的频偏估计（载波间隔），|f| < 0.5。
        residual: 前导补偿残差，[0, 1]，越小越好。
        pilot_residual: 导频最小二乘归一化剩余能量，[0, 1]，越小越好。
        gain: 四个导频观测最小二乘得到的单一复增益。
        symbols: 24 个 QPSK 判决符号（顺序同 DecodeResult）。
        bits: 48 位恢复位串。
    """

    start: int
    frequency_offset: float
    residual: float
    pilot_residual: float
    gain: complex
    symbols: list[complex]
    bits: str


@dataclass(frozen=True)
class RejectedCandidate:
    """质量检查未通过而被拒绝的候选（``reason`` 取 REASON_* 常量之一）。"""

    start: int
    frequency_offset: float
    residual: float
    coherence: float
    reason: str


@dataclass(frozen=True)
class TruncatedCandidate:
    """前导与频偏检查通过、但帧在缓冲区末尾前不完整的候选。

    只登记、不解码、不补零：``available`` 为从 ``start`` 起缓冲区实际
    可用的样本数（< FRAME_LEN）。
    """

    start: int
    frequency_offset: float
    residual: float
    coherence: float
    available: int


@dataclass(frozen=True)
class ScanResult:
    """多帧扫描结果。

    Attributes:
        frames: 选中的互不重叠完整帧，按起点升序。
        rejected: 未通过质量检查的候选（含原因），按起点升序。
        truncated: 尾部不完整帧（未解码、未补零），按起点升序。
        not_selected: 质量合格但因与更优选择重叠而未被选中的候选，
            按起点升序。
    """

    frames: list[ScannedFrame]
    rejected: list[RejectedCandidate]
    truncated: list[TruncatedCandidate]
    not_selected: list[Candidate]


@dataclass(frozen=True)
class _Accepted:
    """通过全部质量检查、等待重叠选择的候选（内部类型）。"""

    candidate: Candidate
    gain: complex
    symbols: list[complex]
    pilot_residual: float


def _select_non_overlapping(
    accepted: list[_Accepted],
) -> tuple[list[_Accepted], list[Candidate]]:
    """区间 DP 选出互不重叠的最优子集。

    帧区间均为 ``[start, start + FRAME_LEN)``。选择顺序：
    1. 完整可解帧数最多；2. 前导残差总和最小；3. 起点序列字典序最小。
    返回 ``(选中的 _Accepted 列表（按起点升序), 未选中的 Candidate 列表)``。
    """
    items = sorted(accepted, key=lambda a: a.candidate.start)
    n = len(items)
    starts = [a.candidate.start for a in items]
    # nxt[i] = 第一个与 items[i] 不重叠的下标（起点 >= start_i + FRAME_LEN）。
    nxt = [bisect.bisect_left(starts, s + FRAME_LEN) for s in starts]

    # dp[i] = (帧数, 负残差总和, 起点元组)：仅使用 items[i:] 的最优选择。
    dp: list[tuple[int, float, tuple[int, ...]]] = [(0, 0.0, ())] * (n + 1)
    took = [False] * n
    for i in range(n - 1, -1, -1):
        skip = dp[i + 1]
        tc, tr, ts = dp[nxt[i]]
        take = (tc + 1, tr - items[i].candidate.residual, (starts[i],) + ts)
        # 先比帧数与残差总和（大者优），并列时起点序列字典序小者优。
        if (take[0], take[1]) > (skip[0], skip[1]) or (
            (take[0], take[1]) == (skip[0], skip[1]) and take[2] < skip[2]
        ):
            dp[i] = take
            took[i] = True
        else:
            dp[i] = skip

    chosen: list[_Accepted] = []
    chosen_ids: set[int] = set()
    i = 0
    while i < n:
        if took[i]:
            chosen.append(items[i])
            chosen_ids.add(i)
            i = nxt[i]
        else:
            i += 1
    rest = [
        items[i].candidate for i in range(n) if i not in chosen_ids
    ]
    return chosen, rest


def scan_frames(
    signal,
    *,
    rho_threshold: float = DEFAULT_RHO_THRESHOLD,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> ScanResult:
    """在有限缓冲区（<= 512 样本）内一次扫描并恢复多帧。

    流程：按既有前导与固定帧长枚举全部候选；对每个候选**独立**求频偏、
    前导残差、导频残差与完整性；残差超过 ``residual_threshold`` 的候选
    拒绝并记录原因；尾部不完整帧另列（不补零、不当完整帧）；最后对合格
    候选用区间 DP 选出互不重叠的一组（帧数最多 → 残差总和最小 → 起点
    序列字典序最小）。

    与 ``decode`` 不同，扫描不抛 ``NoFrameError``：缓冲区无任何候选时
    返回全空的 ``ScanResult``。输入非法仍抛 ``SignalFormatError`` /
    ``ParameterError``；候选数超预算仍抛 ``SearchBudgetError``。
    """
    sig = _validate_signal(signal)
    res_thr = _validate_threshold(residual_threshold, "residual_threshold")
    # find_candidates 内部会校验 rho_threshold 与 max_candidates。
    candidates = find_candidates(
        sig, rho_threshold=rho_threshold, max_candidates=max_candidates
    )

    accepted: list[_Accepted] = []
    rejected: list[RejectedCandidate] = []
    truncated: list[TruncatedCandidate] = []

    for cand in candidates:
        if abs(cand.frequency_offset) >= CFO_LIMIT:
            rejected.append(RejectedCandidate(
                cand.start, cand.frequency_offset, cand.residual,
                cand.coherence, REASON_CFO_OUT_OF_RANGE,
            ))
            continue
        if cand.residual > res_thr:
            rejected.append(RejectedCandidate(
                cand.start, cand.frequency_offset, cand.residual,
                cand.coherence, REASON_PREAMBLE_RESIDUAL,
            ))
            continue
        if cand.start + FRAME_LEN > len(sig):
            truncated.append(TruncatedCandidate(
                cand.start, cand.frequency_offset, cand.residual,
                cand.coherence, len(sig) - cand.start,
            ))
            continue
        try:
            gain, pilot_residual, spectra = _frame_pilot_metrics(
                sig, cand.start, cand.frequency_offset
            )
        except ZeroPilotEnergyError:
            rejected.append(RejectedCandidate(
                cand.start, cand.frequency_offset, cand.residual,
                cand.coherence, REASON_ZERO_PILOT_ENERGY,
            ))
            continue
        if pilot_residual > res_thr:
            rejected.append(RejectedCandidate(
                cand.start, cand.frequency_offset, cand.residual,
                cand.coherence, REASON_PILOT_RESIDUAL,
            ))
            continue
        # pilot_residual <= 阈值 < 1 蕴含 gain 非零（增益为 0 时残差恰为 1）。
        symbols = _symbols_from_spectra(spectra, gain)
        accepted.append(_Accepted(cand, gain, symbols, pilot_residual))

    chosen, not_selected = _select_non_overlapping(accepted)

    frames = [
        ScannedFrame(
            start=a.candidate.start,
            frequency_offset=a.candidate.frequency_offset,
            residual=a.candidate.residual,
            pilot_residual=a.pilot_residual,
            gain=a.gain,
            symbols=a.symbols,
            bits=symbols_to_bits(a.symbols),
        )
        for a in chosen
    ]
    rejected.sort(key=lambda r: r.start)
    truncated.sort(key=lambda t: t.start)
    return ScanResult(
        frames=frames,
        rejected=rejected,
        truncated=truncated,
        not_selected=not_selected,
    )
