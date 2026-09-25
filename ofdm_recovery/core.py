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
DEFAULT_PILOT_RESIDUAL_THRESHOLD = 0.35
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

    gain, symbols, _pilot_residual = _analyze_frame(
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


# ---------------------------------------------------------------- 单帧分析（共享）


def _analyze_frame(sig, start: int, f_offset: float):
    """对已知起点/频偏的完整帧做补偿、去 CP、DFT、导频增益与 QPSK 判决。

    调用方须保证 ``start + FRAME_LEN <= len(sig)``。
    返回 ``(gain, symbols, pilot_residual)``；导频能量不足抛
    ``ZeroPilotEnergyError``。``pilot_residual`` 为导频观测相对最小二乘
    拟合的归一化均方根误差 ``sqrt(Σ|Y−h·p|² / Σ|Y|²)``（4 个导频观测），
    干净帧为 0，导频内容错误的假峰会显著大于 0。
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

    # 导频残差：导频观测与最小二乘拟合 h·p 的归一化偏差。
    err = 0.0
    for yk in spectra:
        for k, p in PILOT_VALUES.items():
            diff = yk[k] - gain * p
            err += diff.real * diff.real + diff.imag * diff.imag
    pilot_residual = math.sqrt(err / pilot_energy)

    symbols: list[complex] = []
    for yk in spectra:
        for kc in SIGNED_DATA_CARRIERS:
            k = kc % N_FFT
            symbols.append(_slice_qpsk(yk[k] / gain))

    return gain, symbols, pilot_residual


# ---------------------------------------------------------------- 多帧扫描


#: 拒绝原因代码（FrameRejection.reason 的取值）。
REASON_PREAMBLE_RESIDUAL = "preamble_residual"
REASON_FREQUENCY_OFFSET = "frequency_offset"
REASON_ZERO_PILOT_ENERGY = "zero_pilot_energy"
REASON_PILOT_RESIDUAL = "pilot_residual"


@dataclass(frozen=True)
class EvaluatedCandidate:
    """通过全部质量门限、进入选择池的完整候选。

    Attributes:
        start: 前导第一个样本的下标。
        frequency_offset: 该候选独立的频偏估计（载波间隔为单位）。
        residual: 前导补偿残差，[0, 1]。
        pilot_residual: 导频观测相对最小二乘拟合的归一化误差，干净帧为 0。
    """

    start: int
    frequency_offset: float
    residual: float
    pilot_residual: float


@dataclass(frozen=True)
class FrameRejection:
    """一个被拒绝的候选及其原因。

    Attributes:
        start: 候选起点下标。
        reason: 机器可读原因代码（``REASON_*`` 常量之一）。
        detail: 面向人的具体说明（含数值）。
    """

    start: int
    reason: str
    detail: str


@dataclass(frozen=True)
class TruncatedFrame:
    """前导可定位、但缓冲区在帧结束前截断的尾部不完整帧。

    只另列说明，绝不补零当完整帧解码。

    Attributes:
        start: 前导起点下标。
        available: 从起点起缓冲区实际可用的样本数。
        needed: 完整帧所需样本数（FRAME_LEN）。
    """

    start: int
    available: int
    needed: int


@dataclass(frozen=True)
class ScanResult:
    """多帧扫描结果。

    Attributes:
        frames: 选中的互不重叠帧的解码结果（``DecodeResult``），按起点升序。
        pool: 通过全部质量门限的候选（选择前），按起点升序。
        rejected: 被质量门限拒绝的候选及原因，按起点升序。
        truncated: 尾部不完整帧（未解码、未补零），按起点升序。
    """

    frames: tuple[DecodeResult, ...]
    pool: tuple[EvaluatedCandidate, ...]
    rejected: tuple[FrameRejection, ...]
    truncated: tuple[TruncatedFrame, ...]


def _selection_better(a, b) -> bool:
    """选择键比较：a 严格优于 b 时返回 True。

    键为 (完整帧数, 前导残差总和, 起点序列)：帧数多者优；并列时残差总和
    小者优；再并列时起点序列字典序小者优。
    """
    if a[0] != b[0]:
        return a[0] > b[0]
    if a[1] != b[1]:
        return a[1] < b[1]
    return a[2] < b[2]


def _select_non_overlapping(pool) -> list[EvaluatedCandidate]:
    """区间动态规划：从候选池选出互不重叠的最优子集。

    候选区间均为 [start, start+FRAME_LEN)。按起点排序后做后缀 DP：
    ``best[i] = max(跳过 i, 选 i + best[首个不冲突候选])``，比较键见
    ``_selection_better``。不会因为看见第一个相关峰就跳过后面整段。
    """
    ordered = sorted(pool, key=lambda c: c.start)
    n = len(ordered)
    starts = [c.start for c in ordered]
    best: list[tuple[int, float, tuple[int, ...]]] = [(0, 0.0, ())] * (n + 1)
    for i in range(n - 1, -1, -1):
        nxt = bisect.bisect_left(starts, ordered[i].start + FRAME_LEN, i + 1)
        rest = best[nxt]
        take = (
            rest[0] + 1,
            rest[1] + ordered[i].residual,
            (ordered[i].start,) + rest[2],
        )
        skip = best[i + 1]
        best[i] = take if _selection_better(take, skip) else skip
    chosen = set(best[0][2])
    return [c for c in ordered if c.start in chosen]


def scan_frames(
    signal,
    *,
    rho_threshold: float = DEFAULT_RHO_THRESHOLD,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
    pilot_threshold: float = DEFAULT_PILOT_RESIDUAL_THRESHOLD,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> ScanResult:
    """在有限缓冲区（<=512 样本）内一次扫描并恢复多帧。

    流程：按既有前导相关与固定帧长枚举全部候选；对每个候选**独立**求
    频偏、前导残差、导频残差与完整性；质量门限（前导残差、频偏范围、
    导频能量、导频残差）拒绝不合格候选并记录原因；尾部不完整帧另列；
    最后用区间动态规划选出互不重叠的最优帧组（完整帧数最多，其次前导
    残差总和最小，再起点序列字典序最小）。

    缓冲区内没有任何候选时返回全空的 ``ScanResult``（扫描是批量操作，
    空结果不是错误）；输入非法仍抛 ``SignalFormatError``/``ParameterError``，
    搜索预算耗尽仍抛 ``SearchBudgetError``。
    """
    sig = _validate_signal(signal)
    res_thr = _validate_threshold(residual_threshold, "residual_threshold")
    pil_thr = _validate_threshold(pilot_threshold, "pilot_threshold")
    # find_candidates 内部校验 rho_threshold 与 max_candidates。
    candidates = find_candidates(
        sig, rho_threshold=rho_threshold, max_candidates=max_candidates
    )

    pool: list[EvaluatedCandidate] = []
    rejected: list[FrameRejection] = []
    truncated: list[TruncatedFrame] = []
    decoded: dict[int, tuple[complex, list[complex], float]] = {}

    for cand in sorted(candidates, key=lambda c: c.start):
        if cand.residual > res_thr:
            rejected.append(FrameRejection(
                cand.start,
                REASON_PREAMBLE_RESIDUAL,
                f"preamble residual {cand.residual:.6f} > "
                f"residual_threshold={res_thr}",
            ))
            continue
        if abs(cand.frequency_offset) >= CFO_LIMIT:
            rejected.append(FrameRejection(
                cand.start,
                REASON_FREQUENCY_OFFSET,
                f"estimated CFO {cand.frequency_offset:.6f} carrier "
                f"spacings; contract requires |f| < {CFO_LIMIT}",
            ))
            continue
        available = len(sig) - cand.start
        if available < FRAME_LEN:
            truncated.append(TruncatedFrame(cand.start, available, FRAME_LEN))
            continue
        try:
            gain, symbols, pilot_residual = _analyze_frame(
                sig, cand.start, cand.frequency_offset
            )
        except ZeroPilotEnergyError as exc:
            rejected.append(FrameRejection(
                cand.start, REASON_ZERO_PILOT_ENERGY, str(exc)
            ))
            continue
        if pilot_residual > pil_thr:
            rejected.append(FrameRejection(
                cand.start,
                REASON_PILOT_RESIDUAL,
                f"pilot residual {pilot_residual:.6f} > "
                f"pilot_threshold={pil_thr} (preamble-like but pilots "
                "do not match the known +/-7 values)",
            ))
            continue
        decoded[cand.start] = (gain, symbols, pilot_residual)
        pool.append(EvaluatedCandidate(
            cand.start, cand.frequency_offset, cand.residual, pilot_residual
        ))

    frames: list[DecodeResult] = []
    for ev in _select_non_overlapping(pool):
        gain, symbols, _pr = decoded[ev.start]
        frames.append(DecodeResult(
            start=ev.start,
            frequency_offset=ev.frequency_offset,
            residual=ev.residual,
            gain=gain,
            symbols=symbols,
            bits=symbols_to_bits(symbols),
        ))

    return ScanResult(
        frames=tuple(frames),
        pool=tuple(pool),
        rejected=tuple(rejected),
        truncated=tuple(truncated),
    )
