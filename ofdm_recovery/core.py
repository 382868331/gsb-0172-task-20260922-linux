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


class StreamStateError(OFDMError):
    """流式解码器状态非法：例如在 finish() 之后继续 feed()。"""


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


def _validate_chunk(chunk) -> list[complex]:
    """流式输入块：允许空块（无操作），不受 512 样本单次上限约束。"""
    if not isinstance(chunk, (list, tuple)):
        raise SignalFormatError(
            "chunk must be a list/tuple of numeric samples, got "
            f"{type(chunk).__name__}"
        )
    return [_finite_complex(v, f"chunk[{i}]") for i, v in enumerate(chunk)]


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
    return _scan_candidates(sig, thr, budget)


def _scan_candidates(
    sig: list[complex], thr: float, budget: int
) -> list[Candidate]:
    """在已校验的复样本序列上穷举候选起点（供离线接口与流式解码共用）。"""
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

    result = _decode_frame_at(sig, best, res_thr)
    return result


def _decode_frame_at(
    sig: list[complex], best: Candidate, res_thr: float
) -> DecodeResult:
    """对已选定的候选起点执行全部后续检查并恢复一帧。

    供 decode() 与 StreamDecoder 共用；失败抛出具体 OFDMError 子类。
    """
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

    symbols: list[complex] = []
    for yk in spectra:
        for kc in SIGNED_DATA_CARRIERS:
            k = kc % N_FFT
            symbols.append(_slice_qpsk(yk[k] / gain))

    return DecodeResult(
        start=best.start,
        frequency_offset=best.frequency_offset,
        residual=best.residual,
        gain=gain,
        symbols=symbols,
        bits=symbols_to_bits(symbols),
    )


# ---------------------------------------------------------------- 流式多帧解码


@dataclass(frozen=True)
class StreamTail:
    """输入结束时的残留报告。

    Attributes:
        leftover_samples: 结束时不属于任何已恢复帧、也未被消耗的样本数
            （帧间噪声与/或残缺帧内容）。
        pending_frame_start: 已定位但未收完整的候选帧头的**绝对**下标；
            没有待定帧头时为 None。
        detail: 人类可读的残留说明。
    """

    leftover_samples: int
    pending_frame_start: int | None
    detail: str


class StreamDecoder:
    """从连续输入流中逐帧恢复多帧的增量解码器。

    输入可以任意分块（不必对齐帧头），帧间允许夹着噪声。用法::

        dec = StreamDecoder()
        for chunk in chunks:
            for result in dec.feed(chunk):
                ...   # result.start 为整流绝对下标
        results, tail = dec.finish()

    语义：

    * 每次 feed() 返回本次新恢复的全部帧（可能为空列表）；
    * 候选帧头后续检查（残差/频偏/导频能量）失败时，只跳过该起点
      一个样本并继续搜索，不会跳过后面真正的帧；
    * 尚未收完整的帧保留在内部缓冲，等下一批数据；finish() 时对仍
      残缺的候选帧头在 StreamTail 中明确报告；
    * 已证明不可能成为帧头（相关窗口已完整且不通过门限）的前缀样本
      作为噪声丢弃，缓冲不会无限增长；
    * feed()/finish() 抛出的错误（如 SearchBudgetError）不改动解码器
      状态，可修正后重试；finish() 之后禁止再 feed()。
    """

    def __init__(
        self,
        *,
        rho_threshold: float = DEFAULT_RHO_THRESHOLD,
        residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ) -> None:
        self._rho = _validate_threshold(rho_threshold, "rho_threshold")
        self._res = _validate_threshold(
            residual_threshold, "residual_threshold"
        )
        self._budget = _validate_budget(max_candidates)
        self._buf: list[complex] = []
        self._offset = 0  # self._buf[0] 在整流中的绝对下标
        self._rejected: set[int] = set()  # 检查失败的候选帧头绝对下标
        self._closed = False

    @property
    def samples_consumed(self) -> int:
        """已确定归属（帧或噪声）的样本数，即当前缓冲起点的绝对下标。"""
        return self._offset

    def feed(self, chunk) -> list[DecodeResult]:
        """追加一块输入，返回本次新恢复的帧（按起点升序）。"""
        if self._closed:
            raise StreamStateError(
                "feed() after finish(): decoder is closed; "
                "create a new StreamDecoder for a new stream"
            )
        samples = _validate_chunk(chunk)
        # 在副本上处理，成功后才提交状态：抛错不留部分变更。
        buf = self._buf + samples
        results, buf, offset, rejected, _pending = self._drain(
            buf, self._offset, set(self._rejected), final=False
        )
        self._buf = buf
        self._offset = offset
        self._rejected = rejected
        return results

    def finish(self) -> tuple[list[DecodeResult], StreamTail]:
        """结束输入：冲刷缓冲，返回 (最后恢复的帧, 残留报告)。"""
        if self._closed:
            raise StreamStateError("finish() called twice")
        results, buf, offset, rejected, pending = self._drain(
            list(self._buf), self._offset, set(self._rejected), final=True
        )
        self._buf = buf
        self._offset = offset
        self._rejected = rejected
        self._closed = True
        if pending is not None:
            start = offset + pending.start
            have = len(buf) - pending.start
            detail = (
                f"input ended with an incomplete frame: candidate header at "
                f"absolute index {start}, only {have} of {FRAME_LEN} "
                f"frame samples received; "
                f"{len(buf)} leftover sample(s) remain undecoded"
            )
            tail = StreamTail(
                leftover_samples=len(buf),
                pending_frame_start=start,
                detail=detail,
            )
        else:
            detail = (
                f"input ended with {len(buf)} leftover sample(s) of "
                "noise; no pending frame header"
            )
            tail = StreamTail(
                leftover_samples=len(buf),
                pending_frame_start=None,
                detail=detail,
            )
        return results, tail

    def _drain(
        self,
        buf: list[complex],
        offset: int,
        rejected: set[int],
        final: bool,
    ) -> tuple[list[DecodeResult], list[complex], int, set[int],
               Candidate | None]:
        """尽力从缓冲中逐帧恢复；返回 (结果, 剩余缓冲, 新偏移, 待定帧头)。"""
        results: list[DecodeResult] = []
        pending: Candidate | None = None
        while True:
            candidates = _scan_candidates(buf, self._rho, self._budget)
            if rejected:
                candidates = [
                    c for c in candidates
                    if offset + c.start not in rejected
                ]
            if not candidates:
                if not final:
                    # 起点 p 的相关窗口为 [p, p+32)：p <= len-32 的位置已
                    # 完全确定且未通过门限，只保留末尾 31 个样本即可。
                    keep = PREAMBLE_LEN - 1
                    if len(buf) > keep:
                        drop = len(buf) - keep
                        buf = buf[drop:]
                        offset += drop
                break
            complete = [
                c for c in candidates if c.start + FRAME_LEN <= len(buf)
            ]
            if not complete:
                # 最早的可用候选帧还没收完整：留到下一批（或 finish 报告）。
                pending = candidates[0]
                break
            best = complete[0]
            # 若某个残差更优的候选与 best 的帧区间重叠、但帧尚未收完整，
            # 现在解码 best 会抢占可能更好的起点 —— 等下一批再裁决。
            blocker = None
            for c in candidates:
                if c.residual >= best.residual:
                    break  # 候选按残差升序，之后的不会更优
                if (abs(c.start - best.start) < FRAME_LEN
                        and c.start + FRAME_LEN > len(buf)):
                    blocker = c
                    break
            if blocker is not None:
                pending = blocker
                break
            try:
                result = _decode_frame_at(buf, best, self._res)
            except (FrameQualityError, FrequencyOffsetError,
                    ZeroPilotEnergyError):
                # 假帧头：只否决该起点（永久：检查只依赖已收齐的样本），
                # 缓冲其余部分原样保留，继续搜索后面真正的帧。
                rejected.add(offset + best.start)
                continue
            results.append(
                DecodeResult(
                    start=offset + best.start,
                    frequency_offset=result.frequency_offset,
                    residual=result.residual,
                    gain=result.gain,
                    symbols=result.symbols,
                    bits=result.bits,
                )
            )
            buf = buf[best.start + FRAME_LEN:]
            offset += best.start + FRAME_LEN
        if pending is not None and pending.start > 0:
            # 待定帧头之前的位置相关窗口已完整且未通过门限，是确定的噪声。
            buf = buf[pending.start:]
            offset += pending.start
            pending = Candidate(
                0, pending.frequency_offset, pending.residual,
                pending.coherence,
            )
        rejected = {r for r in rejected if r >= offset}
        return results, buf, offset, rejected, pending
