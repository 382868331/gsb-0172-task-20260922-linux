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
# 空载波（DC 与 k=8..24）能量占每符号总能量的上限。
# 真帧这些载波恒为 0；定时错误/假前导会使能量泄漏到空载波。
_NULL_CARRIER_FLOOR = 0.05
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

    def __init__(self, message: str, *, start: int | None = None,
                 available: int | None = None, needed: int | None = None):
        super().__init__(message)
        # 残缺帧信息（流式接口在输入结束报告时一定带齐这三个字段）。
        self.start = start
        self.available = available
        self.needed = needed


class StreamStateError(OFDMError):
    """流式接收器在结束之后又收到数据等非法状态转移。"""


class ZeroPilotEnergyError(OFDMError):
    """导频载波能量为零（或低于相对底限），无法估计复增益。"""


class BadFrameStructureError(OFDMError):
    """候选通过前导检查，但空载波泄漏能量过高，帧定时/帧身不可信。"""


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
    """离线恢复一帧。失败时抛出 ``OFDMError`` 的具体子类，不返回半结果。

    候选按 (残差, 起点) 排序后**逐个**验证：某个候选若被频偏、残差、
    截断或零导频检查拒绝，继续尝试下一个候选，从而不跳过其后面真正的
    帧；只有所有候选都被拒绝时，才报告排序最优候选对应的失败原因。
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

    first_error: OFDMError | None = None
    for cand in candidates:
        try:
            return _decode_candidate(sig, cand, res_thr)
        except TruncatedFrameError as exc:
            if first_error is None:
                first_error = exc
        except FrequencyOffsetError as exc:
            if first_error is None:
                first_error = exc
        except FrameQualityError as exc:
            if first_error is None:
                first_error = exc
        except ZeroPilotEnergyError as exc:
            if first_error is None:
                first_error = exc
        except BadFrameStructureError as exc:
            if first_error is None:
                first_error = exc
    assert first_error is not None
    raise first_error


def _decode_candidate(
    sig: list[complex], cand: Candidate, res_thr: float
) -> DecodeResult:
    """验证单个候选并解码；任一检查失败抛对应异常（供调用方换候选）。"""
    if abs(cand.frequency_offset) >= CFO_LIMIT:
        raise FrequencyOffsetError(
            f"estimated CFO at start {cand.start} is "
            f"{cand.frequency_offset:.6f} carrier spacings; contract requires "
            f"|f| < {CFO_LIMIT}"
        )
    if cand.residual > res_thr:
        raise FrameQualityError(
            f"candidate at start {cand.start} has preamble residual "
            f"{cand.residual:.6f} > residual_threshold={res_thr}"
        )

    end = cand.start + FRAME_LEN
    if end > len(sig):
        raise TruncatedFrameError(
            f"frame starts at {cand.start} and needs {FRAME_LEN} samples "
            f"(through index {end - 1}), but signal has only {len(sig)} "
            "samples",
            start=cand.start,
            available=len(sig) - cand.start,
            needed=FRAME_LEN,
        )

    # 全帧频偏补偿（时间原点取帧起点；绝对相位被导频增益吸收）。
    omega = -2.0 * math.pi * cand.frequency_offset / N_FFT
    seg = [
        sig[cand.start + n] * cmath.exp(1j * omega * n)
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
            f"candidate at start {cand.start}: pilot carriers (+/-7) carry "
            "no usable energy (pilot/total energy ratio floor "
            f"{_PILOT_REL_FLOOR:g}); complex gain cannot be estimated"
        )
    # 空载波校验：k=0 与 k=8..24 在真帧恒为 0。定时偏差几个样本的假候选
    # 前导相关也可能很好，但 DFT 后能量会泄漏进空载波。
    for sym_idx, yk in enumerate(spectra):
        sym_energy = sum(v.real * v.real + v.imag * v.imag for v in yk)
        if sym_energy <= 0.0:
            continue
        null_energy = sum(
            yk[k].real * yk[k].real + yk[k].imag * yk[k].imag
            for k in range(8, 25)
        ) + yk[0].real * yk[0].real + yk[0].imag * yk[0].imag
        ratio = null_energy / sym_energy
        if ratio > _NULL_CARRIER_FLOOR:
            raise BadFrameStructureError(
                f"candidate at start {cand.start} symbol {sym_idx}: null "
                f"carriers leak {ratio:.4f} of symbol energy (> "
                f"{_NULL_CARRIER_FLOOR}); frame timing/body not consistent "
                "with the fixed subcarrier layout"
            )
    gain = num / den

    symbols: list[complex] = []
    for yk in spectra:
        for kc in SIGNED_DATA_CARRIERS:
            k = kc % N_FFT
            symbols.append(_slice_qpsk(yk[k] / gain))

    return DecodeResult(
        start=cand.start,
        frequency_offset=cand.frequency_offset,
        residual=cand.residual,
        gain=gain,
        symbols=symbols,
        bits=symbols_to_bits(symbols),
    )


# ---------------------------------------------------------------- 流式接收


@dataclass(frozen=True)
class StreamStats:
    """流式接收器的可观测状态。

    Attributes:
        total_received: 历次 feed 接收并通过校验的样本总数。
        frames_emitted: 已恢复并返回的帧数。
        buffered: 当前内部保留（跨批次拼接）的样本数。
    """

    total_received: int
    frames_emitted: int
    buffered: int


class StreamDecoder:
    """从任意分块的连续输入中逐帧恢复结果。

    用法::

        dec = StreamDecoder()
        results = []
        for chunk in chunks:
            results.extend(dec.feed(chunk))   # 每块可恢复 0..N 帧
        results.extend(dec.finish())          # 无残缺时返回剩余帧

    语义：

    * 输入允许任意切分：块边界不必对齐帧头，尚未收完整的帧自动留到下一批；
    * 帧间噪声中的**误识别候选帧头**在后续检查（频偏/残差/导频）失败后被
      丢弃，搜索从该候选之后继续，绝不跳过后面真正的帧；
    * 已经恢复的帧从搜索区间移除；与已恢复帧区间重叠的候选直接忽略；
    * ``finish`` 时若仍有通过相关门限却收不完整的帧，抛
      :class:`TruncatedFrameError`（异常带 ``start/available/needed``）；
      纯噪声尾部不算残缺帧，作为 ``leftover_samples`` 保留/报告；
    * 校验或搜索失败时本次调用不产生任何部分变更（先在临时副本上计算，
      成功才提交），已返回的帧不受影响。
    """

    def __init__(
        self,
        *,
        rho_threshold: float = DEFAULT_RHO_THRESHOLD,
        residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ):
        self._rho = _validate_threshold(rho_threshold, "rho_threshold")
        self._res_thr = _validate_threshold(
            residual_threshold, "residual_threshold"
        )
        self._budget = _validate_budget(max_candidates)
        # _buf 为拼接缓冲；_base 为 _buf[0] 在整个流中的绝对下标。
        self._buf: list[complex] = []
        self._base = 0
        # 已发射帧占用的绝对区间（互不重叠、按起点排序）。
        self._emitted_spans: list[tuple[int, int]] = []
        self._frames = 0
        self._received = 0
        self._done = False

    # -------------------------------------------------- 属性 / 状态

    @property
    def frames_emitted(self) -> int:
        return self._frames

    @property
    def total_received(self) -> int:
        return self._received

    @property
    def is_finished(self) -> bool:
        return self._done

    @property
    def buffered_samples(self) -> int:
        return len(self._buf)

    def stats(self) -> StreamStats:
        return StreamStats(
            total_received=self._received,
            frames_emitted=self._frames,
            buffered=len(self._buf),
        )

    def leftover_samples(self) -> int:
        """流结束后残留的样本数（未构成可信帧的尾部）。

        ``finish`` 成功后可调用；残缺帧情形应通过抛出的
        ``TruncatedFrameError.available`` 获取残缺长度。
        """
        return len(self._buf)

    # -------------------------------------------------- 输入

    def feed(self, chunk) -> list[DecodeResult]:
        """追加一个数据块，返回本批因此新恢复出的 0..N 帧。

        空元组/空列表是合法的 no-op。任何异常都不改变接收器状态。
        ``finish`` 之后再 feed 抛 :class:`StreamStateError`。
        """
        if self._done:
            raise StreamStateError(
                "stream already finished; create a new StreamDecoder for a "
                "new stream"
            )
        if not isinstance(chunk, (list, tuple)):
            raise SignalFormatError(
                "chunk must be a list/tuple of numeric samples, got "
                f"{type(chunk).__name__}"
            )
        if len(chunk) > MAX_SIGNAL_LEN:
            raise SignalFormatError(
                f"chunk length {len(chunk)} exceeds interface limit "
                f"{MAX_SIGNAL_LEN}"
            )
        samples = [
            _finite_complex(v, f"chunk[{i}]") for i, v in enumerate(chunk)
        ]
        if not samples:
            return []

        # 先在临时状态上计算，任何异常都不产生部分变更。
        snapshot = (
            list(self._buf), self._base, list(self._emitted_spans),
            self._frames, self._received,
        )
        try:
            self._buf.extend(samples)
            self._received += len(samples)
            emitted = self._pump(final=False)
        except BaseException:
            (
                self._buf, self._base, self._emitted_spans,
                self._frames, self._received,
            ) = snapshot
            raise
        return emitted

    def finish(self) -> list[DecodeResult]:
        """宣布输入结束，返回最后能恢复的帧。

        若缓冲中仍有通过相关门限、但样本不足以覆盖完整帧的候选，抛
        :class:`TruncatedFrameError`（消息与 ``start/available/needed``
        明确给出残缺位置与长度）；接收器状态保持在 finish 之前，调用方
        仍可取用此前返回的全部帧。纯噪声/无候选尾部不报错。
        重复调用为幂等 no-op。
        """
        if self._done:
            return []
        snapshot = (
            list(self._buf), self._base, list(self._emitted_spans),
            self._frames, self._received,
        )
        try:
            emitted = self._pump(final=True)
        except BaseException:
            (
                self._buf, self._base, self._emitted_spans,
                self._frames, self._received,
            ) = snapshot
            raise
        self._done = True
        return emitted

    # -------------------------------------------------- 内部

    @staticmethod
    def _overlaps_any(start: int, end: int,
                      spans: list[tuple[int, int]]) -> bool:
        for a, b in spans:
            if start < b and a < end:
                return True
        return False

    def _trim_buffer(self, keep_from_abs: int, *,
                     hard_floor: int | None = None) -> None:
        """丢弃绝对下标 keep_from_abs 之前的样本。

        hard_floor 给出切点硬上界：调用方知道某个位置（最早存活候选，
        或最后已发射帧的起点）之前的样本仍会被相关窗口引用时传入。
        """
        if hard_floor is not None and keep_from_abs > hard_floor:
            keep_from_abs = hard_floor
        if keep_from_abs <= self._base:
            return
        cut = keep_from_abs - self._base
        if cut >= len(self._buf):
            self._buf = []
            self._base = keep_from_abs
        else:
            del self._buf[:cut]
            self._base += cut

    def _pump(self, *, final: bool) -> list[DecodeResult]:
        """在当前缓冲上尽可能多地恢复帧。

        候选按**起点升序**扫描，保证逐帧输出有序：

        * 起点更早的候选帧身不完整且前导检查通过时，必须等待下一批，
          不能越过它发射后面的帧（``final`` 时报残缺）；
        * 完整候选按 (残差, 起点) 顺序逐个验证，任一检查失败就继续
          下一个候选，误识别帧头不会跳过其后面真正的帧。
        """
        emitted: list[DecodeResult] = []
        while True:
            if len(self._buf) < PREAMBLE_LEN:
                # 连一个完整前导窗口都放不下：保留全部样本等待后续；
                # final 时它们作为纯尾部残留（不是残缺帧）。
                return emitted
            candidates = find_candidates(
                self._buf,
                rho_threshold=self._rho,
                max_candidates=self._budget,
            )
            horizon2 = self._base + len(self._buf)
            alive: list[Candidate] = []
            for c in candidates:
                start = self._base + c.start
                if not self._overlaps_any(
                    start, start + FRAME_LEN, self._emitted_spans
                ):
                    alive.append(
                        Candidate(
                            start,
                            c.frequency_offset,
                            c.residual,
                            c.coherence,
                        )
                    )
            complete = sorted(
                (c for c in alive if c.start + FRAME_LEN <= horizon2),
                key=lambda c: (c.residual, c.start),
            )
            incomplete_raw = [
                c for c in alive
                if c.start + FRAME_LEN > horizon2
                and abs(c.frequency_offset) < CFO_LIMIT
                and c.residual <= self._res_thr
            ]
            # 高原尾巴过滤：如果同一帧区间内已存在一个完整、残差更优
            # （或残差相等但起点更靠后）的候选，这个不完整候选只是该
            # 真前导在噪声里拖出的高原边缘，既不值得等待也不阻塞后面
            # 的帧。真帧尚未收完整时（无完整优候选）则保留。
            incomplete = []
            for q in incomplete_raw:
                explained = any(
                    p.start + FRAME_LEN <= horizon2
                    and q.start < p.start + FRAME_LEN
                    and p.start < q.start + FRAME_LEN
                    and (
                        p.residual < q.residual - 1e-12
                        or (abs(p.residual - q.residual) <= 1e-12
                            and p.start > q.start)
                    )
                    for p in alive
                )
                if not explained:
                    incomplete.append(q)

            # 最早的不完整候选（可能是某帧的高原边缘或真起点）。
            earliest_incomplete = (
                min(incomplete, key=lambda c: c.start)
                if incomplete else None
            )

            accepted: DecodeResult | None = None
            accepted_cand: Candidate | None = None
            # 按起点有序地处理“帧槽”：
            # 槽起点 = 不被最早未决候选压过的最早完整候选；
            # 槽内（与之帧区间重叠）的所有完整候选按残差选优验证。
            # 起点晚于最早未决候选、且不与更早完整帧同槽的候选要等待。
            ordered_complete = sorted(complete, key=lambda c: c.start)
            slot_anchor: Candidate | None = None
            for c in ordered_complete:
                if earliest_incomplete is not None and \
                        c.start >= earliest_incomplete.start:
                    continue  # 未决候选之后：等它先落定
                # 与未决候选重叠且残差更差 -> 高原假起点，跳过该候选，
                # 继续找更早的真完整帧；若它就是最早完整候选则整槽等待。
                if earliest_incomplete is not None and \
                        c.start < earliest_incomplete.start + FRAME_LEN and \
                        earliest_incomplete.start < c.start + FRAME_LEN and \
                        earliest_incomplete.residual < c.residual - 1e-12:
                    continue
                slot_anchor = c
                break
            slot = []
            if slot_anchor is not None:
                slot = sorted(
                    (
                        c for c in complete
                        if c.start < slot_anchor.start + FRAME_LEN
                        and slot_anchor.start < c.start + FRAME_LEN
                    ),
                    key=lambda c: (c.residual, c.start),
                )
            for c in slot:
                try:
                    res = _decode_candidate(
                        self._buf,
                        Candidate(
                            c.start - self._base,
                            c.frequency_offset,
                            c.residual,
                            c.coherence,
                        ),
                        self._res_thr,
                    )
                except (FrequencyOffsetError, FrameQualityError,
                        ZeroPilotEnergyError, BadFrameStructureError,
                        TruncatedFrameError):
                    # 误识别候选：只拒绝自己，继续尝试下一个候选，
                    # 不跳过后面真正的帧。
                    continue
                accepted = DecodeResult(
                    start=c.start,
                    frequency_offset=res.frequency_offset,
                    residual=res.residual,
                    gain=res.gain,
                    symbols=res.symbols,
                    bits=res.bits,
                )
                accepted_cand = c
                break

            blocking: Candidate | None = None
            if accepted_cand is None and earliest_incomplete is not None:
                # 阻塞/残缺点：在最早未决候选的帧区间内选前导残差最优
                # 者。高原边缘（残差更大）与真起点（残差 0）会同时出现，
                # 残差点必须指向真正的帧定时。
                region = [
                    q for q in incomplete
                    if q.start < earliest_incomplete.start + FRAME_LEN
                    and earliest_incomplete.start < q.start + FRAME_LEN
                ]
                blocking = min(region, key=lambda c: (c.residual, c.start))

            if blocking is not None:
                if not final:
                    hard_floor = (
                        self._emitted_spans[-1][0]
                        if self._emitted_spans else None
                    )
                    self._trim_buffer(
                        blocking.start, hard_floor=hard_floor
                    )
                    return emitted
                available = horizon2 - blocking.start
                raise TruncatedFrameError(
                    "stream ended with an incomplete frame: preamble starts "
                    f"at {blocking.start} and needs {FRAME_LEN} samples but "
                    f"only {available} are available (residual="
                    f"{blocking.residual:.6f}); {self._frames} complete frame(s) "
                    "were already recovered",
                    start=blocking.start,
                    available=available,
                    needed=FRAME_LEN,
                )

            if accepted is not None:
                assert accepted_cand is not None
                emitted.append(accepted)
                self._frames += 1
                self._emitted_spans.append(
                    (accepted.start, accepted.start + FRAME_LEN)
                )
                self._emitted_spans.sort()
                # 切点不能越过任何仍存活候选的起点（其相关窗口可能跨
                # 切点），也不能越过本帧起点（与本帧重叠的高原候选需要
                # 帧前前导样本）。
                hard_floor = min(
                    min((c.start for c in alive),
                        default=accepted.start + FRAME_LEN),
                    accepted.start,
                )
                self._trim_buffer(
                    accepted.start + FRAME_LEN - (PREAMBLE_LEN - 1),
                    hard_floor=hard_floor,
                )
                continue

            # 没有任何存活候选。最后 PREAMBLE_LEN-1 个样本即使相关窗口
            # 目前已完整落入缓冲，其“第二块”内容还会随未来数据改变，
            # 不能提前判死；任何尾部起点都可能随后续样本成为候选。
            keep = self._base + max(
                0, len(self._buf) - (PREAMBLE_LEN - 1)
            )
            hard_floor = (
                self._emitted_spans[-1][0] if self._emitted_spans else None
            )
            self._trim_buffer(keep, hard_floor=hard_floor)
            return emitted


def decode_stream(
    chunks,
    *,
    rho_threshold: float = DEFAULT_RHO_THRESHOLD,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[DecodeResult]:
    """便捷封装：把一个可迭代的分块输入整体解码为帧列表。

    内部逐块调用 :class:`StreamDecoder`，最后执行 ``finish``：因此输入
    结束时若存在残缺帧，同样抛 :class:`TruncatedFrameError`。
    """
    dec = StreamDecoder(
        rho_threshold=rho_threshold,
        residual_threshold=residual_threshold,
        max_candidates=max_candidates,
    )
    results: list[DecodeResult] = []
    for chunk in chunks:
        results.extend(dec.feed(chunk))
    results.extend(dec.finish())
    return results
