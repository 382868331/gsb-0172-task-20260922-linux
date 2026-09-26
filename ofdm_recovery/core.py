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


class ReservedCarrierError(OFDMError):
    """保留子载波集合非法：与数据/导频/DC 重叠、越界、重复或类型错误。"""


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


# ---------------------------------------------------------------- 保留子载波压峰（发射侧，默认关闭）

# 允许作为保留音调的频点：既非数据、也非导频、也非 DC 的零载波。
RESERVABLE_CARRIERS = tuple(
    k for k in range(N_FFT)
    if k not in DATA_CARRIERS and k not in PILOT_CARRIERS and k != 0
)

DEFAULT_MAX_TR_ITERATIONS = 64
_VERIFY_TOL = 1e-9


def _idft(x: list[complex]) -> list[complex]:
    """朴素 IDFT：``x[n] = (1/N) Σ X[k] exp(+j2πkn/N)``（与 dft 互逆）。"""
    n = len(x)
    return [
        sum(
            (x[k] * cmath.exp(2j * math.pi * k * m / n) for k in range(n)),
            0j,
        ) / n
        for m in range(n)
    ]


def _validate_spectrum(spectrum) -> list[complex]:
    if not isinstance(spectrum, (list, tuple)):
        raise SignalFormatError(
            "spectrum must be a list/tuple of N_FFT complex bins, got "
            f"{type(spectrum).__name__}"
        )
    if len(spectrum) != N_FFT:
        raise SignalFormatError(
            f"spectrum must have exactly {N_FFT} bins, got {len(spectrum)}"
        )
    return [_finite_complex(v, f"spectrum[{i}]") for i, v in enumerate(spectrum)]


def _validate_reserved(reserved_carriers) -> tuple[int, ...]:
    if isinstance(reserved_carriers, (list, tuple)):
        items = list(reserved_carriers)
    elif isinstance(reserved_carriers, (set, frozenset)):
        items = sorted(reserved_carriers)
    else:
        raise ReservedCarrierError(
            "reserved_carriers must be a list/tuple/set of DFT indices, got "
            f"{type(reserved_carriers).__name__}"
        )
    seen: set[int] = set()
    out: list[int] = []
    for i, k in enumerate(items):
        if isinstance(k, bool) or not isinstance(k, int):
            raise ReservedCarrierError(
                f"reserved_carriers[{i}]={k!r} is not an int DFT index"
            )
        if not 0 <= k < N_FFT:
            raise ReservedCarrierError(
                f"reserved_carriers[{i}]={k} outside valid DFT index range "
                f"0..{N_FFT - 1}"
            )
        if k in seen:
            raise ReservedCarrierError(f"reserved carrier {k} listed twice")
        if k in DATA_CARRIERS:
            raise ReservedCarrierError(
                f"reserved carrier {k} overlaps a data carrier; data bins "
                "must stay untouched"
            )
        if k in PILOT_CARRIERS:
            raise ReservedCarrierError(
                f"reserved carrier {k} overlaps a pilot carrier; pilot bins "
                "must stay untouched"
            )
        if k == 0:
            raise ReservedCarrierError(
                "reserved carrier 0 is DC; DC must stay untouched"
            )
        seen.add(k)
        out.append(k)
    return tuple(out)


def _validate_energy_budget(value) -> float:
    _reject_bool(value, "energy_budget")
    if not isinstance(value, (int, float)):
        raise ParameterError("energy_budget must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ParameterError(f"energy_budget must be finite, got {value}")
    if value < 0.0:
        raise ParameterError(f"energy_budget must be >= 0, got {value}")
    return value


def _validate_target_papr(value) -> float:
    _reject_bool(value, "target_papr_db")
    if not isinstance(value, (int, float)):
        raise ParameterError("target_papr_db must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ParameterError(f"target_papr_db must be finite, got {value}")
    return value


def _validate_iterations(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterError("max_iterations must be an int, not bool/float")
    if value < 1:
        raise ParameterError("max_iterations must be >= 1")
    return value


def _papr_db(x: list[complex]) -> float:
    """时域峰均比（dB）：max|x|² / mean|x|²；全零信号定义为 0 dB。"""
    peak = 0.0
    total = 0.0
    for v in x:
        p = v.real * v.real + v.imag * v.imag
        total += p
        if p > peak:
            peak = p
    mean = total / len(x)
    if mean <= 0.0:
        return 0.0
    return 10.0 * math.log10(peak / mean)


def _reserved_energy(X: list[complex], reserved: tuple[int, ...]) -> float:
    return sum(
        X[k].real * X[k].real + X[k].imag * X[k].imag for k in reserved
    )


@dataclass(frozen=True)
class PaprIteration:
    """一轮迭代后的观测记录。

    Attributes:
        iteration: 轮次序号；0 表示未做任何调整的初始符号。
        papr_db: 该轮时域符号的峰均比（dB）。
        reserved_energy: 该轮保留音调上的实际总能量 Σ|X[k]|²。
    """

    iteration: int
    papr_db: float
    reserved_energy: float


@dataclass(frozen=True)
class ToneReservationResult:
    """保留子载波压峰结果。

    Attributes:
        spectrum: 最终频域符号（长度 32）；数据/导频/DC 频点与输入完全一致，
            只有保留频点被赋予复值。
        time_domain: 最终时域符号（spectrum 的 IDFT，长度 32）。
        papr_db: 最终时域峰均比（dB）。
        target_papr_db: 调用方给定的目标峰均比（dB）。
        target_reached: 最终峰均比是否 <= 目标。达不到时本字段为 False，
            库只如实报告，不承诺每组数据都有足够自由度。
        reserved_energy: 保留音调上的实际总能量，保证 <= energy_budget
            （浮点误差 1e-9 相对量级内）。
        energy_budget: 调用方给定的保留音调能量上限。
        iterations: 每轮 (含初始轮 0) 的峰均比与保留能量记录。
    """

    spectrum: list[complex]
    time_domain: list[complex]
    papr_db: float
    target_papr_db: float
    target_reached: bool
    reserved_energy: float
    energy_budget: float
    iterations: tuple[PaprIteration, ...]


def reduce_papr(
    spectrum,
    reserved_carriers,
    *,
    target_papr_db: float,
    energy_budget: float,
    max_iterations: int = DEFAULT_MAX_TR_ITERATIONS,
) -> ToneReservationResult:
    """发射侧保留子载波（tone reservation）迭代投影压峰。

    输入一个长度 32 的频域 OFDM 符号 ``spectrum``（数据/导频/DC 已就位），
    只在 ``reserved_carriers`` 指定的频点上写入复值，通过迭代投影降低时域
    峰均比，同时把保留音调总能量限制在 ``energy_budget`` 以内。

    流程：

    1. 校验：保留集合与数据/导频/DC 重叠、越界、重复一律抛
       ``ReservedCarrierError``；标量参数非法抛 ``ParameterError``。
    2. 每轮按目标峰均比得到门限 T，对时域符号计算超门限残差，
       残差 DFT 后只保留保留频点（投影到保留子空间）并从 X 中减去，
       再把保留频点整体缩放回能量预算球内（投影）。
       裁剪只用于计算修正量；发射的时域波形始终是
       ``IDFT(原数据频谱 + 保留频点修正)``，从不直接裁剪发射波形。
    3. 全程只改保留频点的频域复值；逐轮记录峰均比与保留能量，
       并跟踪历史最优频谱，最终返回最优那一轮（记录仍逐轮如实给出）。
    4. 停止（达标或迭代耗尽）后用直接 DFT 重新变换最终时域符号，
       核对数据/导频/DC 频点与输入一致（容差 1e-9 相对量级）；
       不一致说明实现出错，抛 ``OFDMError``，绝不静默放行。
    5. 达不到目标不抛异常：``target_reached=False`` 如实报告。

    空保留集合或 ``energy_budget=0`` 是合法输入：没有任何自由度，
    直接返回原符号（仅含初始轮记录），通常 ``target_reached=False``。
    """
    X = _validate_spectrum(spectrum)
    reserved = _validate_reserved(reserved_carriers)
    target = _validate_target_papr(target_papr_db)
    budget = _validate_energy_budget(energy_budget)
    max_iter = _validate_iterations(max_iterations)

    original = list(X)
    records: list[PaprIteration] = []

    x = _idft(X)
    records.append(
        PaprIteration(0, _papr_db(x), _reserved_energy(X, reserved))
    )

    best_spectrum = list(X)
    best_papr = records[0].papr_db

    if reserved and budget > 0.0:
        # 内部门限比目标低 1.5 dB：门限若直接取目标，越接近目标残差越小，
        # 收敛会停滞；留 1.5 dB 余量保持修正力度（启发式步长，确定性）。
        target_lin = 10.0 ** ((target - 1.5) / 10.0)
        for it in range(1, max_iter + 1):
            if best_papr <= target:
                break
            # 超门限残差（门限由目标峰均比与当前平均功率给出）。
            mean_power = sum(
                v.real * v.real + v.imag * v.imag for v in x
            ) / N_FFT
            threshold = math.sqrt(mean_power * target_lin)
            residual = [
                v - threshold * v / abs(v) if abs(v) > threshold else 0j
                for v in x
            ]
            # 投影到保留子空间：残差 DFT 后只保留保留频点。
            corr = dft(residual)
            for k in reserved:
                X[k] -= corr[k]
            # 能量投影：超出预算则整体缩放回预算球面。
            energy = _reserved_energy(X, reserved)
            if energy > budget:
                scale = math.sqrt(budget / energy)
                for k in reserved:
                    X[k] *= scale
            x = _idft(X)
            cur_papr = _papr_db(x)
            records.append(
                PaprIteration(it, cur_papr, _reserved_energy(X, reserved))
            )
            if cur_papr < best_papr:
                best_papr = cur_papr
                best_spectrum = list(X)

    # 最终符号取历史最优一轮。
    X = best_spectrum
    x = _idft(X)

    # 停止后强制复核：最终时域重新 DFT，保护频点必须与输入一致。
    check = dft(x)
    protected = (
        list(DATA_CARRIERS) + list(PILOT_CARRIERS) + [0]
        + [k for k in range(N_FFT)
           if k not in reserved and k not in DATA_CARRIERS
           and k not in PILOT_CARRIERS and k != 0]
    )
    ref = max(1.0, max(abs(v) for v in original))
    for k in protected:
        if abs(check[k] - original[k]) > _VERIFY_TOL * ref:
            raise OFDMError(
                "internal consistency check failed: protected bin "
                f"{k} changed during tone reservation "
                f"(|delta|={abs(check[k] - original[k]):.3e}); "
                "this is a bug, not a data problem"
            )
    # 最终频谱就是算法状态 X（与 time_domain 严格互逆）。
    final_spectrum = list(X)
    final_papr = best_papr
    final_energy = _reserved_energy(final_spectrum, reserved)
    return ToneReservationResult(
        spectrum=final_spectrum,
        time_domain=x,
        papr_db=final_papr,
        target_papr_db=target,
        target_reached=final_papr <= target,
        reserved_energy=final_energy,
        energy_budget=budget,
        iterations=tuple(records),
    )
