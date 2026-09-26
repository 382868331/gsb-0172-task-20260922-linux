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

# 保留子载波（发射端 PAPR 压低）默认参数
DEFAULT_TR_MAX_ITERATIONS = 128
_TR_VERIFY_TOL = 1e-9


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
    """保留子载波集合非法：为空、越界、重复，或与数据/导频/DC 重叠。"""


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


# ---------------------------------------------------------------- 保留子载波 PAPR 压低（发射端）

# 允许作为保留音调的 DFT 索引：除数据、导频、DC 外的全部载波（8..24）。
FREE_CARRIERS = tuple(
    k for k in range(N_FFT)
    if k not in DATA_CARRIERS and k not in PILOT_CARRIERS and k != 0
)


def _idft(x) -> list[complex]:
    """朴素 IDFT：x[n] = (1/N) * sum_k X[k] exp(+j2*pi*k*n/N)。"""
    n = len(x)
    return [
        sum(
            (x[k] * cmath.exp(2j * math.pi * k * m / n) for k in range(n)),
            0j,
        )
        / n
        for m in range(n)
    ]


def _papr_linear(x) -> float:
    """线性峰均比 max|x|^2 / mean|x|^2。"""
    powers = [v.real * v.real + v.imag * v.imag for v in x]
    mean = sum(powers) / len(powers)
    if mean <= 0.0:
        return 1.0
    return max(powers) / mean


def _validate_reserved(reserved_carriers) -> tuple[int, ...]:
    if not isinstance(reserved_carriers, (list, tuple)):
        raise ReservedCarrierError(
            "reserved_carriers must be a list/tuple of DFT indices, got "
            f"{type(reserved_carriers).__name__}"
        )
    if len(reserved_carriers) == 0:
        raise ReservedCarrierError(
            "reserved_carriers is empty: no degrees of freedom exist for "
            "peak reduction; provide at least one free carrier in 8..24"
        )
    seen: set[int] = set()
    out: list[int] = []
    protected = set(DATA_CARRIERS) | set(PILOT_CARRIERS) | {0}
    for i, k in enumerate(reserved_carriers):
        if isinstance(k, bool) or not isinstance(k, int):
            raise ReservedCarrierError(
                f"reserved_carriers[{i}]={k!r} is not an int DFT index"
            )
        if not 0 <= k < N_FFT:
            raise ReservedCarrierError(
                f"reserved_carriers[{i}]={k} out of range 0..{N_FFT - 1}"
            )
        if k in protected:
            if k == 0:
                role = "DC"
            elif k in PILOT_CARRIERS:
                role = "pilot"
            else:
                role = "data"
            raise ReservedCarrierError(
                f"reserved_carriers[{i}]={k} overlaps the {role} carrier "
                "layout; reserved tones must leave data, pilots and DC "
                "untouched"
            )
        if k in seen:
            raise ReservedCarrierError(
                f"reserved_carriers[{i}]={k} is duplicated"
            )
        seen.add(k)
        out.append(k)
    return tuple(out)


def _validate_data_symbols(data_symbols) -> list[complex]:
    if not isinstance(data_symbols, (list, tuple)):
        raise SignalFormatError("data_symbols must be a list/tuple")
    if len(data_symbols) != len(SIGNED_DATA_CARRIERS):
        raise SignalFormatError(
            f"data_symbols must hold exactly {len(SIGNED_DATA_CARRIERS)} "
            f"QPSK symbols for one OFDM symbol, got {len(data_symbols)}"
        )
    return [
        _finite_complex(v, f"data_symbols[{i}]")
        for i, v in enumerate(data_symbols)
    ]


def _validate_nonnegative(value, name: str) -> float:
    _reject_bool(value, name)
    if not isinstance(value, (int, float)):
        raise ParameterError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ParameterError(f"{name} must be finite, got {value}")
    if value < 0.0:
        raise ParameterError(f"{name} must be >= 0, got {value}")
    return value


@dataclass(frozen=True)
class ToneReservationResult:
    """保留子载波迭代投影的结果。

    Attributes:
        time_signal: 压低后的 32 点时域 OFDM 符号（不含 CP），未做时域裁剪。
        spectrum: 最终 32 点频域；数据/导频/DC 频点与输入完全一致，
            只有保留频点被调整。
        reserved_carriers: 实际使用的保留频点（升序）。
        papr_history: 每轮（含初始第 0 轮）的线性峰均比。
        reserved_energy_history: 每轮保留音调总能量 sum|X[k]|^2。
        reserved_energy: 最终保留音调总能量，保证 <= max_reserved_energy。
        target_papr: 目标线性峰均比。
        achieved: 最终峰均比是否达到目标；达不到时为 False，只如实报告，
            不承诺每组数据都有足够自由度。
        iterations: 实际执行的投影轮数。
    """

    time_signal: list[complex]
    spectrum: list[complex]
    reserved_carriers: tuple[int, ...]
    papr_history: tuple[float, ...]
    reserved_energy_history: tuple[float, ...]
    reserved_energy: float
    target_papr: float
    achieved: bool
    iterations: int


def reduce_papr_tone_reservation(
    data_symbols,
    reserved_carriers,
    *,
    target_papr_db: float,
    max_reserved_energy: float,
    max_iterations: int = DEFAULT_TR_MAX_ITERATIONS,
) -> ToneReservationResult:
    """只调整保留子载波的复值，用迭代投影压低一个 OFDM 符号的时域峰值。

    流程：构造频域（数据载波取 data_symbols，导频取已知 +/-1，DC 与其余
    为 0，保留频点初始为 0），IDFT 后对超过目标幅度的样本计算时域修正量，
    修正量只投影到保留频点；保留音调总能量超预算时整体等比缩回预算内；
    重复直到峰均比达标或 max_iterations 轮耗尽。

    停止后重新对时域做 DFT，逐一核对数据/导频/DC 频点与输入完全一致
    （容差 1e-9），绝不通过直接裁剪时域波形来破坏载荷。达不到目标时
    achieved=False，如实返回最终峰均比与实际能量。

    max_reserved_energy=0 合法：保留音调被强制为 0，峰均比不会改善，
    结果如实报告未达标。
    """
    symbols = _validate_data_symbols(data_symbols)
    reserved = _validate_reserved(reserved_carriers)
    target_db = _validate_nonnegative(target_papr_db, "target_papr_db")
    if target_db <= 0.0:
        raise ParameterError(f"target_papr_db must be > 0, got {target_db}")
    budget = _validate_nonnegative(max_reserved_energy, "max_reserved_energy")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise ParameterError("max_iterations must be an int, not bool/float")
    if max_iterations < 1:
        raise ParameterError("max_iterations must be >= 1")

    target = 10.0 ** (target_db / 10.0)
    # 迭代在削峰阈值附近渐近收敛；阈值比目标内缩 0.1%，
    # 使有限轮数内最终结果稳定落到目标之下。
    aim = target * (1.0 - 1e-3)
    target_tol = target * (1.0 + 1e-6)

    # 构造初始频域：数据/导频/DC 固定，保留频点为 0。
    spectrum = [0j] * N_FFT
    for j, kc in enumerate(SIGNED_DATA_CARRIERS):
        spectrum[kc % N_FFT] = symbols[j]
    for k, p in PILOT_VALUES.items():
        spectrum[k] = p
    original = list(spectrum)

    time_signal = _idft(spectrum)
    papr_history: list[float] = [_papr_linear(time_signal)]
    energy_history: list[float] = [0.0]

    iterations = 0
    while papr_history[-1] > aim and iterations < max_iterations:
        mean_power = sum(
            v.real * v.real + v.imag * v.imag for v in time_signal
        ) / N_FFT
        clip_amp = math.sqrt(aim * mean_power)
        # 需要削掉的时域修正量（不直接应用到波形，只用于求频域投影）。
        correction = [
            v - clip_amp * v / abs(v) if abs(v) > clip_amp else 0j
            for v in time_signal
        ]
        delta = dft(correction)  # 无归一化 DFT
        # 只把修正投影到保留频点：x -= correction 对应 X[k] -= delta[k]。
        new_reserved = {k: spectrum[k] - delta[k] for k in reserved}
        energy = sum(
            v.real * v.real + v.imag * v.imag for v in new_reserved.values()
        )
        if energy > budget:
            if budget <= 0.0:
                new_reserved = {k: 0j for k in reserved}
            else:
                scale = math.sqrt(budget / energy)
                new_reserved = {k: v * scale for k, v in new_reserved.items()}
            energy = sum(
                v.real * v.real + v.imag * v.imag
                for v in new_reserved.values()
            )
        for k, v in new_reserved.items():
            spectrum[k] = v
        time_signal = _idft(spectrum)
        papr_history.append(_papr_linear(time_signal))
        energy_history.append(energy)
        iterations += 1

    # 停止后强制复核：重新 DFT，确认数据/导频/DC 频点未被破坏，
    # 保留音调能量未超预算。不满足即视为内部错误，不静默放行。
    check = dft(time_signal)
    reserved_set = set(reserved)
    for k in range(N_FFT):
        if k in reserved_set:
            continue
        if abs(check[k] - original[k]) > _TR_VERIFY_TOL * max(
            1.0, abs(original[k])
        ):
            raise OFDMError(
                f"internal verification failed: carrier {k} changed from "
                f"{original[k]} to {check[k]}; refusing to return a "
                "spectrum that corrupts the payload"
            )
    final_energy = energy_history[-1]
    if final_energy > budget * (1.0 + _TR_VERIFY_TOL):
        raise OFDMError(
            f"internal verification failed: reserved energy {final_energy} "
            f"exceeds budget {budget}"
        )

    return ToneReservationResult(
        time_signal=time_signal,
        spectrum=list(spectrum),
        reserved_carriers=tuple(sorted(reserved)),
        papr_history=tuple(papr_history),
        reserved_energy_history=tuple(energy_history),
        reserved_energy=final_energy,
        target_papr=target,
        achieved=papr_history[-1] <= target_tol,
        iterations=iterations,
    )
