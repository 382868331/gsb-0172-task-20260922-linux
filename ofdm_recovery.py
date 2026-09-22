"""OFDM cyclic-prefix boundary and subcarrier recovery (offline, self-contained).

Fixed frame contract (this module implements only this finite subset, not any
industry standard):

* N = 32 subcarriers, CP = 8 samples.
* Preamble = two identical 16-sample blocks (32 samples, no CP on preamble).
* Two OFDM symbols, each CP + N = 40 samples. Frame length = 112.
* Data subcarriers k = +/-1 .. +/-6 (12 per symbol) carry QPSK symbols
  (+/-1 +/- 1j)/sqrt(2); 24 symbols / 48 bits per frame.
* Pilot subcarriers k = -7, +7 carry known real +/-1 symbols.
* k = 0 (DC) and all other bins are zero.

The receiver:

1. Sliding delayed autocorrelation at lag 16 finds *all* candidate frame
   starts (coherence >= threshold).
2. The candidate with the smallest frequency-corrected preamble residual is
   chosen; exact residual ties are broken by the earliest position.
3. The CFO is the phase of the lag-16 correlation (unambiguous for
   |epsilon| < 0.5 subcarrier spacing by contract) and compensated.
4. CP is removed and a textbook radix-2 FFT implemented in this file is run.
5. A single complex gain is estimated by LS on the two known pilots; data
   carriers are equalized and sliced to QPSK.

Only the Python standard library is used.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = [
    "N",
    "CP",
    "BLOCK",
    "PREAMBLE_LEN",
    "SYMBOL_LEN",
    "FRAME_LEN",
    "NUM_DATA_CARRIERS",
    "DATA_CARRIERS",
    "PILOT_CARRIERS",
    "MAX_SIGNAL_LEN",
    "DEFAULT_THRESHOLD",
    "DEFAULT_PILOT_MIN_RATIO",
    "OFDMError",
    "FrameValidationError",
    "FrameNotFoundError",
    "TruncatedFrameError",
    "PilotEnergyError",
    "SearchBudgetExhaustedError",
    "Candidate",
    "RecoveryResult",
    "fft",
    "ifft",
    "make_frame",
    "find_candidates",
    "recover",
]

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

N = 32
CP = 8
BLOCK = 16
PREAMBLE_LEN = 2 * BLOCK            # 32
SYMBOL_LEN = CP + N                 # 40
NUM_OFDM_SYMBOLS = 2
FRAME_LEN = PREAMBLE_LEN + NUM_OFDM_SYMBOLS * SYMBOL_LEN   # 112

DATA_CARRIERS = tuple(range(-6, 0)) + tuple(range(1, 7))   # 12 carriers
NUM_DATA_CARRIERS = len(DATA_CARRIERS)                     # 12
PILOT_CARRIERS = (-7, 7)
BITS_PER_FRAME = NUM_OFDM_SYMBOLS * NUM_DATA_CARRIERS * 2  # 48

MAX_SIGNAL_LEN = 512
DEFAULT_THRESHOLD = 0.8
DEFAULT_PILOT_MIN_RATIO = 1e-12

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Error semantics
# ---------------------------------------------------------------------------

class OFDMError(Exception):
    """Base class for all frame acquisition/decoding errors."""


class FrameValidationError(ValueError, OFDMError):
    """An argument violates the published contract (type, finiteness, size)."""


class FrameNotFoundError(OFDMError):
    """The search was completed exhaustively and no valid frame exists."""


class TruncatedFrameError(FrameNotFoundError):
    """Best candidate is a valid preamble but the signal ends mid-frame.

    Attributes:
        start: candidate start index.
        available: samples available from ``start``.
        required: samples required for a complete frame (``FRAME_LEN``).
    """

    def __init__(self, start: int, available: int, required: int):
        self.start = start
        self.available = available
        self.required = required
        super().__init__(
            f"frame starting at {start} is truncated: {available} samples "
            f"available, {required} required"
        )


class PilotEnergyError(OFDMError):
    """Pilot bins carry (almost) no energy; the complex gain is undefined.

    Attributes:
        start: decoded frame start.
        ratio: pilot-to-used-subcarrier energy ratio that failed the floor.
    """

    def __init__(self, start: int, ratio: float):
        self.start = start
        self.ratio = ratio
        super().__init__(
            f"pilot energy ratio {ratio:.3e} at start {start} is below the "
            f"configured floor; single complex gain cannot be estimated"
        )


class SearchBudgetExhaustedError(OFDMError):
    """The search stopped because its node budget ran out.

    This is distinct from :class:`FrameNotFoundError`: no conclusion about
    the existence of a frame may be drawn when the budget is exhausted.

    Attributes:
        scanned: candidate positions evaluated before exhaustion.
        budget: the configured budget.
    """

    def __init__(self, scanned: int, budget: int):
        self.scanned = scanned
        self.budget = budget
        super().__init__(
            f"search budget exhausted after {scanned} positions (budget="
            f"{budget}); frame existence is undecided"
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _reject_bool(value, name: str):
    if isinstance(value, bool):
        raise FrameValidationError(f"{name} must not be a bool")


def _finite_complex(value, name: str, index: int | None = None) -> complex:
    """Coerce a finite real/complex sample or parameter; locate the offender."""
    where = f" at index {index}" if index is not None else ""
    if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
        raise FrameValidationError(
            f"{name}{where} must be a real or complex number, got "
            f"{type(value).__name__}"
        )
    z = complex(value)
    if not (math.isfinite(z.real) and math.isfinite(z.imag)):
        raise FrameValidationError(f"{name}{where} must be finite (no NaN/Infinity)")
    return z


def _finite_real(value, name: str) -> float:
    _reject_bool(value, name)
    if isinstance(value, complex) or not isinstance(value, (int, float)):
        raise FrameValidationError(f"{name} must be a real number")
    x = float(value)
    if not math.isfinite(x):
        raise FrameValidationError(f"{name} must be finite (no NaN/Infinity)")
    return x


def _positive_int(value, name: str) -> int:
    _reject_bool(value, name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise FrameValidationError(f"{name} must be an integer")
    if value <= 0:
        raise FrameValidationError(f"{name} must be positive, got {value}")
    return value


def _normalize_signal(signal) -> list[complex]:
    if not isinstance(signal, Sequence) or isinstance(signal, (str, bytes)):
        raise FrameValidationError(
            "signal must be a finite sequence of real/complex numbers"
        )
    if len(signal) > MAX_SIGNAL_LEN:
        raise FrameValidationError(
            f"signal length {len(signal)} exceeds the interface limit "
            f"{MAX_SIGNAL_LEN}"
        )
    if len(signal) == 0:
        raise FrameValidationError("signal must be non-empty")
    return [_finite_complex(v, "signal", i) for i, v in enumerate(signal)]


def _normalize_pilots(pilots) -> dict[int, complex]:
    if not isinstance(pilots, Sequence) or isinstance(pilots, (str, bytes)):
        raise FrameValidationError("pilots must be a sequence of (index, value)")
    table: dict[int, complex] = {}
    for i, item in enumerate(pilots):
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes))
            or len(item) != 2
        ):
            raise FrameValidationError(
                f"pilots[{i}] must be a (carrier, value) pair"
            )
        k, val = item
        _reject_bool(k, f"pilots[{i}] carrier")
        if not isinstance(k, int) or isinstance(k, bool):
            raise FrameValidationError(f"pilots[{i}] carrier must be an integer")
        if k not in PILOT_CARRIERS:
            raise FrameValidationError(
                f"pilots[{i}] carrier must be one of {PILOT_CARRIERS}, got {k}"
            )
        z = _finite_complex(val, f"pilots[{i}] value")
        if abs(z.imag) > 1e-12 or abs(abs(z.real) - 1.0) > 1e-12:
            raise FrameValidationError(
                f"pilots[{i}] value must be real +/-1, got {z}"
            )
        table[k] = complex(z.real, 0.0)
    if set(table) != set(PILOT_CARRIERS):
        raise FrameValidationError(
            f"exactly pilots {PILOT_CARRIERS} must be provided"
        )
    return table


# ---------------------------------------------------------------------------
# Core DSP (own implementation; standard library only)
# ---------------------------------------------------------------------------

def fft(x: Sequence[complex], invert: bool = False) -> list[complex]:
    """Radix-2 iterative DFT (length must be a power of two).

    Forward convention: ``X[k] = sum_n x[n] exp(-j 2 pi k n / N)``.
    The inverse divides by N. Inputs are not mutated.
    """
    if not isinstance(x, Sequence) or isinstance(x, (str, bytes)):
        raise FrameValidationError("fft input must be a sequence")
    n = len(x)
    if n == 0 or (n & (n - 1)) != 0:
        raise FrameValidationError(f"fft length must be a power of two, got {n}")
    a = [_finite_complex(v, "fft input", i) for i, v in enumerate(x)]

    # Bit-reversal permutation.
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    sign = 1.0 if invert else -1.0
    length = 2
    while length <= n:
        half = length >> 1
        wlen = cmath.exp(2j * sign * cmath.pi / length)
        for i in range(0, n, length):
            w = 1 + 0j
            for k in range(half):
                u = a[i + k]
                v = a[i + k + half] * w
                a[i + k] = u + v
                a[i + k + half] = u - v
                w *= wlen
        length <<= 1

    if invert:
        a = [v / n for v in a]
    return a


def ifft(x: Sequence[complex]) -> list[complex]:
    """Inverse of :func:`fft`."""
    return fft(x, invert=True)


@dataclass(frozen=True)
class Candidate:
    """A preamble correlation candidate.

    Attributes:
        start: sample index of the candidate frame start.
        coherence: |C| / sqrt(E0 E1) in [0, 1]; 1 means perfectly repeating
            16-sample halves.
        residual: frequency-corrected repetition residual
            (E0 + E1 - 2|C|) / (E0 + E1); 0 is a perfect repetition.
    """

    start: int
    coherence: float
    residual: float


def _correlation_scan(
    signal: Sequence[complex], node_budget: int | None
) -> list[tuple[int, float, float]]:
    """Sliding lag-16 delayed correlation over every legal start.

    Returns raw ``(start, coherence, residual)`` triples (no thresholding).
    Raises :class:`SearchBudgetExhaustedError` if the per-start budget runs
    out before the exhaustive scan completes.
    """
    L = len(signal)
    last = L - PREAMBLE_LEN          # last legal start
    raw: list[tuple[int, float, float]] = []
    if last < 0:
        return raw

    c = 0j
    e0 = 0.0
    e1 = 0.0
    for m in range(BLOCK):
        a0 = signal[m]
        a1 = signal[m + BLOCK]
        c += a1 * a0.conjugate()
        e0 += abs(a0) ** 2
        e1 += abs(a1) ** 2

    for n in range(last + 1):
        if node_budget is not None and n >= node_budget:
            raise SearchBudgetExhaustedError(n, node_budget)
        denom_energy = e0 + e1
        if e0 > 0.0 and e1 > 0.0:
            coherence = abs(c) / math.sqrt(e0 * e1)
            if coherence > 1.0:
                coherence = 1.0
            residual = (denom_energy - 2.0 * abs(c)) / denom_energy
            if residual < 0.0:
                residual = 0.0
            raw.append((n, coherence, residual))

        # Slide windows by one sample for the next iteration.
        if n < last:
            old0 = signal[n]
            mid = signal[n + BLOCK]
            new1 = signal[n + PREAMBLE_LEN]
            c += new1 * mid.conjugate() - mid * old0.conjugate()
            e0 += abs(mid) ** 2 - abs(old0) ** 2
            e1 += abs(new1) ** 2 - abs(mid) ** 2
    return raw


def find_candidates(
    signal: Sequence[complex],
    threshold: float = DEFAULT_THRESHOLD,
    node_budget: int | None = None,
) -> tuple[Candidate, ...]:
    """Return *all* preamble candidates ordered by (residual, start).

    A position is a candidate when its coherence is at least ``threshold``.
    Ties in residual are ordered by the earliest start. This performs an
    exhaustive scan; with a finite ``node_budget`` smaller than the number of
    legal positions it raises :class:`SearchBudgetExhaustedError` instead of
    returning a partial answer.
    """
    sig = _normalize_signal(signal)
    thr = _finite_real(threshold, "threshold")
    if not (0.0 < thr < 1.0):
        raise FrameValidationError("threshold must satisfy 0 < threshold < 1")
    if node_budget is not None:
        node_budget = _positive_int(node_budget, "node_budget")

    raw = _correlation_scan(sig, node_budget)
    candidates = [
        Candidate(n, coh, res)
        for n, coh, res in raw
        if coh >= thr
    ]
    candidates.sort(key=lambda c: (c.residual, c.start))
    return tuple(candidates)


# ---------------------------------------------------------------------------
# Transmit-side convenience builder (uses the same self-owned FFT)
# ---------------------------------------------------------------------------

def _default_preamble() -> list[complex]:
    """Deterministic fixed QPSK preamble block B; frame preamble is [B, B]."""
    state = 0x1234ABCD
    block: list[complex] = []
    for _ in range(BLOCK):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        b0 = (state >> 16) & 1
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        b1 = (state >> 16) & 1
        re = 1.0 if b0 == 0 else -1.0
        im = 1.0 if b1 == 0 else -1.0
        block.append(complex(re * _INV_SQRT2, im * _INV_SQRT2))
    return block


def _bits_to_symbols(bits) -> list[int]:
    if isinstance(bits, str):
        seq = list(bits)
        expected = "0/1 characters"
    else:
        if not isinstance(bits, Sequence) or isinstance(bits, (str, bytes)):
            raise FrameValidationError("bits must be a bit string or sequence")
        seq = list(bits)
        expected = "0/1 integers"
    if len(seq) != BITS_PER_FRAME:
        raise FrameValidationError(
            f"bits must contain exactly {BITS_PER_FRAME} bits "
            f"(2 symbols x {NUM_DATA_CARRIERS} carriers x 2), got {len(seq)}"
        )
    out: list[int] = []
    for i, b in enumerate(seq):
        if isinstance(b, bool) or not isinstance(b, (int, str)):
            raise FrameValidationError(f"bits[{i}] must be {expected}")
        if b not in (0, 1, "0", "1"):
            raise FrameValidationError(f"bits[{i}] must be {expected}, got {b!r}")
        out.append(int(b))
    return out


def make_frame(
    bits,
    *,
    pilots: Sequence = ((-7, 1), (7, 1)),
    preamble: Sequence[complex] | None = None,
) -> list[complex]:
    """Build one 112-sample complex baseband frame from 48 bits.

    Bit mapping per QPSK symbol: first bit sets the real sign
    (0 -> +1, 1 -> -1), second bit sets the imaginary sign. Constellation
    points are divided by sqrt(2). ``preamble`` may override the deterministic
    default 16-sample block B (the frame preamble is always B repeated twice).
    """
    bit_list = _bits_to_symbols(bits)
    pilot_table = _normalize_pilots(pilots)

    if preamble is None:
        block = _default_preamble()
    else:
        if not isinstance(preamble, Sequence) or isinstance(preamble, (str, bytes)):
            raise FrameValidationError("preamble must be a sequence of 16 samples")
        if len(preamble) != BLOCK:
            raise FrameValidationError(f"preamble block must be {BLOCK} samples")
        block = [_finite_complex(v, "preamble", i) for i, v in enumerate(preamble)]

    frame = block + block

    bit_pos = 0
    for _ in range(NUM_OFDM_SYMBOLS):
        freq = [0j] * N
        for k, pv in pilot_table.items():
            freq[k % N] = pv
        for k in DATA_CARRIERS:
            b0 = bit_list[bit_pos]
            b1 = bit_list[bit_pos + 1]
            bit_pos += 2
            re = 1.0 if b0 == 0 else -1.0
            im = 1.0 if b1 == 0 else -1.0
            freq[k % N] = complex(re * _INV_SQRT2, im * _INV_SQRT2)
        time_domain = ifft(freq)
        frame.extend(time_domain[-CP:])       # cyclic prefix
        frame.extend(time_domain)

    assert len(frame) == FRAME_LEN
    return frame


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecoveryResult:
    """Successful frame recovery output.

    Attributes:
        symbols: 24 QPSK decisions (constellation points on the unit-power
            QPSK alphabet), ordered by OFDM symbol then carrier
            ``-6..-1, 1..6``.
        bits: 48-character bit string recovered from ``symbols``.
        start: frame start index in the input signal.
        cfo: carrier frequency offset estimate, in subcarrier-spacing units
            (valid by contract for |cfo| < 0.5).
        residual: winning preamble residual (0 = perfect repetition).
        gain: LS single complex channel gain estimated from the pilots.
        candidates: every correlation candidate considered, ordered as
            :func:`find_candidates` returns them.
    """

    symbols: tuple[complex, ...]
    bits: str
    start: int
    cfo: float
    residual: float
    gain: complex
    candidates: tuple[Candidate, ...] = field(default_factory=tuple)


def _qpsk_slice(z: complex) -> tuple[complex, str]:
    # Exact-zero components resolve deterministically to the positive sign.
    re = 1.0 if z.real >= 0.0 else -1.0
    im = 1.0 if z.imag >= 0.0 else -1.0
    b0 = "0" if re > 0 else "1"
    b1 = "0" if im > 0 else "1"
    return complex(re * _INV_SQRT2, im * _INV_SQRT2), b0 + b1


def recover(
    signal: Sequence[complex],
    *,
    pilots: Sequence = ((-7, 1), (7, 1)),
    threshold: float = DEFAULT_THRESHOLD,
    pilot_min_ratio: float = DEFAULT_PILOT_MIN_RATIO,
    node_budget: int | None = None,
) -> RecoveryResult:
    """Acquire and decode one frame; see module docstring for the pipeline.

    Raises:
        FrameValidationError: invalid arguments (NaN/Infinity, bool where an
            integer is required, length > 512, bad pilot/threshold config).
        FrameNotFoundError: exhaustive scan found no candidate above
            threshold.
        TruncatedFrameError: best candidate cannot hold a complete frame.
        PilotEnergyError: pilot bins carry essentially no energy.
        SearchBudgetExhaustedError: ``node_budget`` ran out before the
            exhaustive candidate scan completed (frame existence undecided).
    """
    sig = _normalize_signal(signal)
    pilot_table = _normalize_pilots(pilots)
    thr = _finite_real(threshold, "threshold")
    if not (0.0 < thr < 1.0):
        raise FrameValidationError("threshold must satisfy 0 < threshold < 1")
    floor = _finite_real(pilot_min_ratio, "pilot_min_ratio")
    if not (0.0 <= floor < 1.0):
        raise FrameValidationError("pilot_min_ratio must satisfy 0 <= ratio < 1")
    if node_budget is not None:
        node_budget = _positive_int(node_budget, "node_budget")

    if len(sig) < FRAME_LEN:
        # Exhaustive, unconditional proof: no complete frame can fit.
        raise FrameNotFoundError(
            f"signal length {len(sig)} is shorter than one frame ({FRAME_LEN})"
        )

    candidates = find_candidates(sig, thr, node_budget)
    if not candidates:
        raise FrameNotFoundError(
            f"no preamble candidate reached coherence threshold {thr}"
        )

    winner = candidates[0]                 # sorted by (residual, start)
    start = winner.start
    if len(sig) - start < FRAME_LEN:
        raise TruncatedFrameError(start, len(sig) - start, FRAME_LEN)

    # --- CFO estimate: lag 16 -> phase pi * epsilon (N = 32) ---------------
    phase = 0.0
    for m in range(BLOCK):
        a0 = sig[start + m]
        a1 = sig[start + m + BLOCK]
        phase_value = a1 * a0.conjugate()
        phase += phase_value
    # The preamble correlation of the winner is real-valued positive after
    # aligning to the best phase; epsilon = angle(C)/pi.
    epsilon = math.atan2(phase.imag, phase.real) / math.pi

    # --- CFO compensation over the frame, CP removal, own DFT --------------
    symbols: list[complex] = []
    bits = ""
    num = 0.0
    den = 0.0
    pilot_energy = 0.0
    used_energy = 0.0

    for sym in range(NUM_OFDM_SYMBOLS):
        base = start + PREAMBLE_LEN + sym * SYMBOL_LEN + CP
        td = [
            sig[base + n]
            * cmath.exp(-2j * math.pi * epsilon * (base - start + n) / N)
            for n in range(N)
        ]
        bins = fft(td)
        for k in (*PILOT_CARRIERS, *DATA_CARRIERS):
            used_energy += abs(bins[k % N]) ** 2
        for k in PILOT_CARRIERS:
            y = bins[k % N]
            x = pilot_table[k]
            pilot_energy += abs(y) ** 2
            num += y * x.conjugate()         # known pilot x is real +/-1
            den += abs(x) ** 2
        for k in DATA_CARRIERS:
            z = bins[k % N]
            # Equalization needs the gain; defer slicing until both symbols
            # contributed to the LS estimate -- store raw bins instead.
            symbols.append(z)

    ratio = pilot_energy / used_energy if used_energy > 0.0 else 0.0
    if pilot_energy <= 0.0 or ratio < floor:
        raise PilotEnergyError(start, ratio)

    gain = num / den

    decisions: list[complex] = []
    bit_chars = ""
    for z in symbols:
        d, b = _qpsk_slice(z / gain)
        decisions.append(d)
        bit_chars += b

    return RecoveryResult(
        symbols=tuple(decisions),
        bits=bit_chars,
        start=start,
        cfo=epsilon,
        residual=winner.residual,
        gain=gain,
        candidates=candidates,
    )
