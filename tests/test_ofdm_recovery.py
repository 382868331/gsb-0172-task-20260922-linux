"""Independent reference synthesizer + acceptance tests for ofdm_recovery.

The reference transmitter below is written from the frame contract directly
(direct-summation DFT, its own bit mapping and preamble) and never calls the
module under test, so expected values are produced independently. Round-trip
consistency with the module's own ``make_frame`` is checked only as a
supplementary test.
"""

import cmath
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ofdm_recovery as ofdm  # noqa: E402

N, CP, BLOCK = 32, 8, 16
FRAME_LEN = 32 + 2 * (CP + N)          # 112
DATA_K = list(range(-6, 0)) + list(range(1, 7))
PILOT_K = (-7, 7)
INV_SQRT2 = 1.0 / math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Independent reference transmitter (does NOT use ofdm_recovery)
# ---------------------------------------------------------------------------

def ref_dft(freq_bins):
    """Direct-summation IDFT: x[n] = (1/N) sum_k X[k] e^{+j 2 pi k n / N}."""
    out = []
    for n in range(N):
        acc = 0j
        for k in range(N):
            acc += freq_bins[k] * cmath.exp(2j * math.pi * k * n / N)
        out.append(acc / N)
    return out


def ref_bits_to_qpsk(bits):
    """bit0 -> real sign (0:+), bit1 -> imag sign (0:+), scaled by 1/sqrt(2)."""
    syms = []
    for i in range(0, len(bits), 2):
        re = 1.0 if bits[i] == 0 else -1.0
        im = 1.0 if bits[i + 1] == 0 else -1.0
        syms.append(complex(re * INV_SQRT2, im * INV_SQRT2))
    return syms


def ref_preamble_block():
    """Fixed deterministic 16-sample block, independent of the module's."""
    block = []
    for m in range(BLOCK):
        phase = math.pi * (m * m % 8) / 4.0   # deterministic quadratic phase
        block.append(cmath.exp(1j * phase))
    return block


def ref_frame(bits, pilots=(1, 1), preamble_block=None):
    """Synthesize one 112-sample frame per the contract, from scratch."""
    assert len(bits) == 48
    block = preamble_block if preamble_block is not None else ref_preamble_block()
    frame = list(block) + list(block)
    qpsk = ref_bits_to_qpsk(bits)
    pos = 0
    for _ in range(2):
        bins = [0j] * N
        for k, pv in zip(PILOT_K, pilots):
            bins[k % N] = complex(pv, 0.0)
        for k in DATA_K:
            bins[k % N] = qpsk[pos]
            pos += 1
        td = ref_dft(bins)
        frame.extend(td[-CP:])
        frame.extend(td)
    assert len(frame) == FRAME_LEN
    return frame


def apply_channel(frame, time_offset, cfo, gain, total_len=None):
    """Place frame at time_offset, apply per-sample CFO and complex gain."""
    n = total_len if total_len is not None else time_offset + len(frame)
    sig = [0j] * n
    for i, s in enumerate(frame):
        idx = time_offset + i
        if 0 <= idx < n:
            sig[idx] = gain * s * cmath.exp(2j * math.pi * cfo * i / N)
    return sig


def add_awgn(sig, sigma, seed=0):
    """Deterministic pseudo-Gaussian noise via Box-Muller on an LCG."""
    state = seed or 1
    out = list(sig)
    i = 0
    while i < len(out):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u1 = ((state >> 11) + 1) / (2 ** 53 + 1)
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u2 = ((state >> 11) + 1) / (2 ** 53 + 1)
        r = sigma * math.sqrt(-2.0 * math.log(u1))
        out[i] += r * math.cos(2 * math.pi * u2)
        if i + 1 < len(out):
            out[i + 1] += r * math.sin(2 * math.pi * u2)
        i += 2
    return out


BITS_A = [int(b) for b in (
    "01100011011001010110110001101100" "0110111100100001"  # arbitrary 48 bits
)]
assert len(BITS_A) == 48


class TestCleanRecovery(unittest.TestCase):
    def test_known_offsets_recovered(self):
        frame = ref_frame(BITS_A)
        sig = apply_channel(frame, time_offset=37, cfo=0.13,
                            gain=2.5 * cmath.exp(1j * 0.7))
        res = ofdm.recover(sig)
        self.assertEqual(res.start, 37)
        self.assertAlmostEqual(res.cfo, 0.13, delta=1e-9)
        self.assertAlmostEqual(abs(res.gain), 2.5, delta=1e-9)
        self.assertLess(res.residual, 1e-12)
        self.assertEqual(res.bits, "".join(str(b) for b in BITS_A))
        self.assertEqual(len(res.symbols), 24)
        for s in res.symbols:
            self.assertAlmostEqual(abs(s), 1.0, delta=1e-12)

    def test_negative_cfo_and_negative_gain(self):
        frame = ref_frame(BITS_A)
        sig = apply_channel(frame, time_offset=5, cfo=-0.31, gain=-1.0 + 0j)
        res = ofdm.recover(sig)
        self.assertEqual(res.start, 5)
        self.assertAlmostEqual(res.cfo, -0.31, delta=1e-9)
        self.assertEqual(res.bits, "".join(str(b) for b in BITS_A))

    def test_cfo_near_half_spacing(self):
        frame = ref_frame(BITS_A)
        sig = apply_channel(frame, time_offset=0, cfo=0.49, gain=1.0 + 0j)
        res = ofdm.recover(sig)
        self.assertAlmostEqual(res.cfo, 0.49, delta=1e-9)
        self.assertEqual(res.bits, "".join(str(b) for b in BITS_A))

    def test_pilot_pattern_minus_plus(self):
        frame = ref_frame(BITS_A, pilots=(-1, 1))
        sig = apply_channel(frame, time_offset=11, cfo=0.05, gain=1.3 + 0.4j)
        res = ofdm.recover(sig, pilots=((-7, -1), (7, 1)))
        self.assertEqual(res.start, 11)
        self.assertEqual(res.bits, "".join(str(b) for b in BITS_A))


class TestNoiseAndCandidates(unittest.TestCase):
    def test_leading_noise_prefix(self):
        """Noise samples before the frame must not shift the start."""
        frame = ref_frame(BITS_A)
        body = apply_channel(frame, time_offset=0, cfo=0.08, gain=1.0 + 0j)
        prefix = add_awgn([0j] * 53, sigma=0.05, seed=7)
        sig = prefix + body
        res = ofdm.recover(sig)
        self.assertEqual(res.start, 53)
        self.assertAlmostEqual(res.cfo, 0.08, delta=1e-9)
        self.assertEqual(res.bits, "".join(str(b) for b in BITS_A))

    def test_tied_candidates_pick_earliest(self):
        """Two identical frames back to back: both starts correlate equally.

        The residual tie must be broken by the earliest position, and the
        decoded bits must come from the first frame.
        """
        bits_b = [1 - b for b in BITS_A]
        sig = ref_frame(BITS_A) + ref_frame(bits_b)
        res = ofdm.recover(sig)
        starts = [c.start for c in res.candidates]
        self.assertIn(0, starts)
        self.assertIn(FRAME_LEN, starts)
        c0 = next(c for c in res.candidates if c.start == 0)
        c1 = next(c for c in res.candidates if c.start == FRAME_LEN)
        self.assertAlmostEqual(c0.residual, c1.residual, delta=1e-12)
        self.assertEqual(res.start, 0)
        self.assertEqual(res.bits, "".join(str(b) for b in BITS_A))

    def test_all_candidates_returned(self):
        sig = add_awgn([0j] * 20, 0.05, seed=3) + ref_frame(BITS_A)
        cands = ofdm.find_candidates(sig)
        self.assertGreaterEqual(len(cands), 1)
        self.assertEqual(cands[0].start, 20)
        # Ordering contract: (residual, start).
        keys = [(c.residual, c.start) for c in cands]
        self.assertEqual(keys, sorted(keys))


class TestRejections(unittest.TestCase):
    def test_zero_pilot_energy(self):
        """Frame with pilot bins forced to zero must raise PilotEnergyError."""
        frame = ref_frame(BITS_A, pilots=(0, 0))
        sig = apply_channel(frame, time_offset=9, cfo=0.1, gain=1.0 + 0j)
        with self.assertRaises(ofdm.PilotEnergyError) as ctx:
            ofdm.recover(sig)
        self.assertEqual(ctx.exception.start, 9)

    def test_truncated_frame(self):
        frame = ref_frame(BITS_A)
        # Total length >= FRAME_LEN, but the frame starting at offset 40 is
        # cut 20 samples short of a complete frame.
        sig = add_awgn([0j] * 40, 0.05, seed=5) + frame[:FRAME_LEN - 20]
        self.assertGreaterEqual(len(sig), FRAME_LEN)
        with self.assertRaises(ofdm.TruncatedFrameError) as ctx:
            ofdm.recover(sig)
        self.assertEqual(ctx.exception.start, 40)
        self.assertEqual(ctx.exception.required, FRAME_LEN)

    def test_no_frame_pure_noise(self):
        sig = add_awgn([0j] * 300, sigma=0.4, seed=11)
        with self.assertRaises(ofdm.FrameNotFoundError):
            ofdm.recover(sig)

    def test_signal_shorter_than_frame(self):
        with self.assertRaises(ofdm.FrameNotFoundError):
            ofdm.recover([0.1 + 0.1j] * 50)

    def test_budget_exhaustion_distinct_from_not_found(self):
        sig = ref_frame(BITS_A)
        with self.assertRaises(ofdm.SearchBudgetExhaustedError) as ctx:
            ofdm.recover(sig, node_budget=1)
        self.assertNotIsInstance(ctx.exception, ofdm.FrameNotFoundError)
        self.assertEqual(ctx.exception.budget, 1)


class TestValidation(unittest.TestCase):
    def test_nan_and_inf_rejected(self):
        for bad in (float("nan"), float("inf"), complex(0, float("inf"))):
            sig = [0j] * 40 + [bad] + [0j] * 80
            with self.assertRaises(ofdm.FrameValidationError):
                ofdm.recover(sig)

    def test_bool_rejected_everywhere(self):
        with self.assertRaises(ofdm.FrameValidationError):
            ofdm.recover([True] * 200)
        with self.assertRaises(ofdm.FrameValidationError):
            ofdm.recover([0j] * 200, node_budget=True)
        with self.assertRaises(ofdm.FrameValidationError):
            ofdm.recover([0j] * 200, threshold=True)
        with self.assertRaises(ofdm.FrameValidationError):
            ofdm.make_frame([True] * 48)

    def test_length_limit(self):
        with self.assertRaises(ofdm.FrameValidationError):
            ofdm.recover([0j] * 513)
        # Exactly 512 is legal input (may simply find no frame).
        try:
            ofdm.recover([0j] * 512)
        except ofdm.FrameNotFoundError:
            pass

    def test_bad_pilot_config(self):
        sig = ref_frame(BITS_A)
        with self.assertRaises(ofdm.FrameValidationError):
            ofdm.recover(sig, pilots=((-7, 1j), (7, 1)))       # not real +/-1
        with self.assertRaises(ofdm.FrameValidationError):
            ofdm.recover(sig, pilots=((-7, 1), (3, 1)))        # wrong carrier
        with self.assertRaises(ofdm.FrameValidationError):
            ofdm.recover(sig, pilots=((-7, 1),))               # missing pilot

    def test_bad_threshold_and_budget(self):
        sig = ref_frame(BITS_A)
        for thr in (0.0, 1.0, -0.5, float("nan")):
            with self.assertRaises(ofdm.FrameValidationError):
                ofdm.recover(sig, threshold=thr)
        with self.assertRaises(ofdm.FrameValidationError):
            ofdm.recover(sig, node_budget=0)

    def test_make_frame_bit_validation(self):
        with self.assertRaises(ofdm.FrameValidationError):
            ofdm.make_frame([0] * 47)
        with self.assertRaises(ofdm.FrameValidationError):
            ofdm.make_frame([0] * 24 + [2] + [0] * 23)


class TestSupplementaryRoundTrip(unittest.TestCase):
    """Round-trip via the module's own make_frame is supplementary only."""

    def test_make_frame_matches_contract_shape(self):
        frame = ofdm.make_frame(BITS_A)
        self.assertEqual(len(frame), FRAME_LEN)
        # Preamble halves identical.
        self.assertEqual(frame[:BLOCK], frame[BLOCK:2 * BLOCK])

    def test_own_builder_round_trip(self):
        frame = ofdm.make_frame(BITS_A)
        sig = apply_channel(frame, time_offset=3, cfo=-0.2, gain=0.8 - 0.6j)
        res = ofdm.recover(sig)
        self.assertEqual(res.bits, "".join(str(b) for b in BITS_A))

    def test_fft_inverts_ifft(self):
        x = [cmath.exp(1j * k * 0.37) for k in range(N)]
        back = ofdm.ifft(ofdm.fft(x))
        for a, b in zip(x, back):
            self.assertAlmostEqual(a, b, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
