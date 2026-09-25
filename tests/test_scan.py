"""多帧扫描接口 scan_frames 的必要测试。

场景：两个频偏不同的相邻帧 + 一个前导形似但导频错误的假峰。
期望值来源：``tests/reference.py`` 独立直接合成器的**输入参数**
（真实起点、频偏、增益、位串），不从被测核心反推。
选择最优性用短缓冲上对候选池的暴力子集枚举独立核对。
"""
from __future__ import annotations

import cmath
import itertools
import unittest

import _bootstrap  # noqa: F401  路径引导

import ofdm_recovery as O
from ofdm_recovery.core import _select_non_overlapping
import reference as R

BITS_A = R.fixed_bits()
BITS_B = [1 - b for b in BITS_A]  # 与 A 不同的确定位串
BITS_A_STR = "".join(str(b) for b in BITS_A)
BITS_B_STR = "".join(str(b) for b in BITS_B)

CFO_A = 0.11
CFO_B = -0.23
GAIN_A = 1.3 * cmath.exp(0.4j)
GAIN_B = 0.9 * cmath.exp(-0.7j)


def make_buffer():
    """前置噪声 + 帧A + 帧B（背靠背）+ 噪声间隔 + 假峰 + 尾部噪声。"""
    fa = R.apply_channel(R.build_frame(BITS_A), cfo=CFO_A, gain=GAIN_A)
    fb = R.apply_channel(R.build_frame(BITS_B), cfo=CFO_B, gain=GAIN_B)
    fake = R.apply_channel(
        R.build_frame(BITS_A, fake_pilots=True), cfo=0.05
    )
    sig = list(R._noise(1, 20, 0.05))
    a0 = len(sig)
    sig += fa
    b0 = len(sig)  # = a0 + FRAME_LEN，与 A 相邻
    sig += fb
    sig += R._noise(2, 16, 0.05)
    f0 = len(sig)
    sig += fake
    sig += R._noise(3, 30, 0.05)
    assert len(sig) <= O.MAX_SIGNAL_LEN
    return sig, a0, b0, f0


def brute_force_best_starts(pool):
    """对候选池暴力枚举全部互不重叠子集，返回最优起点元组。

    最优键：完整帧数最多，其次前导残差总和最小，再起点序列字典序最小。
    与被测 DP 完全独立（itertools 全枚举）。
    """
    best_key = None
    best_starts = ()
    for r in range(len(pool) + 1):
        for combo in itertools.combinations(pool, r):
            starts = tuple(sorted(c.start for c in combo))
            if any(
                starts[i + 1] - starts[i] < O.FRAME_LEN
                for i in range(len(starts) - 1)
            ):
                continue
            total_res = sum(c.residual for c in combo)
            # 元组比较即实现“帧数降序、残差升序、起点字典序升序”。
            key = (-len(combo), total_res, starts)
            if best_key is None or key < best_key:
                best_key = key
                best_starts = starts
    return best_starts


class TestScanTwoFramesAndFakePeak(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sig, cls.a0, cls.b0, cls.f0 = make_buffer()
        cls.result = O.scan_frames(cls.sig)

    def test_two_adjacent_frames_recovered(self):
        starts = [f.start for f in self.result.frames]
        self.assertEqual(starts, [self.a0, self.b0])
        # 两帧相邻：起点差恰为帧长。
        self.assertEqual(self.b0 - self.a0, O.FRAME_LEN)

    def test_absolute_positions(self):
        # 起点是缓冲区内的绝对下标：20 与 20+112。
        self.assertEqual(self.a0, 20)
        self.assertEqual(self.b0, 20 + O.FRAME_LEN)
        self.assertEqual(self.result.frames[0].start, 20)
        self.assertEqual(self.result.frames[1].start, 132)

    def test_per_frame_cfo_and_bits(self):
        fa, fb = self.result.frames
        # 两帧频偏不同，各自独立估计。
        self.assertAlmostEqual(fa.frequency_offset, CFO_A, places=10)
        self.assertAlmostEqual(fb.frequency_offset, CFO_B, places=10)
        self.assertAlmostEqual(fa.gain, GAIN_A, places=10)
        self.assertAlmostEqual(fb.gain, GAIN_B, places=10)
        # QPSK 判决结果（48 位位串）逐帧核对。
        self.assertEqual(fa.bits, BITS_A_STR)
        self.assertEqual(fb.bits, BITS_B_STR)
        self.assertEqual(len(fa.symbols), 24)
        self.assertEqual(len(fb.symbols), 24)

    def test_fake_peak_rejected_with_pilot_reason(self):
        by_start = {r.start: r for r in self.result.rejected}
        self.assertIn(self.f0, by_start)
        rej = by_start[self.f0]
        self.assertEqual(rej.reason, O.REASON_PILOT_RESIDUAL)
        self.assertIn("pilot", rej.detail)
        # 假峰不得进入输出帧。
        self.assertNotIn(self.f0, [f.start for f in self.result.frames])

    def test_selection_is_optimal_by_brute_force(self):
        # 短缓冲上枚举候选池全部子集，独立核对 DP 选择的最优性。
        self.assertLessEqual(len(self.result.pool), 12)  # 保证枚举规模小
        expect = brute_force_best_starts(self.result.pool)
        self.assertEqual(
            tuple(f.start for f in self.result.frames), expect
        )

    def test_pool_and_frames_consistent(self):
        # 输出帧全部来自候选池，且互不重叠。
        pool_starts = {c.start for c in self.result.pool}
        starts = [f.start for f in self.result.frames]
        for s in starts:
            self.assertIn(s, pool_starts)
        for i in range(len(starts) - 1):
            self.assertGreaterEqual(starts[i + 1] - starts[i], O.FRAME_LEN)


class TestScanRejections(unittest.TestCase):
    def test_preamble_residual_rejection(self):
        # 相关性强但两块幅度失配（残差 0.45）的假前导 + 一个正常帧。
        block = R._preamble_block()
        good = R.apply_channel(R.build_frame(BITS_A), cfo=0.1)
        sig = [0j] * 5 + block + [0.3 * x for x in block] + [0j] * 90
        g0 = len(sig)
        sig += good
        result = O.scan_frames(sig)
        by_start = {r.start: r for r in result.rejected}
        self.assertEqual(by_start[5].reason, O.REASON_PREAMBLE_RESIDUAL)
        self.assertEqual([f.start for f in result.frames], [g0])
        self.assertEqual(result.frames[0].bits, BITS_A_STR)

    def test_frequency_offset_rejection(self):
        block = R._preamble_block()
        # 第二块多转 0.6π -> 估计频偏 0.6，超出 |f|<0.5。
        sig = [0j] * 5 + block + [
            x * cmath.exp(0.6j * 3.141592653589793) for x in block
        ] + [0j] * 90
        result = O.scan_frames(sig)
        self.assertEqual(result.frames, ())
        reasons = {r.reason for r in result.rejected}
        self.assertIn(O.REASON_FREQUENCY_OFFSET, reasons)

    def test_zero_pilot_energy_rejection(self):
        frame = R.apply_channel(
            R.build_frame(BITS_A, zero_pilots=True), cfo=0.1
        )
        sig = [0j] * 7 + frame
        result = O.scan_frames(sig)
        self.assertEqual(result.frames, ())
        by_start = {r.start: r for r in result.rejected}
        self.assertEqual(by_start[7].reason, O.REASON_ZERO_PILOT_ENERGY)

    def test_truncated_tail_listed_not_padded(self):
        # 一个完整帧 + 尾部 60 样本的不完整帧（前导完整、帧体截断）。
        good = R.apply_channel(R.build_frame(BITS_A), cfo=0.05)
        partial = R.apply_channel(R.build_frame(BITS_B), cfo=-0.1)[:60]
        sig = [0j] * 4 + good + partial
        result = O.scan_frames(sig)
        self.assertEqual([f.start for f in result.frames], [4])
        self.assertEqual(result.frames[0].bits, BITS_A_STR)
        t0 = 4 + O.FRAME_LEN
        # 截断帧的前导在若干相邻偏移都能相关上，全部另列、一个也不解码。
        trunc_by_start = {t.start: t for t in result.truncated}
        self.assertIn(t0, trunc_by_start)
        trunc = trunc_by_start[t0]
        self.assertEqual(trunc.available, 60)
        self.assertEqual(trunc.needed, O.FRAME_LEN)
        for t in result.truncated:
            self.assertLess(t.available, O.FRAME_LEN)
        # 截断帧未被补零解码：不出现在 frames/pool。
        self.assertNotIn(t0, [f.start for f in result.frames])
        self.assertNotIn(t0, [c.start for c in result.pool])

    def test_no_candidates_returns_empty_result(self):
        sig = R._noise(555, 200, 0.3)
        result = O.scan_frames(sig)
        self.assertEqual(result.frames, ())
        self.assertEqual(result.pool, ())
        self.assertEqual(result.rejected, ())
        self.assertEqual(result.truncated, ())


class TestSelectionDp(unittest.TestCase):
    """区间 DP 选择的针对性用例（合成候选池，独立于信号层面）。"""

    @staticmethod
    def _pool(entries):
        return [
            O.EvaluatedCandidate(start, 0.0, res, 0.0)
            for start, res in entries
        ]

    def test_first_peak_does_not_block_later_segment(self):
        # 残差最低的候选在 50，选它则 0 与 112 都冲突（只剩 1 帧）；
        # DP 必须放弃它，选 {0, 112} 两帧——不是看见第一个峰就定终身。
        pool = self._pool([(50, 0.0), (0, 0.01), (112, 0.01)])
        chosen = _select_non_overlapping(pool)
        self.assertEqual([c.start for c in chosen], [0, 112])

    def test_residual_sum_tiebreak(self):
        # 两种选法都是 1 帧：选残差更小的。
        pool = self._pool([(0, 0.20), (50, 0.05)])
        chosen = _select_non_overlapping(pool)
        self.assertEqual([c.start for c in chosen], [50])

    def test_lexicographic_tiebreak(self):
        # 帧数与残差总和都相同：选起点序列字典序更小的 {0, 117}。
        pool = self._pool([(0, 0.1), (5, 0.1), (117, 0.1)])
        chosen = _select_non_overlapping(pool)
        self.assertEqual([c.start for c in chosen], [0, 117])

    def test_dp_matches_brute_force_on_synthetic_pool(self):
        # 含重叠与并列的合成池，暴力枚举核对。
        pool = self._pool([
            (0, 0.10), (40, 0.05), (112, 0.20), (150, 0.03),
            (224, 0.10), (260, 0.10),
        ])
        chosen = tuple(c.start for c in _select_non_overlapping(pool))
        self.assertEqual(chosen, brute_force_best_starts(pool))


class TestScanValidation(unittest.TestCase):
    def test_signal_format_errors(self):
        with self.assertRaises(O.SignalFormatError):
            O.scan_frames("0101")
        with self.assertRaises(O.SignalFormatError):
            O.scan_frames([])
        with self.assertRaises(O.SignalFormatError):
            O.scan_frames([0j] * (O.MAX_SIGNAL_LEN + 1))
        sig = [0j] * 40
        sig[3] = float("nan")
        with self.assertRaises(O.SignalFormatError) as cm:
            O.scan_frames(sig)
        self.assertIn("signal[3]", str(cm.exception))

    def test_parameter_errors(self):
        sig = R.transmit(BITS_A)
        for bad in (0.0, 1.0, -0.1, float("nan")):
            with self.assertRaises(O.ParameterError):
                O.scan_frames(sig, residual_threshold=bad)
            with self.assertRaises(O.ParameterError):
                O.scan_frames(sig, pilot_threshold=bad)
        with self.assertRaises(O.ParameterError):
            O.scan_frames(sig, max_candidates=True)
        with self.assertRaises(O.ParameterError):
            O.scan_frames(sig, rho_threshold=False)

    def test_budget_exhaustion_still_distinct(self):
        with self.assertRaises(O.SearchBudgetError):
            O.scan_frames([1 + 0j] * 60, max_candidates=5)

    def test_max_signal_len_boundary_accepted(self):
        # 长度恰为 512 的静漠信号：合法输入，空结果而非格式错误。
        result = O.scan_frames([0j] * O.MAX_SIGNAL_LEN)
        self.assertEqual(result.frames, ())


class TestScanRoundTripSupplementary(unittest.TestCase):
    """往返一致性仅作补充：随机位串双帧干净信道下自洽。"""

    def test_random_double_frame_round_trip(self):
        import random

        rng = random.Random(20260925)
        for _ in range(3):
            bits1 = [rng.randrange(2) for _ in range(48)]
            bits2 = [rng.randrange(2) for _ in range(48)]
            off = rng.randrange(0, 40)
            cfo1 = rng.uniform(-0.4, 0.4)
            cfo2 = rng.uniform(-0.4, 0.4)
            f1 = R.apply_channel(R.build_frame(bits1), cfo=cfo1)
            f2 = R.apply_channel(R.build_frame(bits2), cfo=cfo2)
            sig = [0j] * off + f1 + f2
            result = O.scan_frames(sig)
            self.assertEqual(
                [f.start for f in result.frames],
                [off, off + O.FRAME_LEN],
            )
            self.assertEqual(
                result.frames[0].bits, "".join(str(b) for b in bits1)
            )
            self.assertEqual(
                result.frames[1].bits, "".join(str(b) for b in bits2)
            )


if __name__ == "__main__":
    unittest.main()
