"""多帧扫描接口 scan_frames 的必要测试。

场景：有限缓冲区内两个频偏不同的相邻帧 + 一个前导形似但导频错误的
假峰 + 一个尾部不完整帧。期望值来自 tests/reference.py 独立合成器的
输入参数；选择最优性用短缓冲上候选子集的暴力枚举核对。
"""
from __future__ import annotations

import cmath
import itertools
import math
import unittest

import _bootstrap  # noqa: F401  路径引导

import ofdm_recovery as O
from ofdm_recovery import core
import reference as R

BITS_A = R.fixed_bits()
BITS_A_STR = "".join(str(b) for b in BITS_A)
BITS_B = [1 - b for b in BITS_A]  # 第二帧取反，确保两帧位串不同
BITS_B_STR = "".join(str(b) for b in BITS_B)

CFO_A = 0.21
CFO_B = -0.33
GAIN_A = 1.3 * cmath.exp(0.4j)
GAIN_B = 0.9 * cmath.exp(-0.2j)
FRONT = 17          # 前置噪声样本数
START_A = FRONT     # 17
START_B = START_A + O.FRAME_LEN          # 129
START_FAKE = START_B + O.FRAME_LEN       # 241
START_TAIL = START_FAKE + O.FRAME_LEN    # 353
TAIL_KEPT = 60      # 尾部帧只保留前 60 样本（< 112）


def _apply_channel(frame, cfo, gain):
    """与 reference.transmit 相同的逐帧信道：增益 + 帧内频偏。"""
    return [
        x * gain * cmath.exp(2j * math.pi * cfo * n / R.N)
        for n, x in enumerate(frame)
    ]


def _fake_frame_wrong_pilots(bits):
    """前导完全正常、但导频与契约不符的假帧。

    契约要求每符号导频 (+1, -1)；本假帧符号 0 发 (+1, +1)、符号 1 发
    (-1, -1)：任一整数样本错位下四个导频观测都相互抵消，导频残差
    保持 ≈1，保证假峰及其前导相关平台衍生的错位幽灵候选都被拒绝。
    """
    block = R._preamble_block()
    frame = list(block) + list(block)
    syms = R.qpsk_modulate(bits)
    carriers = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
    wrong_pilots = [(1 + 0j, 1 + 0j), (-1 + 0j, -1 + 0j)]
    for s in range(R.N_SYM):
        X = [0j] * R.N
        for j, kc in enumerate(carriers):
            X[kc % R.N] = syms[s * 12 + j]
        X[7], X[-7 % R.N] = wrong_pilots[s]
        time = R._idft(X)
        frame.extend(time[-R.CP:])
        frame.extend(time)
    assert len(frame) == R.FRAME_LEN
    return frame


def build_demo_buffer():
    """构造演示缓冲区（无带内噪声，前置噪声确定性），返回 (signal, 总长)。"""
    sig = list(R._noise(2026, FRONT, 0.05))
    sig += _apply_channel(R.build_frame(BITS_A), CFO_A, GAIN_A)
    sig += _apply_channel(R.build_frame(BITS_B), CFO_B, GAIN_B)
    sig += _apply_channel(
        _fake_frame_wrong_pilots(BITS_A), 0.1, 1.1 * cmath.exp(0.3j)
    )
    tail = _apply_channel(R.build_frame(BITS_A), -0.05, 1.0 + 0j)
    sig += tail[:TAIL_KEPT]
    assert len(sig) == START_TAIL + TAIL_KEPT  # 413 <= 512
    return sig


def _brute_force_best(accepted):
    """暴力枚举 accepted（(start, residual) 列表）的全部子集，
    返回最优互不重叠子集的起点元组；比较规则与契约相同。"""
    best = None
    for r in range(len(accepted) + 1):
        for combo in itertools.combinations(range(len(accepted)), r):
            starts = tuple(sorted(accepted[i][0] for i in combo))
            ok = all(
                starts[j + 1] - starts[j] >= O.FRAME_LEN
                for j in range(len(starts) - 1)
            )
            if not ok:
                continue
            key = (len(combo), -sum(accepted[i][1] for i in combo))
            if (
                best is None
                or key > best[0]
                or (key == best[0] and starts < best[1])
            ):
                best = (key, starts)
    return best[1]


class TestScanDemoBuffer(unittest.TestCase):
    """两个不同频偏的相邻帧 + 假峰 + 尾部截断帧的完整场景。"""

    def setUp(self):
        self.sig = build_demo_buffer()
        self.result = O.scan_frames(self.sig)

    def test_two_frames_recovered_with_absolute_positions(self):
        frames = self.result.frames
        self.assertEqual([f.start for f in frames], [START_A, START_B])
        self.assertAlmostEqual(frames[0].frequency_offset, CFO_A, places=10)
        self.assertAlmostEqual(frames[1].frequency_offset, CFO_B, places=10)
        self.assertAlmostEqual(frames[0].gain, GAIN_A, places=10)
        self.assertAlmostEqual(frames[1].gain, GAIN_B, places=10)
        self.assertLess(frames[0].residual, 1e-9)
        self.assertLess(frames[1].pilot_residual, 1e-9)

    def test_qpsk_symbols_and_bits(self):
        f0, f1 = self.result.frames
        self.assertEqual(f0.bits, BITS_A_STR)
        self.assertEqual(f1.bits, BITS_B_STR)
        self.assertEqual(len(f0.symbols), 24)
        q = 1.0 / math.sqrt(2.0)
        for s in f0.symbols + f1.symbols:
            self.assertAlmostEqual(abs(s.real), q, places=12)
            self.assertAlmostEqual(abs(s.imag), q, places=12)

    def test_fake_peak_rejected_for_wrong_pilots(self):
        fake = [r for r in self.result.rejected if r.start == START_FAKE]
        self.assertEqual(len(fake), 1)
        self.assertEqual(fake[0].reason, O.REASON_PILOT_RESIDUAL)
        # 假峰前导完全正常：前导残差接近 0，靠导频残差拒绝。
        self.assertLess(fake[0].residual, 1e-9)
        # 前导相关平台衍生的错位幽灵候选（±2 样本内）同样全部被拒绝，
        # 不会借错位混入输出帧。
        ghosts = [
            r for r in self.result.rejected
            if 0 < abs(r.start - START_FAKE) <= 2
        ]
        self.assertTrue(ghosts)
        for g in ghosts:
            self.assertEqual(g.reason, O.REASON_PILOT_RESIDUAL)
        for f in self.result.frames:
            self.assertGreater(abs(f.start - START_FAKE), 2)

    def test_truncated_tail_listed_not_decoded(self):
        tails = self.result.truncated
        starts = [t.start for t in tails]
        # 尾部帧及其前导相关平台衍生的错位候选都被另列，不进入 frames。
        self.assertIn(START_TAIL, starts)
        for t in tails:
            self.assertLess(t.available, O.FRAME_LEN)
            self.assertEqual(t.start + t.available, len(self.sig))
            self.assertNotIn(t.start, [f.start for f in self.result.frames])
        exact = [t for t in tails if t.start == START_TAIL]
        self.assertEqual(exact[0].available, TAIL_KEPT)

    def test_frames_sorted_and_non_overlapping(self):
        starts = [f.start for f in self.result.frames]
        self.assertEqual(starts, sorted(starts))
        for a, b in zip(starts, starts[1:]):
            self.assertGreaterEqual(b - a, O.FRAME_LEN)

    def test_selection_is_optimal_by_brute_force(self):
        # 短缓冲上枚举全部合格候选的子集，核对 DP 选择的最优性。
        accepted = [(f.start, f.residual) for f in self.result.frames]
        accepted += [(c.start, c.residual) for c in self.result.not_selected]
        best = _brute_force_best(accepted)
        self.assertEqual(tuple(f.start for f in self.result.frames), best)


class TestScanRejections(unittest.TestCase):
    def test_preamble_residual_rejection(self):
        # 第二块纯幅度缩放：相干性 1 但前导残差 0.45 超阈值。
        block = R._preamble_block()
        sig = [0j] * 5 + block + [0.3 * x for x in block] + [0j] * 90
        result = O.scan_frames(sig)
        self.assertEqual(result.frames, [])
        reasons = {r.reason for r in result.rejected}
        self.assertIn(O.REASON_PREAMBLE_RESIDUAL, reasons)
        top = [r for r in result.rejected if r.start == 5]
        self.assertEqual(top[0].reason, O.REASON_PREAMBLE_RESIDUAL)

    def test_cfo_out_of_range_rejection(self):
        block = R._preamble_block()
        sig = [0j] * 5 + block + [
            x * cmath.exp(0.6j * math.pi) for x in block
        ] + [0j] * 90
        result = O.scan_frames(sig)
        self.assertEqual(result.frames, [])
        top = [r for r in result.rejected if r.start == 5]
        self.assertEqual(top[0].reason, O.REASON_CFO_OUT_OF_RANGE)

    def test_zero_pilot_energy_rejection(self):
        sig = R.transmit(BITS_A, time_offset=9, zero_pilots=True)
        result = O.scan_frames(sig)
        self.assertEqual(result.frames, [])
        top = [r for r in result.rejected if r.start == 9]
        self.assertEqual(top[0].reason, O.REASON_ZERO_PILOT_ENERGY)

    def test_residual_threshold_is_caller_controlled(self):
        block = R._preamble_block()
        sig = [0j] * 5 + block + [0.3 * x for x in block] + [0j] * 90
        # 放宽阈值后该候选不再因前导残差被拒绝（但导频能量为零仍拒绝）。
        result = O.scan_frames(sig, residual_threshold=0.9)
        self.assertNotIn(
            O.REASON_PREAMBLE_RESIDUAL,
            {r.reason for r in result.rejected},
        )

    def test_empty_buffer_returns_empty_result_not_exception(self):
        result = O.scan_frames(R._noise(555, 200, 0.3))
        self.assertEqual(result.frames, [])
        self.assertEqual(result.rejected, [])
        self.assertEqual(result.truncated, [])
        self.assertEqual(result.not_selected, [])

    def test_validation_and_budget_errors_still_raise(self):
        with self.assertRaises(O.SignalFormatError):
            O.scan_frames([float("nan")] * 40)
        with self.assertRaises(O.ParameterError):
            O.scan_frames([0j] * 40, residual_threshold=1.5)
        with self.assertRaises(O.ParameterError):
            O.scan_frames([0j] * 40, rho_threshold=True)
        with self.assertRaises(O.SearchBudgetError):
            O.scan_frames([1 + 0j] * 60, max_candidates=5)
        with self.assertRaises(O.SignalFormatError):
            O.scan_frames([0j] * (O.MAX_SIGNAL_LEN + 1))


class TestIntervalDpOptimality(unittest.TestCase):
    """用合成候选集 + 暴力枚举核对区间 DP（含重叠与平局）。"""

    @staticmethod
    def _accepted(start, residual):
        cand = core.Candidate(start, 0.0, residual, 1.0)
        return core._Accepted(cand, 1 + 0j, [], 0.0)

    def _check(self, cases):
        accepted = [self._accepted(s, r) for s, r in cases]
        chosen, not_selected = core._select_non_overlapping(accepted)
        chosen_starts = tuple(a.candidate.start for a in chosen)
        self.assertEqual(chosen_starts, _brute_force_best(cases))
        # 互补性：未选中的恰为其余候选。
        self.assertEqual(
            len(chosen) + len(not_selected), len(accepted)
        )

    def test_random_instances_against_brute_force(self):
        import random

        rng = random.Random(417)
        for _ in range(30):
            n = rng.randrange(1, 11)
            cases = [
                (rng.randrange(0, 300), rng.random() * 0.3)
                for _ in range(n)
            ]
            self._check(cases)

    def test_count_beats_residual(self):
        # {0, 112} 两帧残差和 0.6，优于不相交单帧 {56} 残差 0.0：
        # 帧数优先于残差总和。
        cases = [(0, 0.3), (56, 0.0), (112, 0.3)]
        self._check(cases)
        chosen, _ = core._select_non_overlapping(
            [self._accepted(s, r) for s, r in cases]
        )
        self.assertEqual(
            [a.candidate.start for a in chosen], [0, 112]
        )

    def test_residual_breaks_count_tie(self):
        # 帧数相同（各 2 帧）：{5, 117} 残差和 0.1 优于 {0, 117} 的 0.4。
        chosen, not_selected = core._select_non_overlapping(
            [self._accepted(0, 0.4), self._accepted(5, 0.1),
             self._accepted(117, 0.0)]
        )
        self.assertEqual([a.candidate.start for a in chosen], [5, 117])
        self.assertEqual([c.start for c in not_selected], [0])

    def test_lexicographic_breaks_full_tie(self):
        # 帧数与残差和都相同：选起点序列字典序最小的 {0, 117}。
        chosen, _ = core._select_non_overlapping(
            [self._accepted(5, 0.0), self._accepted(0, 0.0),
             self._accepted(117, 0.0)]
        )
        self.assertEqual([a.candidate.start for a in chosen], [0, 117])

    def test_greedy_first_peak_would_lose(self):
        # 最早的峰残差最差且与 100 重叠：见峰就取会选 {0, 212}
        # （残差和 0.34）；最优是 {100, 212}（残差和 0）。
        chosen, _ = core._select_non_overlapping(
            [self._accepted(0, 0.34), self._accepted(100, 0.0),
             self._accepted(212, 0.0)]
        )
        self.assertEqual([a.candidate.start for a in chosen], [100, 212])


class TestScanNoisyMultiFrame(unittest.TestCase):
    def test_noisy_adjacent_frames(self):
        # 带内噪声下两帧仍应各自恢复（误差口径同单帧 README）。
        sig = list(R._noise(31, 10, 0.05))
        frame_a = R.transmit(BITS_A, cfo=0.15, inband_scale=0.05, seed=5)
        frame_b = R.transmit(BITS_B, cfo=-0.25, inband_scale=0.05, seed=6)
        sig += frame_a + frame_b
        result = O.scan_frames(sig)
        self.assertEqual([f.start for f in result.frames], [10, 122])
        self.assertLess(abs(result.frames[0].frequency_offset - 0.15), 0.02)
        self.assertLess(abs(result.frames[1].frequency_offset + 0.25), 0.02)
        self.assertEqual(result.frames[0].bits, BITS_A_STR)
        self.assertEqual(result.frames[1].bits, BITS_B_STR)


if __name__ == "__main__":
    unittest.main()
