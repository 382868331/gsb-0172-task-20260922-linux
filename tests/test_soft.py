"""软解调（QPSK / 16QAM LLR）必要测试。

期望值来源：本文件内的**独立**直接概率求和参考实现（逐点 math.exp
累加 P0/P1 再取对数比），不调用被测核心的 log-sum-exp 路径。
"""
from __future__ import annotations

import cmath
import math
import unittest

import _bootstrap  # noqa: F401  路径引导

import ofdm_recovery as O
import reference as R

SQRT1_2 = 1.0 / math.sqrt(2.0)


def direct_llr(y, table, bit_index, var):
    """独立参考：直接对两组星座点求概率和，再取对数比（正值偏向 0）。"""
    p0 = 0.0
    p1 = 0.0
    for bits, point in table.items():
        d2 = (y - point).real ** 2 + (y - point).imag ** 2
        w = math.exp(-d2 / var)
        if bits[bit_index] == 0:
            p0 += w
        else:
            p1 += w
    return math.log(p0) - math.log(p1)


def direct_all(y, table, var):
    return [direct_llr(y, table, b, var) for b in range(len(next(iter(table))))]


class TestQpskSoftAgainstDirectSum(unittest.TestCase):
    VAR = 0.25

    def check(self, y):
        got = O.soft_demodulate([y], modulation="qpsk", noise_variance=self.VAR)
        expect = direct_all(y, O.QPSK_GRAY, self.VAR)
        self.assertEqual(len(got.llrs), 2)
        self.assertEqual(got.erasures, (False,))
        for g, e in zip(got.llrs, expect):
            self.assertAlmostEqual(g, e, places=12)

    def test_constellation_centers(self):
        for point in O.QPSK_GRAY.values():
            self.check(point)

    def test_decision_boundary_midpoints(self):
        for y in (0j, 0.5 + 0j, -0.3j, complex(0.0, SQRT1_2)):
            self.check(y)
        got = O.soft_demodulate([0j], modulation="qpsk", noise_variance=self.VAR)
        self.assertEqual(got.llrs, (0.0, 0.0))

    def test_positive_llr_favors_bit_zero(self):
        got = O.soft_demodulate(
            [complex(SQRT1_2, SQRT1_2)], modulation="qpsk", noise_variance=0.1
        )
        self.assertGreater(got.llrs[0], 0.0)
        self.assertGreater(got.llrs[1], 0.0)
        got = O.soft_demodulate(
            [complex(-SQRT1_2, -SQRT1_2)], modulation="qpsk", noise_variance=0.1
        )
        self.assertLess(got.llrs[0], 0.0)
        self.assertLess(got.llrs[1], 0.0)

    def test_very_low_snr_llr_shrinks_to_zero(self):
        y = complex(SQRT1_2, SQRT1_2)
        got = O.soft_demodulate([y], modulation="qpsk", noise_variance=1e6)
        for v in got.llrs:
            self.assertLess(abs(v), 1e-4)


class TestQam16SoftAgainstDirectSum(unittest.TestCase):
    VAR = 0.2

    def check(self, y):
        got = O.soft_demodulate([y], modulation="16qam", noise_variance=self.VAR)
        expect = direct_all(y, O.QAM16_GRAY, self.VAR)
        self.assertEqual(len(got.llrs), 4)
        for g, e in zip(got.llrs, expect):
            self.assertAlmostEqual(g, e, places=12)

    def test_all_constellation_centers(self):
        self.assertEqual(len(O.QAM16_GRAY), 16)
        for point in O.QAM16_GRAY.values():
            self.check(point)

    def test_gray_table_is_gray_along_each_axis(self):
        scale = math.sqrt(10.0)
        by_row = {}
        for bits, point in O.QAM16_GRAY.items():
            by_row.setdefault(round(point.imag * scale), {})[
                round(point.real * scale)
            ] = bits[:2]
        for row in by_row.values():
            for a, b in ((-3, -1), (-1, 1), (1, 3)):
                x, y = row[a], row[b]
                self.assertEqual(sum(u != v for u, v in zip(x, y)), 1)

    def test_midpoints_and_axis_boundaries(self):
        s = 1.0 / math.sqrt(10.0)
        for y in (0j, complex(0, 2 * s), complex(2 * s, 0), complex(-2 * s, s)):
            self.check(y)
        got = O.soft_demodulate([0j], modulation="16qam", noise_variance=self.VAR)
        # 原点处符号位（b0/b2）LLR 精确为 0；幅度位倾向内层电平，非零，
        # 其值已由 check(0j) 与直接概率求和对照锁定。
        self.assertEqual((got.llrs[0], got.llrs[2]), (0.0, 0.0))

    def test_very_low_snr(self):
        got = O.soft_demodulate(
            [O.QAM16_GRAY[(0, 0, 0, 0)]], modulation="16qam",
            noise_variance=1e6,
        )
        for v in got.llrs:
            self.assertLess(abs(v), 1e-4)


class TestGainsAndErasure(unittest.TestCase):
    def test_gain_scaling_matches_direct_sum(self):
        h = 2.0 * cmath.exp(0.7j)
        s = O.QPSK_GRAY[(0, 1)]
        y = h * s + complex(0.05, -0.03)
        var = 0.1
        got = O.soft_demodulate(
            [y], modulation="qpsk", noise_variance=var, gains=[h]
        )
        expect = direct_all(y / h, O.QPSK_GRAY, var / abs(h) ** 2)
        for g, e in zip(got.llrs, expect):
            self.assertAlmostEqual(g, e, places=12)

    def test_deep_fade_shrinks_llr(self):
        s = O.QPSK_GRAY[(0, 0)]
        var = 0.1
        strong = O.soft_demodulate(
            [s], modulation="qpsk", noise_variance=var, gains=[1 + 0j]
        )
        faded = O.soft_demodulate(
            [1e-3 * s], modulation="qpsk", noise_variance=var, gains=[1e-3 + 0j]
        )
        self.assertGreater(abs(strong.llrs[0]), 1.0)
        self.assertLess(abs(faded.llrs[0]), 1e-2)
        self.assertEqual(faded.erasures, (False,))

    def test_zero_gain_is_erasure(self):
        got = O.soft_demodulate(
            [3.2 + 1.0j], modulation="qpsk", noise_variance=0.1, gains=[0j]
        )
        self.assertEqual(got.llrs, (0.0, 0.0))
        self.assertEqual(got.erasures, (True,))

    def test_explicit_erasure_flag_preserved(self):
        got = O.soft_demodulate(
            [complex(SQRT1_2, SQRT1_2), complex(-SQRT1_2, SQRT1_2)],
            modulation="qpsk",
            noise_variance=0.1,
            erased=[True, False],
        )
        self.assertEqual(got.erasures, (True, False))
        self.assertEqual(got.llrs[:2], (0.0, 0.0))
        self.assertNotEqual(got.llrs[2:], (0.0, 0.0))

    def test_16qam_erasure_zeroes_all_four_bits(self):
        got = O.soft_demodulate(
            [1 + 1j], modulation="16qam", noise_variance=0.1, erased=[True]
        )
        self.assertEqual(got.llrs, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(got.erasures, (True,))


class TestSoftParameterValidation(unittest.TestCase):
    def test_noise_variance_must_be_positive_finite(self):
        for bad in (0.0, -0.5, float("nan"), float("inf"), True, "0.1"):
            with self.assertRaises(O.ParameterError):
                O.soft_demodulate([0j], noise_variance=bad)
        with self.assertRaises(O.ParameterError):
            O.decode(R.transmit(R.fixed_bits()), noise_variance=-1.0)

    def test_unknown_modulation_rejected(self):
        with self.assertRaises(O.ParameterError):
            O.soft_demodulate([0j], modulation="bpsk", noise_variance=0.1)

    def test_length_mismatch_rejected(self):
        with self.assertRaises(O.ParameterError):
            O.soft_demodulate([0j], noise_variance=0.1, gains=[1 + 0j, 1 + 0j])
        with self.assertRaises(O.ParameterError):
            O.soft_demodulate([0j], noise_variance=0.1, erased=[True, False])


class TestDecodeSoftIntegration(unittest.TestCase):
    def setUp(self):
        self.bits = R.fixed_bits()
        self.bits_str = "".join(str(b) for b in self.bits)

    def test_hard_output_unchanged_and_llr_signs_match(self):
        sig = R.transmit(
            self.bits, time_offset=17, cfo=0.11, gain=1.3 * cmath.exp(0.4j)
        )
        hard = O.decode(sig)
        soft = O.decode(sig, noise_variance=1e-3)
        self.assertEqual(soft.bits, hard.bits)
        self.assertEqual(soft.symbols, hard.symbols)
        self.assertEqual(soft.bits, self.bits_str)
        self.assertEqual(len(soft.llrs), 48)
        self.assertEqual(len(soft.erasures), 24)
        self.assertTrue(all(not e for e in soft.erasures))
        for v, b in zip(soft.llrs, soft.bits):
            if b == "0":
                self.assertGreater(v, 0.0)
            else:
                self.assertLess(v, 0.0)

    def test_llrs_none_without_noise_variance(self):
        result = O.decode(R.transmit(self.bits))
        self.assertIsNone(result.llrs)
        self.assertIsNone(result.erasures)

    def test_llrs_match_soft_demodulate_of_compensated_observations(self):
        gain = 0.9 * cmath.exp(-0.3j)
        sig = R.transmit(self.bits, time_offset=5, gain=gain)
        var = 0.02
        result = O.decode(sig, noise_variance=var)
        # 干净信道下补偿后观测等于判决符号；独立用公开软解调接口复核，
        # 方差按 σ²/|gain|² 同步调整（decode 内部使用估计增益）。
        soft = O.soft_demodulate(
            list(result.symbols),
            modulation="qpsk",
            noise_variance=var / abs(result.gain) ** 2,
        )
        for got, expect in zip(result.llrs, soft.llrs):
            self.assertAlmostEqual(got, expect, places=6)


if __name__ == "__main__":
    unittest.main()
