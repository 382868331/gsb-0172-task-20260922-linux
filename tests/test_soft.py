"""软解调（LLR）必要测试：与独立直接概率求和实现对照。

期望值来源：本文件内的 ``_direct_llr`` —— 不调用被测核心的 log-sum-exp，
而是按定义直接计算两组星座点的概率和 P0/P1 再取 log，
作为高精度对照（math.fsum 求和、逐点 math.exp）。
覆盖：星座中心、判决中点、极低信噪比、深衰落、零增益擦除、
16QAM 全表、端到端 decode_soft 及输入校验。
"""
from __future__ import annotations

import math
import unittest

import _bootstrap  # noqa: F401  路径引导

import ofdm_recovery as O
import reference as R

BITS = R.fixed_bits()
BITS_STR = "".join(str(b) for b in BITS)
SQRT1_2 = 1.0 / math.sqrt(2.0)


def _direct_llr(y, sigma2, table):
    """独立对照：直接概率求和，不用 log-sum-exp。

    LLR_i = log( Σ_{s: b_i=0} exp(-|y-s|²/σ²) / Σ_{s: b_i=1} exp(...) )
    """
    weights = [
        math.exp(-((y.real - s.real) ** 2 + (y.imag - s.imag) ** 2) / sigma2)
        for _, s in table
    ]
    nbits = len(table[0][0])
    out = []
    for i in range(nbits):
        p0 = math.fsum(w for (bits, _), w in zip(table, weights) if bits[i] == 0)
        p1 = math.fsum(w for (bits, _), w in zip(table, weights) if bits[i] == 1)
        out.append(math.log(p0) - math.log(p1))
    return out


class TestGrayTables(unittest.TestCase):
    def test_qpsk_table_matches_hard_mapping(self):
        # 显式 Gray 表与硬判决映射一致：位 0 -> 正坐标，实部在前。
        self.assertEqual(
            dict(O.QPSK_GRAY),
            {
                (0, 0): complex(SQRT1_2, SQRT1_2),
                (0, 1): complex(SQRT1_2, -SQRT1_2),
                (1, 1): complex(-SQRT1_2, -SQRT1_2),
                (1, 0): complex(-SQRT1_2, SQRT1_2),
            },
        )

    def test_qam16_table_is_gray_and_unit_energy(self):
        table = dict(O.QAM16_GRAY)
        self.assertEqual(len(table), 16)
        # 相邻电平只差 1 位（Gray）：实轴对 (b0,b1)、虚轴对 (b2,b3)。
        axis = {(0, 0): 3.0, (0, 1): 1.0, (1, 1): -1.0, (1, 0): -3.0}
        inv = 1.0 / math.sqrt(10.0)
        for (b0, b1, b2, b3), s in table.items():
            self.assertAlmostEqual(s.real, axis[(b0, b1)] * inv)
            self.assertAlmostEqual(s.imag, axis[(b2, b3)] * inv)
        # 平均能量为 1。
        energy = math.fsum(abs(s) ** 2 for s in table.values()) / 16.0
        self.assertAlmostEqual(energy, 1.0)


class TestSoftDemodAgainstDirect(unittest.TestCase):
    def _check(self, y, sigma2, modulation, table):
        got = O.soft_demod(y, sigma2, modulation=modulation)
        want = _direct_llr(complex(y), sigma2, table)
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, places=9)

    def test_qpsk_constellation_center(self):
        # 接收点恰在星座中心：LLR 大且符号正确（正坐标 -> 位 0 为正 LLR）。
        for bits, s in O.QPSK_GRAY:
            got = O.soft_demod(s, 0.05)
            for g, b in zip(got, bits):
                self.assertGreater(abs(g), 10.0)
                self.assertEqual(g > 0, b == 0)
        self._check(O.QPSK_GRAY[0][1], 0.05, "qpsk", O.QPSK_GRAY)

    def test_qpsk_decision_midpoint(self):
        # 判决边界中点：实部为 0 -> 位 0 的 LLR 精确为 0。
        got = O.soft_demod(complex(0.0, SQRT1_2), 0.1)
        self.assertEqual(got[0], 0.0)
        self.assertGreater(got[1], 1.0)
        self._check(complex(0.0, 0.2), 0.3, "qpsk", O.QPSK_GRAY)

    def test_very_low_snr_llr_shrinks_to_zero(self):
        # 极低信噪比（噪声方差极大）：LLR 趋近 0 但不硬判。
        got = O.soft_demod(complex(SQRT1_2, -SQRT1_2), 1e6)
        for g in got:
            self.assertLess(abs(g), 1e-5)
        self._check(complex(0.3, -0.4), 1e4, "qpsk", O.QPSK_GRAY)

    def test_qam16_all_points_and_midpoints(self):
        for bits, s in O.QAM16_GRAY:
            got = O.soft_demod(s, 0.02, modulation="qam16")
            for g, b in zip(got, bits):
                self.assertGreater(abs(g), 5.0)
                self.assertEqual(g > 0, b == 0)
        # 对称轴 re=0：符号位 b0 的 LLR 精确为 0；
        # 幅度位 b1（外圈 ±3 vs 内圈 ±1）在原点附近强烈偏向内圈（位 1）。
        got = O.soft_demod(complex(0.0, 0.3), 0.05, modulation="qam16")
        self.assertAlmostEqual(got[0], 0.0, places=12)
        self.assertLess(got[1], -5.0)
        # 相邻电平中点（+1/+3 之间）：LLR 接近 0（软值在边界附近趋近 0）。
        mid = complex(2.0 / math.sqrt(10.0), 0.0)
        got = O.soft_demod(mid, 0.05, modulation="qam16")
        self.assertLess(abs(got[1]), 0.01)
        self.assertGreater(abs(got[0]), 1.0)  # 远离该位边界
        self._check(mid, 0.05, "qam16", O.QAM16_GRAY)
        self._check(complex(0.1, -0.9), 0.4, "qam16", O.QAM16_GRAY)


class TestEqualizedSoftDemod(unittest.TestCase):
    def test_noise_variance_scales_with_gain(self):
        # 信道补偿后噪声方差按 |gain|² 同步调整：
        # soft_demod_equalized(y, g, σ²) == soft_demod(y/g, σ²/|g|²)。
        y, g, s2 = complex(1.3, -0.7), complex(0.5, 0.5), 0.08
        llrs, erased = O.soft_demod_equalized(y, g, s2)
        self.assertFalse(erased)
        want = O.soft_demod(y / g, s2 / (abs(g) ** 2))
        for a, b in zip(llrs, want):
            self.assertAlmostEqual(a, b, places=12)

    def test_deep_fade_llr_near_zero_but_not_erased(self):
        # 深衰落：增益极小 -> 等效噪声巨大 -> LLR≈0，但不是擦除。
        llrs, erased = O.soft_demod_equalized(
            complex(1e-6, 1e-6), 1e-6 + 0j, 0.5
        )
        self.assertFalse(erased)
        for v in llrs:
            self.assertLess(abs(v), 1e-3)

    def test_zero_gain_is_erasure_not_hard_zero(self):
        # 零增益 -> 擦除：LLR 全 0 且置标志，不能硬判成全 0 位。
        for mod, nbits in (("qpsk", 2), ("qam16", 4)):
            llrs, erased = O.soft_demod_equalized(
                complex(0.9, -0.9), 0j, 0.1, modulation=mod
            )
            self.assertTrue(erased)
            self.assertEqual(llrs, [0.0] * nbits)


class TestSoftInputValidation(unittest.TestCase):
    def test_noise_var_must_be_positive_finite_real(self):
        for bad in (0.0, -0.5, float("nan"), float("inf"), True, "0.1"):
            with self.assertRaises(O.ParameterError):
                O.soft_demod(0.5 + 0.5j, bad)
            with self.assertRaises(O.ParameterError):
                O.soft_demod_equalized(0.5 + 0.5j, 1 + 0j, bad)

    def test_unknown_modulation_rejected(self):
        for bad in ("bpsk", "", True, 16):
            with self.assertRaises(O.ParameterError):
                O.soft_demod(0.5 + 0.5j, 0.1, modulation=bad)

    def test_nonfinite_symbol_rejected(self):
        with self.assertRaises(O.SignalFormatError):
            O.soft_demod(complex(float("nan"), 0.0), 0.1)


class TestDecodeSoft(unittest.TestCase):
    def test_end_to_end_qpsk_llrs_match_bits_and_order(self):
        sig = R.transmit(
            BITS, time_offset=23, cfo=0.137, gain=1.7,
            front_scale=0.05, inband_scale=0.05, trailing=25, seed=42,
        )
        hard = O.decode(sig)
        soft = O.decode_soft(sig, 0.02)
        # 硬判决输出保持原位序，与 decode 完全一致。
        self.assertEqual(soft.bits, hard.bits)
        self.assertEqual(soft.symbols, hard.symbols)
        self.assertEqual(soft.bits, BITS_STR)
        # 48 个 LLR（只含数据载波；导频/DC/保护带不混入），无擦除。
        self.assertEqual(len(soft.llrs), 48)
        self.assertEqual(len(soft.erasures), 24)
        self.assertFalse(any(soft.erasures))
        # 低噪声下每位 LLR 符号与硬判决位一致（正值偏向位 0）。
        for llr, b in zip(soft.llrs, BITS):
            self.assertEqual(llr > 0, b == 0)

    def test_decode_soft_qam16_length(self):
        sig = R.transmit(BITS, time_offset=5, cfo=-0.1, gain=1.2, seed=7)
        soft = O.decode_soft(sig, 0.05, modulation="qam16")
        self.assertEqual(len(soft.llrs), 96)
        self.assertEqual(len(soft.erasures), 24)
        self.assertEqual(soft.bits, BITS_STR)  # 硬判决仍是 QPSK 位串

    def test_decode_soft_rejects_bad_noise_var(self):
        sig = R.transmit(BITS)
        with self.assertRaises(O.ParameterError):
            O.decode_soft(sig, 0.0)
        with self.assertRaises(O.ParameterError):
            O.decode_soft(sig, -1.0)


if __name__ == "__main__":
    unittest.main()
