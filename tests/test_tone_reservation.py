"""保留子载波压峰（tone reservation）的必要测试。

独立核对方式：本文件自带朴素直接 DFT/IDFT（不调用被测核心的 dft/_idft），
对返回的时域符号重新变换，核对数据/导频/DC 频点未变、保留能量不超标、
报告峰均比与实测一致。符号串与期望值均为确定性构造。
"""
from __future__ import annotations

import cmath
import math
import unittest

import _bootstrap  # noqa: F401  路径引导

import ofdm_recovery as O

N = 32
SQRT1_2 = 1.0 / math.sqrt(2.0)


def naive_dft(x):
    """独立朴素 DFT（本测试文件自有实现，不导入被测核心）。"""
    n = len(x)
    return [
        sum(x[m] * cmath.exp(-2j * math.pi * k * m / n) for m in range(n))
        for k in range(n)
    ]


def naive_idft(X):
    n = len(X)
    return [
        sum(X[k] * cmath.exp(2j * math.pi * k * m / n) for k in range(n)) / n
        for m in range(n)
    ]


def measured_papr_db(x):
    powers = [v.real * v.real + v.imag * v.imag for v in x]
    mean = sum(powers) / len(powers)
    if mean == 0.0:
        return 0.0
    return 10.0 * math.log10(max(powers) / mean)


def peaky_spectrum():
    """确定性“明显峰值”频域符号：12 个数据载波全部同相 (1+j)/√2。

    同相多载波在时域形成明显相干峰，PAPR 显著高于随机符号串。
    """
    X = [0j] * N
    for k in O.DATA_CARRIERS:
        X[k] = complex(SQRT1_2, SQRT1_2)
    X[7] = 1 + 0j     # +7 导频
    X[25] = -1 + 0j   # -7 导频
    return X


RESERVED = (8, 9, 10, 11, 12, 20, 21, 22, 23, 24)


class TestReservedSetValidation(unittest.TestCase):
    def test_overlap_with_data_rejected(self):
        for k in (1, 6, 26, 31):
            with self.assertRaises(O.ReservedCarrierError):
                O.reduce_papr(
                    peaky_spectrum(), [k],
                    target_papr_db=4.0, energy_budget=1.0,
                )

    def test_overlap_with_pilot_rejected(self):
        for k in O.PILOT_CARRIERS:
            with self.assertRaises(O.ReservedCarrierError) as cm:
                O.reduce_papr(
                    peaky_spectrum(), [k],
                    target_papr_db=4.0, energy_budget=1.0,
                )
            self.assertIn("pilot", str(cm.exception))

    def test_dc_rejected(self):
        with self.assertRaises(O.ReservedCarrierError) as cm:
            O.reduce_papr(
                peaky_spectrum(), [0],
                target_papr_db=4.0, energy_budget=1.0,
            )
        self.assertIn("DC", str(cm.exception))

    def test_duplicate_out_of_range_and_type_rejected(self):
        with self.assertRaises(O.ReservedCarrierError):
            O.reduce_papr(
                peaky_spectrum(), [9, 9],
                target_papr_db=4.0, energy_budget=1.0,
            )
        for bad in (-1, 32, 100):
            with self.assertRaises(O.ReservedCarrierError):
                O.reduce_papr(
                    peaky_spectrum(), [bad],
                    target_papr_db=4.0, energy_budget=1.0,
                )
        with self.assertRaises(O.ReservedCarrierError):
            O.reduce_papr(
                peaky_spectrum(), [True],
                target_papr_db=4.0, energy_budget=1.0,
            )
        with self.assertRaises(O.ReservedCarrierError):
            O.reduce_papr(
                peaky_spectrum(), [9.5],
                target_papr_db=4.0, energy_budget=1.0,
            )
        with self.assertRaises(O.ReservedCarrierError):
            O.reduce_papr(
                peaky_spectrum(), 9,
                target_papr_db=4.0, energy_budget=1.0,
            )

    def test_scalar_parameter_validation(self):
        spec = peaky_spectrum()
        with self.assertRaises(O.ParameterError):
            O.reduce_papr(spec, RESERVED, target_papr_db=float("nan"),
                          energy_budget=1.0)
        with self.assertRaises(O.ParameterError):
            O.reduce_papr(spec, RESERVED, target_papr_db=True,
                          energy_budget=1.0)
        with self.assertRaises(O.ParameterError):
            O.reduce_papr(spec, RESERVED, target_papr_db=4.0,
                          energy_budget=-0.5)
        with self.assertRaises(O.ParameterError):
            O.reduce_papr(spec, RESERVED, target_papr_db=4.0,
                          energy_budget=float("inf"))
        with self.assertRaises(O.ParameterError):
            O.reduce_papr(spec, RESERVED, target_papr_db=4.0,
                          energy_budget=1.0, max_iterations=0)
        with self.assertRaises(O.ParameterError):
            O.reduce_papr(spec, RESERVED, target_papr_db=4.0,
                          energy_budget=1.0, max_iterations=True)

    def test_spectrum_format_validation(self):
        with self.assertRaises(O.SignalFormatError):
            O.reduce_papr([0j] * 31, RESERVED,
                          target_papr_db=4.0, energy_budget=1.0)
        with self.assertRaises(O.SignalFormatError):
            O.reduce_papr("not a spectrum", RESERVED,
                          target_papr_db=4.0, energy_budget=1.0)
        bad = [0j] * N
        bad[3] = float("nan")
        with self.assertRaises(O.SignalFormatError) as cm:
            O.reduce_papr(bad, RESERVED,
                          target_papr_db=4.0, energy_budget=1.0)
        self.assertIn("spectrum[3]", str(cm.exception))


class TestEmptyAndZeroBudget(unittest.TestCase):
    """空保留集合与零能量预算：无自由度，必须原样返回并如实报告。"""

    def test_empty_reserved_set_returns_input_unchanged(self):
        spec = peaky_spectrum()
        result = O.reduce_papr(
            spec, [], target_papr_db=1.0, energy_budget=1.0
        )
        self.assertFalse(result.target_reached)
        self.assertEqual(len(result.iterations), 1)  # 仅初始轮
        self.assertEqual(result.iterations[0].iteration, 0)
        self.assertEqual(result.reserved_energy, 0.0)
        for got, want in zip(result.spectrum, spec):
            self.assertEqual(got, want)
        # 独立 IDFT 核对时域与频域互逆。
        expect_time = naive_idft(spec)
        for got, want in zip(result.time_domain, expect_time):
            self.assertAlmostEqual(got.real, want.real, places=12)
            self.assertAlmostEqual(got.imag, want.imag, places=12)

    def test_zero_budget_returns_input_unchanged(self):
        spec = peaky_spectrum()
        result = O.reduce_papr(
            spec, RESERVED, target_papr_db=1.0, energy_budget=0.0
        )
        self.assertFalse(result.target_reached)
        self.assertEqual(len(result.iterations), 1)
        self.assertEqual(result.reserved_energy, 0.0)
        for got, want in zip(result.spectrum, spec):
            self.assertEqual(got, want)
        # 初始 PAPR 就是最终 PAPR。
        self.assertAlmostEqual(
            result.papr_db,
            measured_papr_db(naive_idft(spec)),
            places=12,
        )


class TestToneReservationReduction(unittest.TestCase):
    def setUp(self):
        self.spec = peaky_spectrum()
        self.initial_papr = measured_papr_db(naive_idft(self.spec))
        # 构造前提：该确定性符号串必须有明显峰值。
        self.assertGreater(self.initial_papr, 6.0)

    def test_reduction_reaches_target_and_reports_iterations(self):
        result = O.reduce_papr(
            self.spec, RESERVED,
            target_papr_db=6.0, energy_budget=6.0, max_iterations=64,
        )
        self.assertTrue(result.target_reached)
        self.assertLessEqual(result.papr_db, 6.0)
        self.assertLess(result.papr_db, self.initial_papr)
        # 每轮记录：初始轮 + 至少一轮调整，PAPR 与能量均为有限值。
        self.assertGreater(len(result.iterations), 1)
        for rec in result.iterations:
            self.assertTrue(math.isfinite(rec.papr_db))
            self.assertTrue(math.isfinite(rec.reserved_energy))
        # 最终结果取自历史最优一轮。
        self.assertEqual(
            result.papr_db,
            min(rec.papr_db for rec in result.iterations),
        )

    def test_frequency_domain_constraints_verified_independently(self):
        result = O.reduce_papr(
            self.spec, RESERVED,
            target_papr_db=6.0, energy_budget=6.0, max_iterations=64,
        )
        # 独立直接 DFT 重新变换返回的时域符号。
        check = naive_dft(result.time_domain)
        ref = max(abs(v) for v in self.spec)
        for k in range(N):
            if k in RESERVED:
                continue
            # 数据、导频、DC 与其余零载波全部不得改变。
            self.assertAlmostEqual(check[k].real, self.spec[k].real,
                                   delta=1e-9 * ref)
            self.assertAlmostEqual(check[k].imag, self.spec[k].imag,
                                   delta=1e-9 * ref)
        # 返回频谱与独立核对一致，且保留频点确实被写入。
        moved = 0
        for k in RESERVED:
            self.assertAlmostEqual(result.spectrum[k].real, check[k].real,
                                   delta=1e-9 * ref)
            self.assertAlmostEqual(result.spectrum[k].imag, check[k].imag,
                                   delta=1e-9 * ref)
            if abs(check[k]) > 1e-6:
                moved += 1
        self.assertGreater(moved, 0)

    def test_energy_budget_respected(self):
        budget = 6.0
        result = O.reduce_papr(
            self.spec, RESERVED,
            target_papr_db=6.0, energy_budget=budget, max_iterations=64,
        )
        energy = sum(
            abs(result.spectrum[k]) ** 2 for k in RESERVED
        )
        self.assertLessEqual(energy, budget * (1.0 + 1e-9))
        self.assertAlmostEqual(energy, result.reserved_energy, places=9)
        for rec in result.iterations:
            self.assertLessEqual(rec.reserved_energy, budget * (1.0 + 1e-9))

    def test_reported_papr_matches_independent_measurement(self):
        result = O.reduce_papr(
            self.spec, RESERVED,
            target_papr_db=6.0, energy_budget=6.0, max_iterations=64,
        )
        self.assertAlmostEqual(
            result.papr_db,
            measured_papr_db(result.time_domain),
            places=9,
        )

    def test_impossible_target_reports_not_reached_without_exception(self):
        # PAPR 物理下界是 0 dB；目标 -1 dB 必然达不到，只报告不抛错。
        result = O.reduce_papr(
            self.spec, RESERVED,
            target_papr_db=-1.0, energy_budget=6.0, max_iterations=8,
        )
        self.assertFalse(result.target_reached)
        self.assertGreaterEqual(result.papr_db, 0.0)
        # 数据频点仍未被触碰。
        check = naive_dft(result.time_domain)
        for k in O.DATA_CARRIERS:
            self.assertAlmostEqual(check[k].real, self.spec[k].real,
                                   delta=1e-9)
            self.assertAlmostEqual(check[k].imag, self.spec[k].imag,
                                   delta=1e-9)

    def test_deterministic_repeatability(self):
        a = O.reduce_papr(self.spec, RESERVED, target_papr_db=6.0,
                          energy_budget=6.0)
        b = O.reduce_papr(self.spec, RESERVED, target_papr_db=6.0,
                          energy_budget=6.0)
        self.assertEqual(a, b)


class TestZeroSpectrum(unittest.TestCase):
    def test_zero_spectrum_papr_defined_as_zero_db(self):
        result = O.reduce_papr(
            [0j] * N, RESERVED, target_papr_db=0.5, energy_budget=1.0
        )
        self.assertEqual(result.papr_db, 0.0)
        self.assertTrue(result.target_reached)


if __name__ == "__main__":
    unittest.main()
