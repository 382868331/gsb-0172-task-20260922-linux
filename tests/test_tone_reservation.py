"""保留子载波 PAPR 压低的必要测试。

独立核对方式：本文件自带朴素 DFT/IDFT 与 PAPR 计算（不调用被测核心的
dft/_idft/_papr_linear），对返回的时域信号重新变换，独立验证频域约束
（数据/导频/DC 未变、保留能量在预算内）与峰均比历史。
"""
from __future__ import annotations

import cmath
import math
import unittest

import _bootstrap  # noqa: F401  路径引导

import ofdm_recovery as O

N = 32
SQRT1_2 = 1.0 / math.sqrt(2.0)
DATA_IDX = (1, 2, 3, 4, 5, 6, 26, 27, 28, 29, 30, 31)
PILOT_IDX = (7, 25)


def ref_dft(x):
    n = len(x)
    return [
        sum((x[m] * cmath.exp(-2j * math.pi * k * m / n) for m in range(n)), 0j)
        for k in range(n)
    ]


def ref_papr_db(x):
    powers = [abs(v) ** 2 for v in x]
    return 10.0 * math.log10(max(powers) / (sum(powers) / len(powers)))


def peaky_symbols():
    """确定性、有明显峰值的符号串：多数载波同相，在 n=0 附近叠出尖峰。"""
    q = SQRT1_2
    signs = [1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1]
    return [complex(q * s, q * s) for s in signs]


def expected_spectrum(symbols):
    """按固定布局独立构造期望频域（数据/导频/DC 部分）。"""
    X = [0j] * N
    carriers = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
    for j, kc in enumerate(carriers):
        X[kc % N] = symbols[j]
    X[7] = 1 + 0j
    X[25] = -1 + 0j
    return X


class TestReservedSetValidation(unittest.TestCase):
    def test_empty_reserved_set_rejected(self):
        with self.assertRaises(O.ReservedCarrierError) as cm:
            O.reduce_papr_tone_reservation(
                peaky_symbols(), [], target_papr_db=4.0, max_reserved_energy=1.0
            )
        self.assertIn("empty", str(cm.exception))

    def test_overlap_with_data_rejected(self):
        with self.assertRaises(O.ReservedCarrierError) as cm:
            O.reduce_papr_tone_reservation(
                peaky_symbols(), [8, 3], target_papr_db=4.0,
                max_reserved_energy=1.0,
            )
        self.assertIn("data", str(cm.exception))

    def test_overlap_with_pilot_rejected(self):
        for k in (7, 25):
            with self.assertRaises(O.ReservedCarrierError) as cm:
                O.reduce_papr_tone_reservation(
                    peaky_symbols(), [k], target_papr_db=4.0,
                    max_reserved_energy=1.0,
                )
            self.assertIn("pilot", str(cm.exception))

    def test_overlap_with_dc_rejected(self):
        with self.assertRaises(O.ReservedCarrierError) as cm:
            O.reduce_papr_tone_reservation(
                peaky_symbols(), [0], target_papr_db=4.0,
                max_reserved_energy=1.0,
            )
        self.assertIn("DC", str(cm.exception))

    def test_bad_index_type_range_duplicate(self):
        for bad in ([True], [8.5], [-1], [32], [9, 9]):
            with self.assertRaises(O.ReservedCarrierError):
                O.reduce_papr_tone_reservation(
                    peaky_symbols(), bad, target_papr_db=4.0,
                    max_reserved_energy=1.0,
                )

    def test_bad_scalar_parameters(self):
        with self.assertRaises(O.ParameterError):
            O.reduce_papr_tone_reservation(
                peaky_symbols(), [8], target_papr_db=float("nan"),
                max_reserved_energy=1.0,
            )
        with self.assertRaises(O.ParameterError):
            O.reduce_papr_tone_reservation(
                peaky_symbols(), [8], target_papr_db=4.0,
                max_reserved_energy=-0.5,
            )
        with self.assertRaises(O.ParameterError):
            O.reduce_papr_tone_reservation(
                peaky_symbols(), [8], target_papr_db=4.0,
                max_reserved_energy=1.0, max_iterations=True,
            )
        with self.assertRaises(O.SignalFormatError):
            O.reduce_papr_tone_reservation(
                peaky_symbols()[:5], [8], target_papr_db=4.0,
                max_reserved_energy=1.0,
            )


class TestZeroBudget(unittest.TestCase):
    """全部允许预算为 0：保留音调必须为 0，载荷不变，如实报告未达标。"""

    def test_zero_budget_no_change(self):
        symbols = peaky_symbols()
        res = O.reduce_papr_tone_reservation(
            symbols, [8, 12, 16, 20, 24],
            target_papr_db=3.0, max_reserved_energy=0.0,
        )
        self.assertFalse(res.achieved)
        self.assertEqual(res.reserved_energy, 0.0)
        self.assertTrue(all(e == 0.0 for e in res.reserved_energy_history))
        # 独立 DFT 复核：频域与未处理布局完全一致（保留频点也是 0）。
        check = ref_dft(res.time_signal)
        want = expected_spectrum(symbols)
        for k in range(N):
            self.assertAlmostEqual(check[k].real, want[k].real, places=9)
            self.assertAlmostEqual(check[k].imag, want[k].imag, places=9)
        # PAPR 历史每项都等于初始 PAPR（没有任何改善）。
        init_db = ref_papr_db(res.time_signal)
        for p in res.papr_history:
            self.assertAlmostEqual(10.0 * math.log10(p), init_db, places=9)


class TestPeakReduction(unittest.TestCase):
    def test_peaky_sequence_reduced_and_payload_intact(self):
        symbols = peaky_symbols()
        reserved = list(range(8, 25))
        budget = 8.0
        res = O.reduce_papr_tone_reservation(
            symbols, reserved, target_papr_db=4.0,
            max_reserved_energy=budget,
        )
        self.assertTrue(res.achieved)
        self.assertEqual(res.iterations, len(res.papr_history) - 1)
        self.assertEqual(len(res.papr_history), len(res.reserved_energy_history))
        # 独立 PAPR 复核：最终时域峰均比确实 <= 目标（4 dB）。
        final_db = ref_papr_db(res.time_signal)
        self.assertLessEqual(final_db, 4.0 + 1e-4)
        self.assertLess(final_db, ref_papr_db(
            # 未处理参考：同一布局直接 IDFT。
            [sum(expected_spectrum(symbols)[k]
                  * cmath.exp(2j * math.pi * k * n / N) for k in range(N)) / N
             for n in range(N)]
        ))
        # 独立 DFT 复核：数据/导频/DC 频点与输入完全一致。
        check = ref_dft(res.time_signal)
        want = expected_spectrum(symbols)
        for k in list(DATA_IDX) + list(PILOT_IDX) + [0]:
            self.assertAlmostEqual(check[k].real, want[k].real, places=9)
            self.assertAlmostEqual(check[k].imag, want[k].imag, places=9)
        # 保留频点确实被使用，且总能量在预算内。
        reserved_energy = sum(abs(check[k]) ** 2 for k in reserved)
        self.assertGreater(reserved_energy, 0.0)
        self.assertLessEqual(reserved_energy, budget * (1.0 + 1e-9))
        self.assertAlmostEqual(res.reserved_energy, reserved_energy, places=9)
        # 返回的 spectrum 字段与独立 DFT 一致。
        for k in range(N):
            self.assertAlmostEqual(res.spectrum[k].real, check[k].real, places=9)
            self.assertAlmostEqual(res.spectrum[k].imag, check[k].imag, places=9)

    def test_unachievable_target_reported_honestly(self):
        # 5 个保留音调、预算 4.0、目标 3 dB：自由度不够，必须如实报告。
        res = O.reduce_papr_tone_reservation(
            peaky_symbols(), [8, 12, 16, 20, 24],
            target_papr_db=3.0, max_reserved_energy=4.0,
        )
        self.assertFalse(res.achieved)
        self.assertGreater(res.papr_history[-1], res.target_papr)
        self.assertLessEqual(res.reserved_energy, 4.0 * (1.0 + 1e-9))
        # 即便如此，数据频点也不能被破坏。
        check = ref_dft(res.time_signal)
        want = expected_spectrum(peaky_symbols())
        for k in DATA_IDX:
            self.assertAlmostEqual(check[k].real, want[k].real, places=9)
            self.assertAlmostEqual(check[k].imag, want[k].imag, places=9)

    def test_already_below_target_no_iteration(self):
        # 目标宽松（20 dB）：初始即达标，0 轮迭代，保留能量为 0。
        res = O.reduce_papr_tone_reservation(
            peaky_symbols(), [8, 16, 24],
            target_papr_db=20.0, max_reserved_energy=1.0,
        )
        self.assertTrue(res.achieved)
        self.assertEqual(res.iterations, 0)
        self.assertEqual(res.reserved_energy, 0.0)
        self.assertEqual(len(res.papr_history), 1)


class TestReceiverUnaffected(unittest.TestCase):
    """保留子载波是发射端可选功能：接收 decode 接口与结果不变。"""

    def test_decode_still_default_off(self):
        import reference as R

        bits = R.fixed_bits()
        sig = R.transmit(bits, time_offset=17, cfo=0.1, gain=1.3 * cmath.exp(0.2j))
        result = O.decode(sig)
        self.assertEqual(result.bits, "".join(str(b) for b in bits))
        self.assertEqual(result.start, 17)


if __name__ == "__main__":
    unittest.main()

