"""本次迭代测试：最多 3 抽头固定复数 FIR 信道的估计、均衡与深零点擦除。

期望值来源：
* 真实抽头/增益/频偏/时偏 = ``tests/reference.py`` 的**输入参数**；
* 信道频响 ``H(k)=Σ_l h_l exp(-j2πkl/N)`` 与零点位置由测试内独立解析式
  手算给出（不调用被测实现反推）；
* 位串来自 ``reference.fixed_bits()``。
旧行为（L=1、平坦增益、候选搜索、各类拒绝）仍由 ``test_ofdm.py`` 覆盖。
"""
from __future__ import annotations

import cmath
import dataclasses
import math
import unittest

import _bootstrap  # noqa: F401  路径引导

import ofdm_recovery as O
import reference as R

BITS = R.fixed_bits()
BITS_STR = "".join(str(b) for b in BITS)
CARRIERS = list(range(-6, 0)) + list(range(1, 7))
SQRT1_2 = 1.0 / math.sqrt(2.0)
GAIN = 1.7 * cmath.exp(0.6j)


def analytic_response(taps, kc):
    """独立手算：H(k)=Σ_l taps[l]·exp(-j2π k l /32)。"""
    return sum(
        complex(taps[l]) * cmath.exp(-2j * math.pi * kc * l / 32)
        for l in range(len(taps))
    )


def channel_ratio(taps, kc, gain=GAIN):
    """|H(k)|/sqrt(Σ|h_l|²)；整体增益在比值中约去。"""
    h = [gain * t for t in taps]
    rms = math.sqrt(sum(abs(v) ** 2 for v in h))
    return abs(analytic_response(h, kc)) / rms


def nonerased_match(result):
    return all(
        result.bits[i] == BITS_STR[i]
        for i in range(48)
        if result.bits[i] != "x"
    )


class TestCleanFir(unittest.TestCase):
    def test_single_tap_is_flat_gain(self):
        sig = R.transmit(BITS, time_offset=12, cfo=0.1, gain=GAIN, taps=(1,))
        result = O.decode(sig, channel_taps=1)
        self.assertEqual(result.start, 12)
        self.assertAlmostEqual(result.frequency_offset, 0.1, places=10)
        self.assertAlmostEqual(abs(result.gain - GAIN), 0.0, places=10)
        self.assertEqual(len(result.channel), 1)
        self.assertEqual(result.erased, ())
        self.assertEqual(result.bits, BITS_STR)

    def test_default_three_tap_model_on_flat_channel(self):
        # 不显式给抽头数：默认按 3 抽头估计，平坦信道下后两抽头应为 0。
        sig = R.transmit(BITS, time_offset=31, cfo=0.137, gain=GAIN,
                         trailing=19)
        result = O.decode(sig)
        self.assertEqual(result.start, 31)
        self.assertEqual(len(result.channel), 3)
        self.assertAlmostEqual(abs(result.channel[0] - GAIN), 0.0, places=10)
        self.assertLess(abs(result.channel[1]), 1e-9)
        self.assertLess(abs(result.channel[2]), 1e-9)
        self.assertEqual(result.gain, result.channel[0])
        self.assertEqual(result.erased, ())
        self.assertEqual(result.bits, BITS_STR)

    def test_two_tap_mild_distortion_exact_recovery(self):
        taps = (1 + 0j, 0.2 - 0.1j)
        sig = R.transmit(BITS, time_offset=23, cfo=0.137, gain=GAIN,
                         taps=taps)
        result = O.decode(sig)
        self.assertEqual(result.start, 23)
        self.assertAlmostEqual(result.frequency_offset, 0.137, places=10)
        self.assertLess(result.residual, 1e-9)
        for l in range(3):
            truth = GAIN * taps[l] if l < 2 else 0j
            self.assertLess(abs(result.channel[l] - truth), 1e-9)
        self.assertEqual(result.erased, ())
        self.assertEqual(result.bits, BITS_STR)

    def test_three_tap_with_zero_middle(self):
        taps = (1 + 0j, 0j, 0.3 + 0.2j)
        sig = R.transmit(BITS, time_offset=9, cfo=-0.11, gain=GAIN,
                         taps=taps)
        result = O.decode(sig)
        self.assertEqual(result.start, 9)
        for l, t in enumerate(taps):
            self.assertLess(abs(result.channel[l] - GAIN * t), 1e-9)
        self.assertAlmostEqual(result.frequency_offset, -0.11, places=10)
        self.assertEqual(result.bits, BITS_STR)
        self.assertEqual(result.erased, ())

    def test_weak_first_tap_recovered_via_safe_window(self):
        # 后主径主导的 3 抽头：全 16 点窗在真起点的相干系数跌破默认门限
        # （旧 find_candidates 会漏检真起点），但瞬态安全窗仍可精确定位。
        taps = (0.63 - 0.28j, 1.03 - 0.42j, 1.45 - 0.13j)
        sig = R.transmit(BITS, time_offset=32, taps=taps)
        full = {c.start for c in O.find_candidates(sig)}
        self.assertNotIn(32, full)  # 全窗门限确实会漏掉真起点
        result = O.decode(sig)
        self.assertEqual(result.start, 32)
        for l, t in enumerate(taps):
            self.assertLess(abs(result.channel[l] - t), 1e-9)
        self.assertEqual(result.bits, BITS_STR)


class TestDeepNullErasure(unittest.TestCase):
    TAPS = (1 + 0j, -1 + 0j)  # H(k)=1-exp(-j2πk/32)，±1 处为深衰落

    def test_analytic_ratios_match_erasure_set(self):
        # 独立手算各载波归一化幅度，验证默认门限 0.25 的判定。
        below = [
            kc for kc in CARRIERS
            if channel_ratio(self.TAPS, kc) < O.DEFAULT_ERASURE_THRESHOLD
        ]
        self.assertEqual(below, [-1, 1])
        others = [kc for kc in CARRIERS if kc not in (-1, 1)]
        self.assertTrue(
            all(
                channel_ratio(self.TAPS, kc) > O.DEFAULT_ERASURE_THRESHOLD
                for kc in others
            )
        )

        sig = R.transmit(BITS, time_offset=23, cfo=0.137, gain=GAIN,
                         taps=self.TAPS)
        result = O.decode(sig)
        self.assertEqual(result.erased, (-1, 1))
        self.assertTrue(nonerased_match(result))

    def test_erased_symbols_are_none_and_bits_marked_x(self):
        sig = R.transmit(BITS, time_offset=4, taps=self.TAPS)
        result = O.decode(sig)
        # 每个擦除载波在两个符号上各占一个符号位置；顺序为
        # 符号 0→1，每符号载波 -6..-1、+1..+6。
        expected_none = {
            s * 12 + CARRIERS.index(kc)
            for s in range(2)
            for kc in (-1, 1)
        }
        self.assertEqual(
            {i for i, s in enumerate(result.symbols) if s is None},
            expected_none,
        )
        for i, s in enumerate(result.symbols):
            if i not in expected_none:
                self.assertIsInstance(s, complex)
                self.assertAlmostEqual(abs(s), 1.0, places=10)
        # 位串：擦除符号对应两位 'x'，其余位与真值一致。
        erased_bit_positions = {
            2 * i + c for i in expected_none for c in (0, 1)
        }
        for i, ch in enumerate(result.bits):
            if i in erased_bit_positions:
                self.assertEqual(ch, "x")
            else:
                self.assertEqual(ch, BITS_STR[i])

    def test_exact_null_on_single_carrier_handbuilt(self):
        # 构造在 k=+3 处精确零点的 2 抽头：
        # H(3)=1 + a·exp(-j2π·3/32)=0  ->  a=-exp(+j2π·3/32)。
        a = -cmath.exp(2j * math.pi * 3 / 32)
        taps = (1 + 0j, a)
        self.assertAlmostEqual(abs(analytic_response(taps, 3)), 0.0,
                               places=12)
        sig = R.transmit(BITS, time_offset=15, taps=taps)
        result = O.decode(sig)
        self.assertIn(3, result.erased)
        self.assertNotIn(-3, result.erased)
        self.assertTrue(nonerased_match(result))
        for l, t in enumerate(taps):
            self.assertLess(abs(result.channel[l] - t), 1e-9)

    def test_erasure_threshold_zero_decodes_everything(self):
        sig = R.transmit(BITS, time_offset=7, taps=self.TAPS)
        result = O.decode(sig, erasure_threshold=0.0)
        self.assertEqual(result.erased, ())
        # 干净信号下即使深衰落载波也可恢复（只是噪声余量极小）。
        self.assertEqual(result.bits, BITS_STR)

    def test_erasure_threshold_one_erases_all_data_carriers(self):
        sig = R.transmit(BITS, time_offset=7, taps=self.TAPS)
        result = O.decode(sig, erasure_threshold=1.0)
        self.assertEqual(len(result.erased), 12)
        self.assertTrue(all(s is None for s in result.symbols))
        self.assertEqual(result.bits, "x" * 48)


class TestFirWithNoise(unittest.TestCase):
    NOISE = 0.05

    def test_notched_channel_multiple_seeds(self):
        for seed in (42, 99, 777, 1, 2):
            with self.subTest(seed=seed):
                sig = R.transmit(
                    BITS,
                    time_offset=23,
                    cfo=0.137,
                    gain=GAIN,
                    taps=(1 + 0j, -1 + 0j),
                    front_scale=self.NOISE,
                    inband_scale=self.NOISE,
                    trailing=20,
                    seed=seed,
                )
                result = O.decode(sig)
                self.assertEqual(result.start, 23)
                self.assertLess(abs(result.frequency_offset - 0.137), 0.02)
                # 擦除集合在噪声下稳定。
                self.assertEqual(result.erased, (-1, 1))
                self.assertTrue(nonerased_match(result))
                # 抽头幅度估计误差（噪声标准差约 0.041，抽头模约 1.7）。
                self.assertLess(
                    abs(result.channel[0] - GAIN) / abs(GAIN), 0.10
                )
                self.assertLess(
                    abs(result.channel[1] + GAIN) / abs(GAIN), 0.10
                )

    def test_three_tap_null_seed_stable(self):
        taps = (1 + 0j, 0j, -cmath.exp(1j * math.pi / 4))
        for seed in (42, 99, 777):
            with self.subTest(seed=seed):
                sig = R.transmit(
                    BITS,
                    time_offset=23,
                    cfo=0.137,
                    gain=GAIN,
                    taps=taps,
                    front_scale=self.NOISE,
                    inband_scale=self.NOISE,
                    trailing=20,
                    seed=seed,
                )
                result = O.decode(sig)
                self.assertEqual(result.start, 23)
                self.assertEqual(result.erased, (2,))
                self.assertTrue(nonerased_match(result))


class TestFirBoundaries(unittest.TestCase):
    def test_large_cfo_with_fir(self):
        sig = R.transmit(
            BITS, time_offset=6, cfo=-0.45, gain=0.5 - 0.5j,
            taps=(1 + 0j, -1 + 0j),
        )
        result = O.decode(sig)
        self.assertEqual(result.start, 6)
        self.assertAlmostEqual(result.frequency_offset, -0.45, places=9)
        self.assertEqual(result.erased, (-1, 1))
        self.assertTrue(nonerased_match(result))

    def test_explicit_two_tap_parameter(self):
        taps = (1 + 0j, 0.2 - 0.1j)
        sig = R.transmit(BITS, time_offset=3, gain=GAIN, taps=taps)
        result = O.decode(sig, channel_taps=2)
        self.assertEqual(len(result.channel), 2)
        for l, t in enumerate(taps):
            self.assertLess(abs(result.channel[l] - GAIN * t), 1e-9)
        self.assertEqual(result.bits, BITS_STR)

    def test_full_length_signal_with_fir(self):
        taps = (1 + 0j, -1 + 0j)
        sig = R.transmit(BITS, time_offset=400, cfo=0.08, taps=taps)
        self.assertEqual(len(sig), O.MAX_SIGNAL_LEN)
        result = O.decode(sig)
        self.assertEqual(result.start, 400)
        self.assertEqual(result.erased, (-1, 1))
        self.assertTrue(nonerased_match(result))

    def test_back_to_back_frames_earliest_wins_with_fir(self):
        frame = R.build_frame(BITS)
        sig = [0j] * 3 + frame + frame
        result = O.decode(sig)
        self.assertEqual(result.start, 3)
        self.assertTrue(nonerased_match(result))

    def test_result_is_immutable(self):
        sig = R.transmit(BITS, taps=(1 + 0j, -1 + 0j))
        result = O.decode(sig)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.start = 5  # type: ignore[misc]
        self.assertIsInstance(result.erased, tuple)
        self.assertIsInstance(result.channel, tuple)


class TestFirRejections(unittest.TestCase):
    def test_channel_taps_validation(self):
        sig = R.transmit(BITS)
        for bad in (0, 4, -1, 2.0, True, False):
            with self.assertRaises(O.ParameterError):
                O.decode(sig, channel_taps=bad)

    def test_erasure_threshold_validation(self):
        sig = R.transmit(BITS)
        for bad in (-0.1, 1.1, float("nan"), float("inf"), True):
            with self.assertRaises(O.ParameterError):
                O.decode(sig, erasure_threshold=bad)

    def test_truncation_still_rejected_under_fir_model(self):
        sig = R.transmit(
            BITS, time_offset=10, taps=(1 + 0j, -1 + 0j), truncate=45
        )
        with self.assertRaises(O.TruncatedFrameError):
            O.decode(sig)

    def test_zero_pilots_still_rejected_under_fir_model(self):
        sig = R.transmit(BITS, zero_pilots=True)
        with self.assertRaises(O.ZeroPilotEnergyError):
            O.decode(sig)

    def test_all_zero_channel_is_reported_as_no_frame(self):
        # 全零 FIR：前导无能量，穷举后无可定位候选（证明无帧）。
        sig = R.transmit(BITS, time_offset=5, taps=(0j, 0j, 0j))
        with self.assertRaises(O.NoFrameError):
            O.decode(sig)

    def test_pure_noise_no_frame_under_fir_model(self):
        sig = R._noise(555, 200, 0.3)
        with self.assertRaises(O.NoFrameError) as cm:
            O.decode(sig)
        # 消息仍带可行起点扫描数量（200-32+1=169）。
        self.assertIn("169", str(cm.exception))

    def test_budget_exhausted_distinct_under_fir_model(self):
        # 常量信号在安全窗上每个位置都是候选；小预算必须报“预算耗尽”。
        with self.assertRaises(O.SearchBudgetError) as cm:
            O.decode([1 + 0j] * 60, max_candidates=5)
        self.assertIn("not a proof", str(cm.exception))

    def test_invalid_fir_signal_samples_raise(self):
        sig = [1 + 0j] * 60
        sig[10] = complex(0.0, float("nan"))
        with self.assertRaises(O.SignalFormatError) as cm:
            O.decode(sig)
        self.assertIn("signal[10]", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
