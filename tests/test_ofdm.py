"""必要测试：输入校验、DFT、端到端恢复、并列候选、零导频、截断、无帧/预算。

期望值来源：``tests/reference.py`` 的**独立**直接合成器及其输入参数
（真实时偏、频偏、增益、位串），不从被测核心反推。往返一致仅作补充。
"""
from __future__ import annotations

import cmath
import math
import unittest

import _bootstrap  # noqa: F401  路径引导

import ofdm_recovery as O
import reference as R

BITS = R.fixed_bits()
BITS_STR = "".join(str(b) for b in BITS)
SQRT1_2 = 1.0 / math.sqrt(2.0)


class TestInputValidation(unittest.TestCase):
    def test_nan_inf_rejected_with_index(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            sig = [0j] * 40
            sig[7] = bad
            with self.assertRaises(O.SignalFormatError) as cm:
                O.decode(sig)
            self.assertIn("signal[7]", str(cm.exception))

    def test_complex_nonfinite_rejected(self):
        with self.assertRaises(O.SignalFormatError):
            O.decode([complex(1.0, float("nan"))] * 40)
        with self.assertRaises(O.SignalFormatError):
            O.find_candidates([complex(float("inf"), 0.0)] * 40)

    def test_bool_rejected_as_sample_and_as_int_param(self):
        with self.assertRaises(O.SignalFormatError):
            O.decode([True] * 60)
        with self.assertRaises(O.ParameterError):
            O.find_candidates([1 + 0j] * 60, max_candidates=True)
        with self.assertRaises(O.ParameterError):
            O.find_candidates([1 + 0j] * 60, rho_threshold=False)

    def test_type_and_length_limits(self):
        with self.assertRaises(O.SignalFormatError):
            O.decode("010101")
        with self.assertRaises(O.SignalFormatError):
            O.decode([])
        with self.assertRaises(O.SignalFormatError) as cm:
            O.decode([0j] * (O.MAX_SIGNAL_LEN + 1))
        self.assertIn(str(O.MAX_SIGNAL_LEN), str(cm.exception))
        # 上限边界本身可被接受（静漠信号随后报“无帧”，而非格式错误）。
        with self.assertRaises(O.NoFrameError):
            O.decode([0j] * O.MAX_SIGNAL_LEN)

    def test_threshold_range(self):
        sig = R.transmit(BITS)
        for bad in (0.0, 1.0, -0.2, 1.5):
            with self.assertRaises(O.ParameterError):
                O.decode(sig, rho_threshold=bad)
        with self.assertRaises(O.ParameterError):
            O.decode(sig, residual_threshold=float("nan"))
        with self.assertRaises(O.ParameterError):
            O.find_candidates(sig, max_candidates=0)


class TestDft(unittest.TestCase):
    """用已知变换对校验自行 DFT（期望值不调用被测代码）。"""

    def test_delta_is_constant(self):
        x = [1 + 0j] + [0j] * 31
        X = O.dft(x)
        self.assertEqual(len(X), 32)
        for v in X:
            self.assertAlmostEqual(v.real, 1.0, places=12)
            self.assertAlmostEqual(v.imag, 0.0, places=12)

    def test_constant_is_dc_bin(self):
        X = O.dft([1.0] * 8)
        for k, v in enumerate(X):
            self.assertAlmostEqual(abs(v), 8.0 if k == 0 else 0.0, places=10)

    def test_single_bin_inverse_pair(self):
        # x[n] = exp(j2π k0 n/N) 应只在 DFT 索引 k0 处有能量 N。
        n = 16
        k0 = 5
        x = [cmath.exp(2j * math.pi * k0 * m / n) for m in range(n)]
        X = O.dft(x)
        for k, v in enumerate(X):
            expect = n if k == k0 else 0.0
            self.assertAlmostEqual(abs(v), expect, places=10)


class TestEndToEndClean(unittest.TestCase):
    def test_known_time_offset_cfo_gain(self):
        gain = 1.7 * cmath.exp(0.6j)
        sig = R.transmit(
            BITS, time_offset=31, cfo=0.137, gain=gain, trailing=19
        )
        self.assertEqual(len(sig), 31 + O.FRAME_LEN + 19)  # =162，远低于 512
        result = O.decode(sig)

        self.assertEqual(result.start, 31)
        self.assertAlmostEqual(result.frequency_offset, 0.137, places=10)
        self.assertLess(abs(result.frequency_offset), 0.5)
        self.assertAlmostEqual(result.residual, 0.0, places=12)
        self.assertAlmostEqual(result.gain, gain, places=10)
        self.assertEqual(len(result.symbols), 24)
        self.assertEqual(result.bits, BITS_STR)

    def test_symbol_values_and_ordering(self):
        sig = R.transmit(BITS, time_offset=4)
        result = O.decode(sig)
        # 前两个符号对应位对 (0,1)->(-I 轴)，(1,0)，顺序 -6..-1,+1..+6。
        self.assertAlmostEqual(result.symbols[0].real, SQRT1_2, places=10)
        self.assertAlmostEqual(result.symbols[0].imag, -SQRT1_2, places=10)
        self.assertAlmostEqual(result.symbols[1].real, -SQRT1_2, places=10)
        self.assertAlmostEqual(result.symbols[1].imag, SQRT1_2, places=10)
        # 每符号 12 个数据载波，共 24 符号 48 位。
        self.assertEqual(len(result.bits), 48)

    def test_negative_cfo_boundary_inside_half(self):
        sig = R.transmit(BITS, time_offset=6, cfo=-0.45, gain=0.5 - 0.5j)
        result = O.decode(sig)
        self.assertEqual(result.start, 6)
        self.assertAlmostEqual(result.frequency_offset, -0.45, places=10)
        self.assertAlmostEqual(result.gain, 0.5 - 0.5j, places=10)
        self.assertEqual(result.bits, BITS_STR)

    def test_full_512_length_boundary(self):
        # 时偏 400 + 帧 112 = 512，恰为接口上限。
        sig = R.transmit(BITS, time_offset=400, cfo=0.08)
        self.assertEqual(len(sig), O.MAX_SIGNAL_LEN)
        result = O.decode(sig)
        self.assertEqual(result.start, 400)
        self.assertEqual(result.bits, BITS_STR)


class TestNoisyRecovery(unittest.TestCase):
    NOISE_SCALE = 0.05

    def setUp(self):
        # 确定性噪声：独立参考以固定种子生成，非预录结果。
        self.sig = R.transmit(
            BITS,
            time_offset=23,
            cfo=0.2,
            gain=1.0,
            front_scale=self.NOISE_SCALE,
            inband_scale=self.NOISE_SCALE,
            trailing=30,
            seed=99,
        )

    def test_metrics_within_documented_error_budget(self):
        result = O.decode(self.sig)
        self.assertEqual(result.start, 23)
        # README 记载的误差口径（噪声标准差约 0.041 时）：
        self.assertLess(abs(result.frequency_offset - 0.2), 0.02)
        self.assertLess(abs(result.gain - 1.0), 0.10)
        self.assertLess(result.residual, 0.02)
        self.assertEqual(result.bits, BITS_STR)  # 此噪声下位串仍全对

    def test_second_seed(self):
        sig = R.transmit(
            BITS,
            time_offset=11,
            cfo=-0.17,
            front_scale=self.NOISE_SCALE,
            inband_scale=self.NOISE_SCALE,
            seed=777,
        )
        result = O.decode(sig)
        self.assertEqual(result.start, 11)
        self.assertLess(abs(result.frequency_offset + 0.17), 0.02)
        self.assertEqual(result.bits, BITS_STR)


class TestCandidateTie(unittest.TestCase):
    def test_equal_residual_earliest_position_wins(self):
        # 两帧背靠背：两处前导残差都为 0，必须选最早的。
        frame = R.build_frame(BITS)
        sig = [0j] * 3 + frame + frame
        candidates = O.find_candidates(sig)
        top = [c for c in candidates if abs(c.residual) < 1e-12]
        self.assertEqual([c.start for c in top[:2]], [3, 3 + O.FRAME_LEN])
        result = O.decode(sig)
        self.assertEqual(result.start, 3)
        self.assertEqual(result.bits, BITS_STR)

    def test_plateau_does_not_raise_under_default_budget(self):
        # 常量信号形成长相关高原，全部候选仍应被枚举而非静默截断。
        candidates = O.find_candidates([1 + 0j] * 60)
        self.assertEqual(len(candidates), 60 - O.PREAMBLE_LEN + 1)
        self.assertEqual(candidates[0].start, 0)


class TestRejections(unittest.TestCase):
    def test_zero_pilot_energy(self):
        sig = R.transmit(BITS, zero_pilots=True)
        with self.assertRaises(O.ZeroPilotEnergyError) as cm:
            O.decode(sig)
        self.assertIn("pilot", str(cm.exception).lower())

    def test_truncated_after_preamble(self):
        # 帧起点 10，只给到帧起点后 45 样本（不足两个 OFDM 符号）。
        sig = R.transmit(BITS, time_offset=10, truncate=45)
        with self.assertRaises(O.TruncatedFrameError):
            O.decode(sig)

    def test_truncated_inside_second_symbol(self):
        sig = R.transmit(BITS, time_offset=2, truncate=O.FRAME_LEN - 3)
        with self.assertRaises(O.TruncatedFrameError):
            O.decode(sig)

    def test_no_frame_is_proven_absence(self):
        # 纯噪声：穷举全部可行起点仍无候选。
        sig = R._noise(555, 200, 0.3)
        with self.assertRaises(O.NoFrameError) as cm:
            O.decode(sig)
        self.assertNotIsInstance(cm.exception, O.SearchBudgetError)
        # 消息带扫描范围：200 样本有 200-32+1=169 个可行起点。
        self.assertIn("169", str(cm.exception))

    def test_budget_exhaustion_is_distinct(self):
        # 常量信号全部位置都是候选；预算 5 必须明确报“预算耗尽”，
        # 不能混同于“证明无帧”。
        with self.assertRaises(O.SearchBudgetError) as cm:
            O.find_candidates([1 + 0j] * 60, max_candidates=5)
        self.assertIn("not a proof", str(cm.exception))

    def test_cfo_outside_contract_rejected(self):
        block = R._preamble_block()
        # 第二块整体多转 0.6π -> 估计频偏 0.6 个载波间隔，超出 |f|<0.5。
        sig = [0j] * 5 + block + [
            x * cmath.exp(0.6j * math.pi) for x in block
        ] + [0j] * 90
        with self.assertRaises(O.FrequencyOffsetError):
            O.decode(sig)

    def test_residual_detects_amplitude_mismatch(self):
        # 相关相干性为 1（纯缩放完全相关），但残差 0.45 暴露幅度失配：
        # 验证择优确实依赖前导残差，而不是只看相关。
        block = R._preamble_block()
        sig = [0j] * 5 + block + [0.3 * x for x in block] + [0j] * 90
        candidates = O.find_candidates(sig)
        self.assertGreaterEqual(candidates[0].coherence, 0.99)
        self.assertGreater(candidates[0].residual, 0.35)
        with self.assertRaises(O.FrameQualityError):
            O.decode(sig)


class TestRoundTripSupplementary(unittest.TestCase):
    """往返一致性仅作补充：多组随机位串干净信道下自洽。"""

    def test_round_trip_all_bit_patterns(self):
        import random

        rng = random.Random(20260922)
        for _ in range(5):
            bits = [rng.randrange(2) for _ in range(48)]
            sig = R.transmit(bits, time_offset=rng.randrange(0, 50))
            result = O.decode(sig)
            self.assertEqual(
                result.bits, "".join(str(b) for b in bits)
            )


if __name__ == "__main__":
    unittest.main()
