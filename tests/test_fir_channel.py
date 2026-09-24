"""FIR 信道迭代的新增测试：3 抽头固定复数 FIR、信道估计、ZF 均衡、深零点擦除。

期望值来源：
* 抽头、时偏、频偏是传入独立合成器 ``tests/reference.py`` 的**已知输入参数**；
* 频响 ``H(k)=Σ_l h_l exp(-j2πkl/32)``、RMS 增益、邻载波相对响应、
  导频 LS 增益 (H(7)+H(25))/2 等均在本文件内用解析式独立计算；
* 位串真值来自合成器输入位，绝不调用被测核心反推。
"""
from __future__ import annotations

import cmath
import math
import random
import unittest

import _bootstrap  # noqa: F401  路径引导

import ofdm_recovery as O
import reference as R

BITS = R.fixed_bits()
BITS_STR = "".join(str(b) for b in BITS)
SGN = O.SIGNED_DATA_CARRIERS


def channel_response(taps, k):
    """独立解析式：H(k) = Σ_l h_l exp(-j2πkl/32)。"""
    return sum(
        complex(taps[l]) * cmath.exp(-2j * math.pi * k * l / O.N_FFT)
        for l in range(len(taps))
    )


def padded(taps):
    """把 1..3 抽头补成 3 元组。"""
    taps = [complex(t) for t in taps]
    return tuple(taps + [0j] * (3 - len(taps)))


def unerased_match(bits_out, erased_positions):
    """未擦除位与真值逐位一致（擦除位置为符号索引 0..23）。"""
    erased_bits = set()
    for idx in erased_positions:
        erased_bits.add(2 * idx)
        erased_bits.add(2 * idx + 1)
    if len(bits_out) != 48:
        return False
    for i in range(48):
        if i in erased_bits:
            if bits_out[i] != "x":
                return False
        elif bits_out[i] != BITS_STR[i]:
            return False
    return True


def null_taps(k0):
    """2 抽头 h=[1, -exp(j2πk0/32)]：H(k0)=0 的解析构造。"""
    return [1 + 0j, -cmath.exp(2j * math.pi * k0 / O.N_FFT)]


class TestFirEstimationClean(unittest.TestCase):
    """无噪声：抽头、频偏、起点应到机器精度；非零点载波 48 位全对。"""

    def test_single_complex_tap_matches_legacy_flat(self):
        gain = 1.7 * cmath.exp(0.6j)
        # 平坦增益既可走旧路径（gain=），也可走显式单抽头（taps=），
        # 两者是同一物理信道，恢复结果应一致。
        legacy = R.transmit(BITS, time_offset=31, cfo=0.137, gain=gain,
                            trailing=19)
        sig = R.transmit(BITS, time_offset=31, cfo=0.137, taps=[gain],
                         trailing=19)
        result = O.decode(sig)
        legacy_result = O.decode(legacy)
        self.assertEqual(result.start, 31)
        self.assertAlmostEqual(result.frequency_offset, 0.137, places=12)
        self.assertAlmostEqual(result.residual, 0.0, places=12)
        self.assertAlmostEqual(result.gain, gain, places=10)
        self.assertAlmostEqual(legacy_result.gain, result.gain, places=12)
        self.assertEqual(len(result.channel.taps), 3)
        self.assertAlmostEqual(result.channel.taps[0], gain, places=10)
        self.assertAlmostEqual(abs(result.channel.taps[1]), 0.0, places=10)
        self.assertAlmostEqual(abs(result.channel.taps[2]), 0.0, places=10)
        self.assertEqual(result.bits, BITS_STR)
        self.assertEqual(result.erased_carriers, ())

    def test_three_taps_time_offset_cfo(self):
        taps = [0.8 + 0.3j, -0.4 + 0.5j, 0.1 - 0.2j]
        sig = R.transmit(BITS, time_offset=17, cfo=-0.21, taps=taps)
        result = O.decode(sig)
        self.assertEqual(result.start, 17)
        self.assertAlmostEqual(result.frequency_offset, -0.21, places=10)
        # 多径下前导相关残差含瞬态项，非零；训练拟合残差才是 ≈0 的量。
        self.assertLess(result.residual, 0.05)
        self.assertAlmostEqual(result.channel.fit_residual, 0.0, places=10)
        for got, want in zip(result.channel.taps, padded(taps)):
            self.assertAlmostEqual(got, want, places=9)
        self.assertEqual(result.bits, BITS_STR)

    def test_frequency_response_and_rms_are_analytic(self):
        taps = [0.6 - 0.2j, 0.3 + 0.4j, -0.5 + 0.1j]
        sig = R.transmit(BITS, time_offset=5, cfo=0.08, taps=taps)
        result = O.decode(sig)
        h = padded(taps)
        want_rms = math.sqrt(sum(abs(t) ** 2 for t in h))
        self.assertAlmostEqual(result.channel.rms_gain, want_rms, places=10)
        self.assertEqual(len(result.channel.frequency_response), 32)
        for k in range(32):
            self.assertAlmostEqual(
                result.channel.frequency_response[k],
                channel_response(h, k),
                places=9,
            )

    def test_legacy_gain_field_is_pilot_ls_under_multipath(self):
        # 导频 +7 发 +1、-7（索引 25）发 -1：
        # 单一增益 LS 解为 gain = (H(7) + H(25)) / 2（解析，独立计算）。
        taps = [0.9 + 0.0j, 0.2 - 0.3j, -0.1 + 0.4j]
        sig = R.transmit(BITS, time_offset=8, taps=taps)
        result = O.decode(sig)
        h = padded(taps)
        want_gain = (channel_response(h, 7) + channel_response(h, 25)) / 2.0
        self.assertAlmostEqual(result.gain, want_gain, places=9)
        # 均衡使用的是完整频响，位串仍全对。
        self.assertEqual(result.bits, BITS_STR)


class TestDeepNullErasure(unittest.TestCase):
    """深零点载波必须擦除（不伪造判决），其余位无噪声下全部正确。"""

    def test_exact_null_on_every_data_carrier(self):
        # 小规模枚举：在每个数据载波 ±1..±6 上构造解析精确零点。
        for k0 in SGN:
            with self.subTest(k0=k0):
                sig = R.transmit(BITS, time_offset=9, taps=null_taps(k0))
                result = O.decode(sig)
                self.assertEqual(result.start, 9)
                self.assertEqual(result.erased_carriers, (k0,))
                idx = SGN.index(k0)
                self.assertEqual(
                    result.erased_positions, (idx, idx + 12)
                )
                # 该载波两个符号上均为 None，bits 中对应两位为 'x'。
                self.assertIsNone(result.symbols[idx])
                self.assertIsNone(result.symbols[idx + 12])
                self.assertTrue(
                    unerased_match(result.bits, result.erased_positions)
                )
                self.assertEqual(result.bits.count("x"), 4)

    def test_null_plus_third_tap_still_erased(self):
        # 3 抽头、零点精确落在 +3：解析构造 H(3)=0，即
        # h0 + h1·e^{-jθ} + h2·e^{-j2θ} = 0（θ=2π·3/32），
        # 取 h0=1、h2=0.3+0.2j，反解 h1 = -(1 + h2·e^{-j2θ})·e^{jθ}。
        theta = 2.0 * math.pi * 3 / O.N_FFT
        h2 = 0.3 + 0.2j
        h1 = -(1.0 + h2 * cmath.exp(-2j * theta)) * cmath.exp(1j * theta)
        taps = [1 + 0j, h1, h2]
        # 解析自检：该构造在 k=3 处确为零点。
        self.assertLess(abs(channel_response(taps, 3)), 1e-12)
        # 期望擦除集完全由解析式算出：|H(k)| < 0.1·rms(H) 的数据载波。
        rms = math.sqrt(sum(abs(t) ** 2 for t in padded(taps)))
        want_erased = tuple(
            kc for kc in SGN
            if abs(channel_response(taps, kc % O.N_FFT)) < 0.1 * rms
        )
        sig = R.transmit(BITS, time_offset=23, cfo=0.137, taps=taps)
        result = O.decode(sig)
        self.assertEqual(result.erased_carriers, want_erased)
        self.assertIn(3, result.erased_carriers)
        self.assertAlmostEqual(result.frequency_offset, 0.137, places=10)
        for got, want in zip(result.channel.taps, padded(taps)):
            self.assertAlmostEqual(got, want, places=9)
        self.assertTrue(unerased_match(result.bits, result.erased_positions))

    def test_null_on_pilot_does_not_erase_data(self):
        # 零点落在导频 +7：数据载波无一擦除，48 位全对，且不得误报零导频。
        taps = null_taps(7)
        sig = R.transmit(BITS, time_offset=12, cfo=0.1, taps=taps)
        result = O.decode(sig)
        self.assertEqual(result.start, 12)
        self.assertEqual(result.erased_carriers, ())
        self.assertEqual(result.bits, BITS_STR)
        # +7 导频观测为 0，另一导频承载增益：LS 增益解析为 H(25)/2。
        self.assertAlmostEqual(
            result.gain, channel_response(padded(taps), 25) / 2.0, places=9
        )

    def test_erasure_threshold_boundary_is_analytic(self):
        # 邻载波相对响应（解析）：|d|=0 -> 0；|d|=1 -> 2sin(pi/32)/sqrt2
        # ≈ 0.1386；|d|=2 ≈ 0.2759。
        d1 = 2.0 * math.sin(math.pi / O.N_FFT) / math.sqrt(2.0)
        d2 = 2.0 * math.sin(2.0 * math.pi / O.N_FFT) / math.sqrt(2.0)
        self.assertLess(d1, 0.2)
        self.assertGreater(d1, 0.1)
        self.assertGreater(d2, 0.2)

        sig = R.transmit(BITS, taps=null_taps(3))
        # 默认 0.1：只擦零点 +3。
        r_default = O.decode(sig)
        self.assertEqual(r_default.erased_carriers, (3,))
        # 门限放到 0.2：+2、+3、+4 全部擦除（+3 两侧 d=1）。
        r_wide = O.decode(sig, erasure_threshold=0.2)
        self.assertEqual(r_wide.erased_carriers, (2, 3, 4))
        # 门限远小于估计零点的数值残差（~2e-16 相对）时无擦除，
        # 强行均衡零点载波（调用方自担数值风险），位串不含 'x'。
        r_off = O.decode(sig, erasure_threshold=1e-300)
        self.assertEqual(r_off.erased_carriers, ())
        self.assertNotIn("x", r_off.bits)

    def test_null_response_is_exactly_zero(self):
        # 估计频响在被擦载波上确为零点（机器精度）。
        sig = R.transmit(BITS, taps=null_taps(-4))
        result = O.decode(sig)
        self.assertLess(
            abs(result.channel.frequency_response[(-4) % O.N_FFT]), 1e-9
        )

    def test_erased_bits_with_small_noise(self):
        taps = null_taps(3)
        sig = R.transmit(
            BITS, time_offset=14, cfo=0.1, taps=taps,
            inband_scale=0.02, seed=20260924,
        )
        result = O.decode(sig)
        self.assertEqual(result.start, 14)
        self.assertEqual(result.erased_carriers, (3,))
        self.assertTrue(
            unerased_match(result.bits, result.erased_positions)
        )


class TestFirTimingAndTies(unittest.TestCase):
    def test_multipath_back_to_back_earliest(self):
        taps = [0.7 + 0.2j, -0.5 + 0.4j, 0.2 - 0.3j]
        frame = R.build_frame(BITS)
        fir_frame = R._fir_frame(frame, taps)
        sig = [0j] * 4 + fir_frame + fir_frame
        result = O.decode(sig)
        self.assertEqual(result.start, 4)
        self.assertEqual(result.bits, BITS_STR)

    def test_multipath_full_length_boundary(self):
        taps = [0.8 + 0.1j, 0.3 - 0.4j, -0.2 + 0.2j]
        sig = R.transmit(BITS, time_offset=400, cfo=0.08, taps=taps)
        self.assertEqual(len(sig), O.MAX_SIGNAL_LEN)
        result = O.decode(sig)
        self.assertEqual(result.start, 400)
        self.assertEqual(result.bits, BITS_STR)

    def test_truncated_under_multipath(self):
        sig = R.transmit(
            BITS, time_offset=10, taps=[1 + 0j, 0.5j, -0.3],
            truncate=O.FRAME_LEN - 3,
        )
        with self.assertRaises(O.TruncatedFrameError):
            O.decode(sig)


class TestFirValidation(unittest.TestCase):
    def test_bad_erasure_threshold(self):
        sig = R.transmit(BITS, taps=[1 + 0j, 0.2j])
        for bad in (0.0, -0.1, float("nan"), float("inf")):
            with self.assertRaises(O.ParameterError):
                O.decode(sig, erasure_threshold=bad)
        with self.assertRaises(O.ParameterError):
            O.decode(sig, erasure_threshold=True)
        with self.assertRaises(O.ParameterError):
            O.decode(sig, erasure_threshold="0.1")

    def test_zero_pilots_still_rejected_under_multipath(self):
        # 信道在导频处有能量，但发射端置零导频：必须仍被识别拒绝。
        taps = [1 + 0j, 0.2 - 0.1j, 0.1 + 0.1j]
        sig = R.transmit(BITS, taps=taps, zero_pilots=True)
        with self.assertRaises(O.ZeroPilotEnergyError) as cm:
            O.decode(sig)
        self.assertIn("pilot", str(cm.exception).lower())

    def test_no_frame_under_multipath_like_noise(self):
        sig = R._noise(555, 200, 0.3)
        with self.assertRaises(O.NoFrameError):
            O.decode(sig)

    def test_failed_decode_does_not_mutate_input(self):
        sig = R.transmit(BITS, time_offset=5, truncate=45)
        snapshot = list(sig)
        with self.assertRaises(O.TruncatedFrameError):
            O.decode(sig)
        self.assertEqual(sig, snapshot)

    def test_successful_decode_does_not_mutate_input(self):
        sig = list(R.transmit(BITS, time_offset=7, taps=[1 + 0j, 0.3j]))
        snapshot = list(sig)
        O.decode(sig)
        self.assertEqual(sig, snapshot)


class TestSymbolsToBitsErasures(unittest.TestCase):
    def test_none_symbol_becomes_xx(self):
        s = 1.0 / math.sqrt(2.0)
        self.assertEqual(
            O.symbols_to_bits([complex(s, s), None, complex(-s, -s)]),
            "00xx11",
        )

    def test_rejects_bad_symbol_type(self):
        with self.assertRaises(O.SignalFormatError):
            O.symbols_to_bits([1])  # int 不是合法符号（须 complex 或 None）
        with self.assertRaises(O.SignalFormatError):
            O.symbols_to_bits([True])
        with self.assertRaises(O.SignalFormatError):
            O.symbols_to_bits("01")


class TestFirRoundTripSupplementary(unittest.TestCase):
    """补充：随机位串 + 随机温和多径（无零点）下 48 位自洽。"""

    def test_random_bits_and_taps(self):
        rng = random.Random(20260924)
        for _ in range(5):
            bits = [rng.randrange(2) for _ in range(48)]
            # h0 主导：|H(k)| >= |h0|-|h1|-|h2| > 0.4，远高于擦除门限，
            # 不会出现深零点，故应恢复全部 48 位。
            taps = [
                1.0 + 0.0j,
                complex(rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25)),
                complex(rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)),
            ]
            toff = rng.randrange(0, 40)
            cfo = rng.uniform(-0.4, 0.4)
            sig = R.transmit(bits, time_offset=toff, cfo=cfo, taps=taps)
            result = O.decode(sig)
            self.assertEqual(result.start, toff)
            self.assertEqual(result.erased_carriers, ())
            self.assertEqual(
                result.bits, "".join(str(b) for b in bits)
            )


if __name__ == "__main__":
    unittest.main()
