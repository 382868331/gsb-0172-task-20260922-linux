"""流式多帧解码测试：任意分块、帧间噪声、假帧头、残缺帧报告。

期望值来源：tests/reference.py 独立合成器的输入参数（位串、时偏、
频偏、增益），不从被测核心反推。
"""
from __future__ import annotations

import cmath
import unittest

import _bootstrap  # noqa: F401  路径引导

import ofdm_recovery as O
import reference as R

BITS = R.fixed_bits()
BITS_STR = "".join(str(b) for b in BITS)
BITS2 = [1 - b for b in BITS]  # 第二帧用不同的确定位串
BITS2_STR = "".join(str(b) for b in BITS2)


def build_stream():
    """噪声 + 帧1 + 噪声 + 帧2 + 噪声 + 残缺帧头，记录真实绝对位置。"""
    gain1 = 1.7 * cmath.exp(0.6j)
    gain2 = 0.9 * cmath.exp(-0.4j)
    seg1 = R.transmit(BITS, time_offset=17, cfo=0.137, gain=gain1,
                      front_scale=0.05, inband_scale=0.05, seed=42)
    gap = R._noise(7, 23, 0.05)
    seg2 = R.transmit(BITS2, time_offset=0, cfo=-0.21, gain=gain2,
                      inband_scale=0.05, seed=99)
    tail_noise = R._noise(13, 11, 0.05)
    partial = R.transmit(BITS, time_offset=0, cfo=0.05, seed=5,
                         truncate=70)  # 帧头可定位但帧不完整
    stream = list(seg1) + list(gap) + list(seg2) + list(tail_noise) \
        + list(partial)
    start1 = 17
    start2 = len(seg1) + len(gap)
    start3 = start2 + len(seg2) + len(tail_noise)
    return stream, start1, start2, start3


def feed_in_chunks(dec, stream, sizes):
    """按给定块尺寸序列喂入，汇总全部恢复帧与最终残留。"""
    results = []
    pos = 0
    i = 0
    while pos < len(stream):
        n = sizes[i % len(sizes)]
        results.extend(dec.feed(stream[pos:pos + n]))
        pos += n
        i += 1
    last, tail = dec.finish()
    results.extend(last)
    return results, tail


class TestStreamMultiFrame(unittest.TestCase):
    def test_multi_frame_arbitrary_chunks(self):
        stream, s1, s2, s3 = build_stream()
        dec = O.StreamDecoder()
        # 任意分块：1、7、40、113 循环，绝不对齐帧头。
        results, tail = feed_in_chunks(dec, stream, [1, 7, 40, 113])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].start, s1)
        self.assertEqual(results[0].bits, BITS_STR)
        self.assertAlmostEqual(results[0].frequency_offset, 0.137, delta=0.02)
        self.assertEqual(results[1].start, s2)
        self.assertEqual(results[1].bits, BITS2_STR)
        self.assertAlmostEqual(results[1].frequency_offset, -0.21, delta=0.02)
        # 结尾残缺帧必须被明确报告。
        self.assertEqual(tail.pending_frame_start, s3)
        self.assertGreater(tail.leftover_samples, 0)
        self.assertIn("incomplete frame", tail.detail)

    def test_chunking_invariance(self):
        stream, s1, s2, _s3 = build_stream()
        r1, t1 = feed_in_chunks(O.StreamDecoder(), stream, [1])
        r2, t2 = feed_in_chunks(O.StreamDecoder(), stream, [len(stream)])
        r3, t3 = feed_in_chunks(O.StreamDecoder(), stream, [3, 111, 2, 56])
        for rs in (r2, r3):
            self.assertEqual([r.start for r in rs], [s1, s2])
            self.assertEqual([r.bits for r in rs],
                             [r.bits for r in r1])
        self.assertEqual(t1, t2)
        self.assertEqual(t2, t3)

    def test_false_header_does_not_skip_real_frame(self):
        # 假帧头：前导第二块幅度失配（残差 0.45 > 门限），候选成立但
        # 后续检查在任何移位下都失败；解码器不得因此跳过后面真正的帧。
        block = R._preamble_block()
        fake = list(block) + [0.3 * x for x in block] + list(R._noise(4, 80, 0.5))
        real = R.transmit(BITS2, time_offset=0, cfo=0.1, seed=77)
        gap = R._noise(3, 9, 0.05)
        stream = list(fake) + list(gap) + list(real)
        real_start = len(fake) + len(gap)
        dec = O.StreamDecoder()
        results, tail = feed_in_chunks(dec, stream, [13, 250])
        self.assertEqual([r.start for r in results], [real_start])
        self.assertEqual(results[0].bits, BITS2_STR)
        self.assertIsNone(tail.pending_frame_start)

    def test_clean_ending_reports_only_noise(self):
        seg = R.transmit(BITS, time_offset=5, trailing=20, front_scale=0.05,
                         seed=8)
        dec = O.StreamDecoder()
        results, tail = feed_in_chunks(dec, seg, [64])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].start, 5)
        self.assertIsNone(tail.pending_frame_start)
        self.assertEqual(tail.leftover_samples, 20)

    def test_pure_noise_yields_nothing(self):
        noise = R._noise(1, 200, 0.3)
        dec = O.StreamDecoder()
        results, tail = feed_in_chunks(dec, noise, [17])
        self.assertEqual(results, [])
        self.assertIsNone(tail.pending_frame_start)

    def test_empty_chunk_is_noop(self):
        dec = O.StreamDecoder()
        self.assertEqual(dec.feed([]), [])
        results, tail = dec.finish()
        self.assertEqual(results, [])
        self.assertEqual(tail.leftover_samples, 0)


class TestStreamValidation(unittest.TestCase):
    def test_bad_chunk_rejected_with_index(self):
        dec = O.StreamDecoder()
        with self.assertRaises(O.SignalFormatError) as cm:
            dec.feed([0j, 0j, float("nan")])
        self.assertIn("chunk[2]", str(cm.exception))
        with self.assertRaises(O.SignalFormatError):
            dec.feed([True] * 4)
        with self.assertRaises(O.SignalFormatError):
            dec.feed("0101")

    def test_bad_params_rejected(self):
        with self.assertRaises(O.ParameterError):
            O.StreamDecoder(rho_threshold=1.5)
        with self.assertRaises(O.ParameterError):
            O.StreamDecoder(residual_threshold=float("nan"))
        with self.assertRaises(O.ParameterError):
            O.StreamDecoder(max_candidates=True)

    def test_feed_after_finish_rejected(self):
        dec = O.StreamDecoder()
        dec.finish()
        with self.assertRaises(O.StreamStateError):
            dec.feed([0j])
        with self.assertRaises(O.StreamStateError):
            dec.finish()

    def test_failed_feed_leaves_state_unchanged(self):
        # 构造候选数超过预算的输入：SearchBudgetError 抛出后，
        # 解码器状态不变，提高预算重试应恢复正常。
        frame = R.build_frame(BITS)
        # 周期 16 的序列在每个起点都有 coherence=1，候选数巨大。
        tricky = list(frame[:16]) * 30
        dec = O.StreamDecoder(max_candidates=4)
        with self.assertRaises(O.SearchBudgetError):
            dec.feed(tricky)
        self.assertEqual(dec.samples_consumed, 0)
        dec2 = O.StreamDecoder(max_candidates=4096)
        results, _tail = feed_in_chunks(dec2, tricky, [512])
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
