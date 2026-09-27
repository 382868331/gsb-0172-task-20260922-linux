"""流式接收器测试：任意分块、帧间噪声、误识别候选、残缺报告。

期望值（真实位串、帧起点）来自 ``tests/reference.py`` 的独立合成器与
固定输入参数，不从被测核心反推。
"""
from __future__ import annotations

import random
import unittest

import _bootstrap  # noqa: F401  路径引导

import ofdm_recovery as O
import reference as R

BITS = R.fixed_bits()
BITS_STR = "".join(str(b) for b in BITS)


def other_bits(seed=77):
    rng = random.Random(seed)
    return [rng.randrange(2) for _ in range(48)]


class TestStreamChunking(unittest.TestCase):
    def _make_two_frames(self):
        # 两帧内容不同，间隔 13 个噪声样本；总长 5+112+13+112+7 = 249。
        f0 = R.build_frame(BITS)
        f1 = R.build_frame(other_bits())
        noise = R._noise(321, 13 + 5 + 7, 0.02)
        sig = list(noise[:5]) + f0 + noise[5:18] + f1 + noise[18:]
        return sig, 5, 5 + O.FRAME_LEN + 13

    def test_arbitrary_chunk_sizes(self):
        sig, s0, s1 = self._make_two_frames()
        # 故意用不规则块边界（包含 1 样本块），且不对齐帧头。
        boundaries = [0, 1, 2, 7, 30, 55, 120, 121, 200, 240, len(sig)]
        dec = O.StreamDecoder()
        out = []
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            out.extend(dec.feed(sig[a:b]))
        out.extend(dec.finish())
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].start, s0)
        self.assertEqual(out[1].start, s1)
        self.assertEqual(out[0].bits, BITS_STR)
        self.assertEqual(out[1].bits, "".join(map(str, other_bits())))
        self.assertEqual(dec.frames_emitted, 2)
        self.assertEqual(dec.total_received, len(sig))

    def test_sample_by_sample(self):
        sig, s0, s1 = self._make_two_frames()
        dec = O.StreamDecoder()
        out = []
        for x in sig:
            got = dec.feed([x])
            self.assertLessEqual(len(got), 1)
            out.extend(got)
        out.extend(dec.finish())
        self.assertEqual([r.start for r in out], [s0, s1])

    def test_small_chunks(self):
        # 帧头在块中间出现：每块 3 个样本。
        sig, _, _ = self._make_two_frames()
        out = O.decode_stream([sig[i:i + 3] for i in range(0, len(sig), 3)])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].bits, BITS_STR)

    def test_empty_chunk_is_noop(self):
        dec = O.StreamDecoder()
        self.assertEqual(dec.feed([]), [])
        self.assertEqual(dec.stats().total_received, 0)


class TestFalseHeader(unittest.TestCase):
    def _make_signal_with_false_header(self):
        # 假前导：两个块相干性=1 但第二块幅度 0.3 -> 残差 0.45 被拒绝；
        # 中间隔 80 个噪声样本后才是真帧。
        block = R._preamble_block()
        rng = random.Random(5)
        noise = [
            complex(rng.random() - 0.5, rng.random() - 0.5)
            for _ in range(80)
        ]
        frame = R.build_frame(BITS)
        sig = (
            [0j] * 5
            + list(block) + [0.3 * x for x in block]
            + noise + frame + [0j] * 20
        )
        return sig, 5 + 32 + 80

    def test_false_header_does_not_skip_real_frame_offline(self):
        sig, true_start = self._make_signal_with_false_header()
        result = O.decode(sig)
        self.assertEqual(result.start, true_start)
        self.assertEqual(result.bits, BITS_STR)

    def test_false_header_in_stream_with_split(self):
        sig, true_start = self._make_signal_with_false_header()
        # 块边界正好切在假前导与真帧内部。
        chunks = [sig[:40], sig[40:90], sig[90:140], sig[140:]]
        dec = O.StreamDecoder()
        out = []
        for ch in chunks:
            out.extend(dec.feed(ch))
        out.extend(dec.finish())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].start, true_start)
        self.assertEqual(out[0].bits, BITS_STR)


class TestIncompleteRetention(unittest.TestCase):
    def test_partial_frame_waits_for_next_batch(self):
        frame = R.build_frame(BITS)
        dec = O.StreamDecoder()
        self.assertEqual(dec.feed(frame[:50]), [])       # 前导+部分帧身
        self.assertGreaterEqual(dec.buffered_samples, 50)
        out = dec.feed(frame[50:])
        out += dec.finish()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].start, 0)
        self.assertEqual(out[0].bits, BITS_STR)

    def test_preamble_only_waits_then_completes(self):
        frame = R.build_frame(BITS)
        dec = O.StreamDecoder()
        # 先只给前导 32 样本（相关窗口恰好完整）。
        self.assertEqual(dec.feed(frame[:O.PREAMBLE_LEN]), [])
        out = dec.feed(frame[O.PREAMBLE_LEN:])
        out += dec.finish()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].bits, BITS_STR)

    def test_frame_split_into_four_arbitrary_pieces(self):
        frame = R.build_frame(BITS)
        parts = [frame[:17], frame[17:63], frame[63:100], frame[100:]]
        out = O.decode_stream(parts)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].bits, BITS_STR)

    def test_truncated_frame_reported_at_finish(self):
        frame = R.build_frame(BITS)
        dec = O.StreamDecoder()
        dec.feed(frame[:45])  # 前导完整、帧身只到第一个符号附近
        with self.assertRaises(O.TruncatedFrameError) as cm:
            dec.finish()
        exc = cm.exception
        self.assertEqual(exc.start, 0)
        self.assertEqual(exc.available, 45)
        self.assertEqual(exc.needed, O.FRAME_LEN)
        # 状态回滚到 finish 之前：样本仍保留，补全后可以继续。
        self.assertFalse(dec.is_finished)
        out = dec.feed(frame[45:]) + dec.finish()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].bits, BITS_STR)
        self.assertTrue(dec.is_finished)

    def test_truncated_after_offset_reports_absolute_start(self):
        frame = R.build_frame(BITS)
        sig = R._noise(909, 11, 0.01) + frame[:60]
        dec = O.StreamDecoder()
        dec.feed(sig)
        with self.assertRaises(O.TruncatedFrameError) as cm:
            dec.finish()
        self.assertEqual(cm.exception.start, 11)
        self.assertEqual(cm.exception.available, 60)
        self.assertIn("11", str(cm.exception))

    def test_noise_tail_is_not_truncated_frame(self):
        noise = R._noise(424, 200, 0.05)
        dec = O.StreamDecoder()
        dec.feed(noise[:100])
        dec.feed(noise[100:])
        # 无候选 -> 不报错；末尾 <32 样本作为残留保留。
        self.assertEqual(dec.finish(), [])
        self.assertLess(dec.leftover_samples(), O.PREAMBLE_LEN)

    def test_bad_preamble_tail_is_not_truncated_frame(self):
        # 幅度失配的假前导 + 少量后续样本（不足整帧）：
        # 残差检查应直接拒绝它，finish 报“无残缺帧”，而不是 Truncated。
        block = R._preamble_block()
        tail = list(block) + [0.3 * x for x in block] + [0j] * 40
        dec = O.StreamDecoder()
        dec.feed(tail)
        self.assertEqual(dec.finish(), [])


class TestStreamStateAndValidation(unittest.TestCase):
    def test_feed_after_finish_rejected(self):
        dec = O.StreamDecoder()
        dec.feed(R.build_frame(BITS))
        dec.finish()
        with self.assertRaises(O.StreamStateError):
            dec.feed([1 + 0j])
        # 重复 finish 是幂等 no-op。
        self.assertEqual(dec.finish(), [])

    def test_bad_sample_leaves_state_unchanged(self):
        dec = O.StreamDecoder()
        good = R.build_frame(BITS)
        earlier = dec.feed(good)
        self.assertEqual(len(earlier), 1)
        bad = [0j] * 10
        bad[3] = float("nan")
        with self.assertRaises(O.SignalFormatError) as cm:
            dec.feed(bad)
        self.assertIn("chunk[3]", str(cm.exception))
        # 失败批次没有进入缓冲，已有帧与计数不变。
        self.assertEqual(dec.total_received, O.FRAME_LEN)
        self.assertEqual(dec.frames_emitted, 1)
        self.assertEqual(dec.finish(), [])
        self.assertEqual(earlier[0].bits, BITS_STR)

    def test_chunk_length_limit(self):
        dec = O.StreamDecoder()
        with self.assertRaises(O.SignalFormatError):
            dec.feed([0j] * (O.MAX_SIGNAL_LEN + 1))
        with self.assertRaises(O.SignalFormatError):
            dec.feed("0101")

    def test_three_frames_with_noise_and_chunks(self):
        f0 = R.build_frame(BITS)
        f1 = R.build_frame(other_bits(1))
        f2 = R.build_frame(other_bits(2))
        gap = R._noise(3, 9, 0.02)
        sig = f0 + gap + f1 + R._noise(4, 4, 0.02) + f2
        starts = [0, O.FRAME_LEN + 9, 2 * O.FRAME_LEN + 13]
        dec = O.StreamDecoder()
        out = []
        for i in range(0, len(sig), 37):  # 块长与任何结构都不对齐
            out.extend(dec.feed(sig[i:i + 37]))
        out.extend(dec.finish())
        self.assertEqual([r.start for r in out], starts)


if __name__ == "__main__":
    unittest.main()
