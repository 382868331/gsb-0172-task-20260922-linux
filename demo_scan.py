"""多帧扫描演示：一个有限缓冲区内一次扫出多帧。

场景（总长 413 样本，<= 512 上限）：
  * 前置噪声 17 样本；
  * 帧 A（起点 17，频偏 +0.21，增益 1.3·e^{j0.4}）；
  * 帧 B（起点 129，频偏 -0.33，增益 0.9·e^{-j0.2}），与 A 相邻、频偏不同；
  * 假峰（起点 241）：前导完全正常，但导频与契约不符；
  * 尾部帧（起点 353）：只保留前 60 样本，不完整。

运行：python demo_scan.py
数据由仓库内确定性独立合成器 tests/reference.py 生成，无网络、无预录结果。
"""
from __future__ import annotations

import cmath
import itertools
import math
import os
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import ofdm_recovery as O  # noqa: E402
import reference as R  # noqa: E402

BITS_A = R.fixed_bits()
BITS_B = [1 - b for b in BITS_A]
EXPECT_A = "".join(str(b) for b in BITS_A)
EXPECT_B = "".join(str(b) for b in BITS_B)

CFO_A, CFO_B = 0.21, -0.33
GAIN_A = 1.3 * cmath.exp(0.4j)
GAIN_B = 0.9 * cmath.exp(-0.2j)
START_A, START_B = 17, 17 + O.FRAME_LEN
START_FAKE = START_B + O.FRAME_LEN
START_TAIL = START_FAKE + O.FRAME_LEN
TAIL_KEPT = 60


def _apply_channel(frame, cfo, gain):
    return [
        x * gain * cmath.exp(2j * math.pi * cfo * n / R.N)
        for n, x in enumerate(frame)
    ]


def _fake_frame_wrong_pilots(bits):
    """前导正常、导频与契约不符的假帧（符号 0 发 (+1,+1)，符号 1 发 (-1,-1)）。"""
    block = R._preamble_block()
    frame = list(block) + list(block)
    syms = R.qpsk_modulate(bits)
    carriers = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
    wrong = [(1 + 0j, 1 + 0j), (-1 + 0j, -1 + 0j)]
    for s in range(R.N_SYM):
        X = [0j] * R.N
        for j, kc in enumerate(carriers):
            X[kc % R.N] = syms[s * 12 + j]
        X[7], X[-7 % R.N] = wrong[s]
        t = R._idft(X)
        frame.extend(t[-R.CP:])
        frame.extend(t)
    return frame


def build_buffer():
    sig = list(R._noise(2026, START_A, 0.05))
    sig += _apply_channel(R.build_frame(BITS_A), CFO_A, GAIN_A)
    sig += _apply_channel(R.build_frame(BITS_B), CFO_B, GAIN_B)
    sig += _apply_channel(
        _fake_frame_wrong_pilots(BITS_A), 0.1, 1.1 * cmath.exp(0.3j)
    )
    sig += _apply_channel(R.build_frame(BITS_A), -0.05, 1 + 0j)[:TAIL_KEPT]
    return sig


def _brute_force_best(accepted):
    """枚举 accepted（(start, residual)）全部子集，返回最优起点元组。"""
    best = None
    for r in range(len(accepted) + 1):
        for combo in itertools.combinations(range(len(accepted)), r):
            starts = tuple(sorted(accepted[i][0] for i in combo))
            if any(
                starts[j + 1] - starts[j] < O.FRAME_LEN
                for j in range(len(starts) - 1)
            ):
                continue
            key = (len(combo), -sum(accepted[i][1] for i in combo))
            if best is None or key > best[0] or (
                key == best[0] and starts < best[1]
            ):
                best = (key, starts)
    return best[1]


def main() -> int:
    print("=" * 64)
    print("OFDM 多帧扫描演示 —— 有限缓冲区一次扫出多帧")
    print(f"参数: N={O.N_FFT}, CP={O.CP_LEN}, 帧长={O.FRAME_LEN}, "
          f"缓冲区上限={O.MAX_SIGNAL_LEN}")
    print("=" * 64)

    sig = build_buffer()
    print(f"\n缓冲区长度 = {len(sig)} 样本；布局：")
    print(f"  帧 A  起点 {START_A:>3}  频偏 {CFO_A:+.2f}  增益 {GAIN_A:.3f}")
    print(f"  帧 B  起点 {START_B:>3}  频偏 {CFO_B:+.2f}  增益 {GAIN_B:.3f}")
    print(f"  假峰  起点 {START_FAKE:>3}  前导正常、导频错误")
    print(f"  尾帧  起点 {START_TAIL:>3}  仅 {TAIL_KEPT} 样本（不完整）")

    t0 = time.perf_counter()
    result = O.scan_frames(sig)
    dt = time.perf_counter() - t0

    print(f"\n[选中帧] {len(result.frames)} 帧（扫描耗时 {dt * 1000:.1f} ms）")
    ok = True
    for f, expect, true_cfo in (
        (result.frames[0], EXPECT_A, CFO_A),
        (result.frames[1], EXPECT_B, CFO_B),
    ):
        match = f.bits == expect
        ok &= match
        print(f"  起点 {f.start:>3}  频偏 {f.frequency_offset:+.6f} "
              f"(真实 {true_cfo:+.2f})  前导残差 {f.residual:.2e}  "
              f"导频残差 {f.pilot_residual:.2e}")
        print(f"    增益 {f.gain:.5f}  位串({len(f.bits)}): {f.bits}")
        print(f"    位串匹配: {match}")

    print(f"\n[拒绝候选] {len(result.rejected)} 个（按起点）")
    for r in result.rejected:
        print(f"  起点 {r.start:>3}  前导残差 {r.residual:.4f}  "
              f"原因: {r.reason}")

    print(f"\n[尾部不完整] {len(result.truncated)} 个（只登记，不补零解码）")
    for t in result.truncated:
        print(f"  起点 {t.start:>3}  可用 {t.available}/{O.FRAME_LEN} 样本")

    print(f"\n[合格但未选中] {len(result.not_selected)} 个（重叠竞争中落败）")
    for c in result.not_selected:
        print(f"  起点 {c.start:>3}  前导残差 {c.residual:.4f}")

    # 短缓冲暴力枚举全部合格候选的子集，核对 DP 选择的最优性。
    accepted = [(f.start, f.residual) for f in result.frames]
    accepted += [(c.start, c.residual) for c in result.not_selected]
    best = _brute_force_best(accepted)
    chosen = tuple(f.start for f in result.frames)
    print(f"\n[最优性核对] 合格候选 {len(accepted)} 个，"
          f"暴力枚举全部子集的最优起点序列 = {best}")
    print(f"  DP 选择 = {chosen}  一致: {chosen == best}")

    expect_frames = (START_A, START_B) == chosen
    fake_rejected = any(
        r.start == START_FAKE and r.reason == O.REASON_PILOT_RESIDUAL
        for r in result.rejected
    )
    tail_listed = any(t.start == START_TAIL for t in result.truncated)
    print("\n[演示断言]")
    print(f"  两帧起点 == ({START_A}, {START_B}): {expect_frames}")
    print(f"  假峰以 pilot_residual 拒绝: {fake_rejected}")
    print(f"  尾帧列入 truncated: {tail_listed}")
    print(f"  位串全部匹配: {ok}")
    print(f"  DP 选择最优: {chosen == best}")
    if all([expect_frames, fake_rejected, tail_listed, ok, chosen == best]):
        print("\n演示完成：全部断言通过，均为实时计算。")
        return 0
    print("\n演示失败：存在未通过的断言。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
