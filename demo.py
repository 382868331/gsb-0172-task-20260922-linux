"""固定输入演示：真实计算一次正常恢复与一次实际触发的拒绝。

运行：python demo.py
数据由仓库内确定性独立合成器 tests/reference.py 生成，无网络、无预录结果。
"""
from __future__ import annotations

import cmath
import os
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import ofdm_recovery as O  # noqa: E402
import reference as R  # noqa: E402


def main() -> int:
    bits = R.fixed_bits()
    expected = "".join(str(b) for b in bits)

    print("=" * 64)
    print("OFDM 循环前缀边界与子载波恢复 —— 固定输入演示")
    print(f"参数: N={O.N_FFT}, CP={O.CP_LEN}, 帧长={O.FRAME_LEN}, "
          f"信号长度上限={O.MAX_SIGNAL_LEN}")
    print("=" * 64)

    # 1) 正常帧：已知时偏 23、频偏 +0.137 载波间隔、复增益、确定性前置噪声。
    gain = 1.7 * cmath.exp(0.6j)
    signal = R.transmit(
        bits,
        time_offset=23,
        cfo=0.137,
        gain=gain,
        front_scale=0.05,
        inband_scale=0.05,
        trailing=25,
        seed=42,
    )
    t0 = time.perf_counter()
    result = O.decode(signal)
    dt = time.perf_counter() - t0

    print("\n[正常帧] 接收长度 =", len(signal), "样本")
    print(f"  恢复起点 start   = {result.start}        (真实 23)")
    print(f"  频偏 f (载波间隔) = {result.frequency_offset:+.6f}  (真实 +0.137000)")
    print(f"  前导残差 residual = {result.residual:.6f}    (0 为完全一致)")
    print(f"  复增益 gain       = {result.gain:.5f}  (真实 {gain:.5f})")
    print(f"  恢复位串 ({len(result.bits)} 位):")
    print("   ", result.bits)
    print("  位串匹配:", result.bits == expected)
    print(f"  解码耗时: {dt * 1000:.1f} ms")

    # 2) 坏帧边界：前导可定位，但信号在第二个 OFDM 符号结束前被截断，
    #    接收机必须实际抛出 TruncatedFrameError 拒绝，而不是输出半结果。
    bad = R.transmit(bits, time_offset=10, truncate=70)
    print("\n[坏帧] 帧起点 10，帧长需要", O.FRAME_LEN,
          "样本，但信号在该起点后仅保留 70 样本（总长度", len(bad), "）")
    try:
        O.decode(bad)
    except O.TruncatedFrameError as exc:
        print(f"  已拒绝: {type(exc).__name__}")
        print(f"  原因: {exc}")
    else:
        print("  错误：截断帧未被拒绝")
        return 1

    print("\n演示完成：1 个正常恢复 + 1 个实际触发的拒绝，均为实时计算。")
    return 0


def stream_demo() -> int:
    """流式多帧：任意分块、帧间噪声、假帧头跳过、结尾残缺报告。"""
    print("\n" + "=" * 64)
    print("流式多帧演示：连续输入任意分块，逐帧返回恢复结果")
    print("=" * 64)

    bits2 = [1 - b for b in R.fixed_bits()]
    expected2 = "".join(str(b) for b in bits2)

    # 流 = 噪声 + 帧1 + 噪声 + 假帧头 + 噪声 + 帧2 + 噪声 + 残缺帧头。
    # 假帧头：前导第二块幅度失配（残差超门限），任何移位都会检查失败。
    block = R._preamble_block()
    fake = list(block) + [0.3 * x for x in block] + list(R._noise(4, 80, 0.5))
    seg1 = R.transmit(R.fixed_bits(), time_offset=9, cfo=0.11,
                      gain=1.4 * cmath.exp(0.3j), front_scale=0.05,
                      inband_scale=0.05, seed=2024)
    gap1 = R._noise(5, 13, 0.05)
    gap2 = R._noise(6, 21, 0.05)
    seg2 = R.transmit(bits2, time_offset=0, cfo=-0.18,
                      gain=0.8 * cmath.exp(-0.5j), inband_scale=0.05,
                      seed=7)
    gap3 = R._noise(8, 15, 0.05)
    partial = R.transmit(R.fixed_bits(), time_offset=0, cfo=0.05,
                         seed=11, truncate=70)
    stream = list(seg1) + list(gap1) + list(fake) + list(gap2) \
        + list(seg2) + list(gap3) + list(partial)
    true1 = 9
    true2 = len(seg1) + len(gap1) + len(fake) + len(gap2)
    true_partial = true2 + len(seg2) + len(gap3)

    print(f"流总长 {len(stream)} 样本：帧1@{true1}、假帧头(残差超门限)、"
          f"帧2@{true2}、残缺帧头@{true_partial}，帧间夹噪声")

    dec = O.StreamDecoder()
    results = []
    pos = 0
    sizes = [1, 37, 64, 5, 200]  # 任意分块，绝不对齐帧头
    i = 0
    t0 = time.perf_counter()
    while pos < len(stream):
        n = sizes[i % len(sizes)]
        results.extend(dec.feed(stream[pos:pos + n]))
        pos += n
        i += 1
    last, tail = dec.finish()
    results.extend(last)
    dt = time.perf_counter() - t0

    print(f"按块尺寸 {sizes} 喂入，边到边解出 {len(results)} 帧 "
          f"(假帧头已跳过)，耗时 {dt * 1000:.1f} ms")
    ok = True
    for idx, r in enumerate(results):
        truth = (true1, "".join(str(b) for b in R.fixed_bits()))[0] \
            if idx == 0 else true2
        exp_bits = "".join(str(b) for b in R.fixed_bits()) \
            if idx == 0 else expected2
        match = r.bits == exp_bits
        ok = ok and match and r.start == truth
        print(f"  帧{idx + 1}: start={r.start} (真实 {truth})  "
              f"f={r.frequency_offset:+.4f}  residual={r.residual:.4f}  "
              f"位串匹配: {match}")
    print(f"  结束残留: {tail.detail}")
    ok = ok and tail.pending_frame_start == true_partial
    print("  流式演示:", "全部符合预期" if ok else "与预期不符")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main() or stream_demo())
