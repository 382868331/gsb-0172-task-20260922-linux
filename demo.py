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


if __name__ == "__main__":
    raise SystemExit(main())
