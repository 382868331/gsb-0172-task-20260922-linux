"""软解调迭代演示：QPSK/16QAM 每比特 LLR（正值偏向比特 0）。

运行：python demo_iteration.py
全部数据由仓库内确定性代码实时生成，无网络、无 sleep、无预录结果。
"""
from __future__ import annotations

import cmath
import math
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

    print("=" * 68)
    print("OFDM 软解调演示：QPSK/16QAM 每比特 LLR（正值偏向比特 0）")
    print("=" * 68)

    # 1) 正常帧：含噪信道下 decode 同时给出硬判决与每比特 LLR。
    t0 = time.perf_counter()
    signal = R.transmit(
        bits,
        time_offset=23,
        cfo=0.137,
        gain=1.7 * cmath.exp(0.6j),
        front_scale=0.05,
        inband_scale=0.05,
        trailing=25,
        seed=42,
    )
    result = O.decode(signal, noise_variance=0.02)
    dt = time.perf_counter() - t0
    print("\n[正常帧] 时偏 23 / 频偏 +0.137 / 复增益 1.7*exp(j0.6)，含噪")
    print(f"  起点={result.start}  频偏={result.frequency_offset:+.6f}  "
          f"残差={result.residual:.6f}")
    print(f"  硬判决位串匹配: {result.bits == expected}（位序与旧版一致）")
    print(f"  数据比特 LLR 共 {len(result.llrs)} 个（导频/DC/保护带不在内），"
          f"擦除标志 {len(result.erasures)} 个，全为 False: "
          f"{not any(result.erasures)}")
    print("  前 8 个比特:  位  ->  LLR")
    for i in range(8):
        print(f"    bit[{i:2d}] = {result.bits[i]}  ->  "
              f"{result.llrs[i]:+10.3f}")
    mags = [abs(v) for v in result.llrs]
    print(f"  |LLR| 最小/最大 = {min(mags):.3f} / {max(mags):.3f}")
    print(f"  解码+软解调耗时: {dt * 1000:.1f} ms")

    # 2) 边界演示：观测扫过判决边界，LLR 连续趋近并穿过 0。
    print("\n[边界] QPSK 观测沿实轴穿过判决边界（虚部固定在星座中心）")
    var = 0.1
    for re in (0.7, 0.3, 0.05, 0.0, -0.05, -0.3, -0.7):
        y = complex(re, 1.0 / math.sqrt(2.0))
        soft = O.soft_demodulate([y], modulation="qpsk", noise_variance=var)
        print(f"  y = {re:+.2f} + 0.707j  ->  LLR(b0) = {soft.llrs[0]:+8.3f}"
              f"   LLR(b1) = {soft.llrs[1]:+8.3f}")
    print("  -> 越靠近边界（实部=0）LLR 越接近 0，过边界后变号")

    # 3) 16QAM：星座中心、Gray 中点与深衰落/擦除。
    print("\n[16QAM] 显式 Gray 表软解调")
    center = O.QAM16_GRAY[(0, 1, 0, 1)]
    soft = O.soft_demodulate([center], modulation="16qam", noise_variance=var)
    print(f"  星座中心 {center:.3f}（位组 0101）-> "
          f"LLR = {[round(v, 3) for v in soft.llrs]}")
    mid = complex(0.0, center.imag)
    soft = O.soft_demodulate([mid], modulation="16qam", noise_variance=var)
    print(f"  实轴中点 {mid:.3f}            -> "
          f"LLR = {[round(v, 3) for v in soft.llrs]}（b0 精确为 0）")
    faded = O.soft_demodulate(
        [1e-3 * center], modulation="16qam", noise_variance=var,
        gains=[1e-3 + 0j],
    )
    print(f"  深衰落 |h|=1e-3 同一符号     -> "
          f"LLR = {[round(v, 6) for v in faded.llrs]}（趋近 0）")
    erased = O.soft_demodulate(
        [center], modulation="16qam", noise_variance=var, gains=[0j],
    )
    print(f"  零增益（擦除）               -> "
          f"LLR = {list(erased.llrs)}，擦除标志 = {list(erased.erasures)}")

    # 4) 显式擦除：LLR=0 且保留擦除标志，不硬判成全 0。
    soft = O.soft_demodulate(
        [complex(1 / math.sqrt(2), 1 / math.sqrt(2))],
        modulation="qpsk", noise_variance=var, erased=[True],
    )
    print("\n[擦除] 显式擦除一个 QPSK 符号 -> "
          f"LLR = {list(soft.llrs)}，erasures = {list(soft.erasures)}")
    print("  -> 擦除位不被硬判成全 0，而是显式标记为未知")

    print("\n演示完成：正常帧软解调 + 判决边界/深衰落/擦除边界，均为实时计算。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
