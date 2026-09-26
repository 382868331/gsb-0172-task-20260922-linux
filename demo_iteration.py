"""软解调迭代演示：正常帧 LLR 输出 + 判决边界/深衰落/擦除边界样例。

运行：python demo_iteration.py
数据由仓库内确定性独立合成器 tests/reference.py 生成，全部实时计算，
无 sleep、无预录结果（实测远低于 90 秒预算）。
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
    print("OFDM 软解调（每比特 LLR，正值偏向位 0）—— 迭代演示")
    print("=" * 64)
    t_all = time.perf_counter()

    # 1) 正常帧： decode_soft 在硬判决之后追加每比特 LLR。
    gain = 1.7 * cmath.exp(0.6j)
    signal = R.transmit(
        bits, time_offset=23, cfo=0.137, gain=gain,
        front_scale=0.05, inband_scale=0.05, trailing=25, seed=42,
    )
    t0 = time.perf_counter()
    soft = O.decode_soft(signal, 0.02)
    dt = time.perf_counter() - t0
    print("\n[正常帧] QPSK 软解调（噪声方差 0.02）")
    print(f"  起点={soft.start} 频偏={soft.frequency_offset:+.6f}")
    print(f"  位串匹配: {soft.bits == expected}  (硬判决位序与 decode 完全一致)")
    print(f"  LLR 数量: {len(soft.llrs)}（仅数据载波，导频/DC/保护带不混入）")
    print(f"  擦除标志: {sum(soft.erasures)} / {len(soft.erasures)} 个符号")
    print("  前 12 位 (位, LLR):")
    for i in range(12):
        print(f"    位{i:02d} = {soft.bits[i]}  LLR = {soft.llrs[i]:+10.4f}")
    sign_ok = all((v > 0) == (b == "0") for v, b in zip(soft.llrs, soft.bits))
    print("  全部 48 位 LLR 符号与硬判决一致:", sign_ok)
    print(f"  软解调耗时: {dt * 1000:.1f} ms")

    # 2) 16QAM 星座软解调（同一接收管线，每符号 4 位）。
    soft16 = O.decode_soft(signal, 0.02, modulation="qam16")
    print("\n[16QAM] 同一帧按 16QAM Gray 表软解调")
    print(f"  LLR 数量: {len(soft16.llrs)}（24 符号 x 4 位）")
    print("  前 8 个 LLR:", "".join(f"{v:+.3f}" for v in soft16.llrs[:8]))

    # 3) 边界样例：软值在判决边界附近趋近 0。
    print("\n[边界] 判决边界附近 LLR -> 0（QPSK，噪声方差 0.1）")
    print("  星座中心 (硬判自信):", O.soft_demod(complex(2**-0.5, 2**-0.5), 0.1))
    print("  实部接近边界 0.05 :", O.soft_demod(complex(0.05, 2**-0.5), 0.1))
    print("  恰在边界 (实部=0)  :", O.soft_demod(complex(0.0, 2**-0.5), 0.1))
    print("  极低信噪比 σ²=1e6  :", O.soft_demod(complex(2**-0.5, -(2**-0.5)), 1e6))

    # 4) 深衰落与零增益擦除。
    print("\n[边界] 信道衰落对 LLR 的影响（y = 1+j，σ² = 0.1）")
    for g in (1.0 + 0j, 0.1 + 0j, 1e-3 + 0j):
        llrs, erased = O.soft_demod_equalized(1 + 1j, g, 0.1)
        print(f"  增益 |g|={abs(g):<6} LLR={[round(v, 4) for v in llrs]} 擦除={erased}")
    llrs, erased = O.soft_demod_equalized(1 + 1j, 0j, 0.1)
    print(f"  零增益        LLR={llrs} 擦除={erased}（不硬判成全 0 位）")

    print(f"\n演示完成，总耗时 {(time.perf_counter() - t_all) * 1000:.1f} ms，均为实时计算。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
