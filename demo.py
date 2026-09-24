"""固定输入演示：真实计算正常恢复、深零点擦除边界与实际触发的拒绝。

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
          f"信号长度上限={O.MAX_SIGNAL_LEN}, FIR 抽头上限={O.MAX_CHANNEL_TAPS}")
    print("=" * 64)

    # 1) 正常帧：已知时偏 23、频偏 +0.137、3 抽头复数 FIR、确定性噪声。
    taps = [0.8 + 0.3j, -0.4 + 0.5j, 0.1 - 0.2j]
    signal = R.transmit(
        bits,
        time_offset=23,
        cfo=0.137,
        taps=taps,
        front_scale=0.05,
        inband_scale=0.05,
        trailing=25,
        seed=42,
    )
    t0 = time.perf_counter()
    result = O.decode(signal)
    dt = time.perf_counter() - t0

    print("\n[正常帧·3 抽头 FIR] 接收长度 =", len(signal), "样本")
    print(f"  恢复起点 start   = {result.start}        (真实 23)")
    print(f"  频偏 f (载波间隔) = {result.frequency_offset:+.6f}  (真实 +0.137000)")
    print(f"  训练拟合残差 J    = {result.channel.fit_residual:.2e}  (0 为完全一致)")
    print("  估计抽头 (实时计算) vs 真实抽头:")
    for got, want in zip(result.channel.taps, taps):
        print(f"    {got:+.5f}   vs   {want:+.5f}")
    print(f"  擦除载波: {result.erased_carriers or '无'}")
    print(f"  恢复位串 ({len(result.bits)} 位):")
    print("   ", result.bits)
    print("  位串匹配:", result.bits == expected)
    print(f"  解码耗时: {dt * 1000:.1f} ms")

    # 2) 边界：2 抽头信道在数据载波 +3 上有解析精确零点
    #    h = [1, -exp(j2π·3/32)]  =>  H(3) = 0。
    #    该载波两个符号必须被标为擦除（None/'x'），其余 44 位仍全部恢复。
    null_taps = [1 + 0j, -cmath.exp(2j * cmath.pi * 3 / O.N_FFT)]
    sig_null = R.transmit(bits, time_offset=9, taps=null_taps)
    r2 = O.decode(sig_null)
    unerased_ok = all(
        r2.bits[i] == expected[i]
        for i in range(48)
        if i // 2 not in r2.erased_positions
    )
    print("\n[边界·深零点擦除] 信道 [1, -exp(j2π·3/32)]，H(+3)=0（解析）")
    print(f"  擦除载波: {r2.erased_carriers}  (符号位置 {r2.erased_positions})")
    print(f"  估计频响 |H(+3)| = {abs(r2.channel.frequency_response[3]):.2e}"
          f"  (RMS 增益 {r2.channel.rms_gain:.4f})")
    print(f"  恢复位串 ({len(r2.bits)} 位, 'x'=擦除):")
    print("   ", r2.bits)
    print("  未擦除位全部正确:", unerased_ok)

    # 3) 坏帧边界：前导可定位，但信号在第二个 OFDM 符号结束前被截断，
    #    接收机必须实际抛出 TruncatedFrameError 拒绝，而不是输出半结果。
    bad = R.transmit(bits, time_offset=10, taps=[1 + 0j, 0.5j, -0.3],
                     truncate=70)
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

    # 4) 非法参数：erasure_threshold 必须为正有限数，实际抛出 ParameterError。
    print("\n[非法参数] erasure_threshold=0.0 调用 decode")
    try:
        O.decode(sig_null, erasure_threshold=0.0)
    except O.ParameterError as exc:
        print(f"  已拒绝: {type(exc).__name__}")
        print(f"  原因: {exc}")
    else:
        print("  错误：非法门限未被拒绝")
        return 1

    print("\n演示完成：1 个正常恢复 + 1 个深零点擦除边界 + 2 个实际触发的"
          "拒绝，均为实时计算。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
