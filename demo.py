"""固定输入演示：3 抽头 FIR 信道下的正常恢复、擦除边界与实际触发的拒绝。

运行：python demo.py
数据由仓库内确定性独立合成器 tests/reference.py 生成，无网络、无预录结果。
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


def analytic_ratios(taps):
    """演示内独立手算：|H(k)|/sqrt(Σ|h|²)，H(k)=Σ h_l exp(-j2πkl/32)。"""
    rms = math.sqrt(sum(abs(t) ** 2 for t in taps))
    out = {}
    for kc in range(-7, 8):
        if kc == 0:
            continue
        hk = sum(
            taps[l] * cmath.exp(-2j * math.pi * kc * l / 32)
            for l in range(len(taps))
        )
        out[kc] = abs(hk) / rms
    return out


def main() -> int:
    bits = R.fixed_bits()
    expected = "".join(str(b) for b in bits)

    print("=" * 68)
    print("OFDM 恢复演示：最多 3 抽头固定复数 FIR 信道估计 / 均衡 / 深零点擦除")
    print(f"参数: N={O.N_FFT}, CP={O.CP_LEN}, 帧长={O.FRAME_LEN}, "
          f"抽头上限={O.MAX_CHANNEL_TAPS}, 信号长度上限={O.MAX_SIGNAL_LEN}")
    print("=" * 68)

    # 1) 正常：h=(1, -1) 两抽头（±1 载波深衰落），带时偏/频偏/增益/噪声。
    gain = 1.7 * cmath.exp(0.6j)
    taps = (1 + 0j, -1 + 0j)
    signal = R.transmit(
        bits,
        time_offset=23,
        cfo=0.137,
        gain=gain,
        taps=taps,
        front_scale=0.05,
        inband_scale=0.05,
        trailing=25,
        seed=42,
    )
    t0 = time.perf_counter()
    result = O.decode(signal)
    dt = time.perf_counter() - t0

    truth = [gain * 1.0, gain * (-1.0), 0j]
    print("\n[正常] 2 抽头 FIR h=(1, -1)，时偏 23、频偏 +0.137、增益 1.7·e^{j0.6}、噪声 0.05")
    print("  接收长度 =", len(signal), "样本")
    print(f"  恢复起点 start   = {result.start}                (真实 23)")
    print(f"  频偏 f (载波间隔) = {result.frequency_offset:+.6f}      (真实 +0.137000)")
    print(f"  前导残差 residual = {result.residual:.6f}        (0 为完全一致)")
    for l in range(3):
        print(f"  抽头 h[{l}] = {result.channel[l]: .5f}   (真实 {truth[l]: .5f})")
    ratios = analytic_ratios(taps)
    print("  手算 |H(k)|/rms（不含整体增益）：",
          " ".join(f"{ratios[k]:.2f}" for k in range(-6, 0)),
          "|",
          " ".join(f"{ratios[k]:.2f}" for k in range(1, 7)))
    print(f"  深零点擦除载波 (门限 {O.DEFAULT_ERASURE_THRESHOLD}) = {result.erased}"
          "   (手算预测 (-1, 1))")
    print(f"  恢复位串（擦除位标 x，共 48 位）:")
    print("   ", result.bits)
    nonerased_ok = all(
        result.bits[i] == expected[i]
        for i in range(48)
        if result.bits[i] != "x"
    )
    print(f"  非擦除位全部正确: {nonerased_ok}；擦除位数 = "
          f"{result.bits.count('x')} / 48")
    print(f"  解码耗时: {dt * 1000:.1f} ms")

    # 2) 边界：把擦除门限调到 1.0，所有数据载波都判为擦除（不产生硬判决）。
    print("\n[边界] erasure_threshold=1.0：每个数据载波都被标擦除")
    erased_all = O.decode(signal, erasure_threshold=1.0)
    print(f"  擦除载波数 = {len(erased_all.erased)}（共 12 个），"
          f"判决符号全为 None = {all(s is None for s in erased_all.symbols)}")
    print("  位串:", erased_all.bits)
    # 另一极端：门限 0.0，深零点也均衡判决（干净信号下仍正确，仅噪声余量极小）。
    clean_notch = R.transmit(bits, time_offset=7, taps=taps)
    forced = O.decode(clean_notch, erasure_threshold=0.0)
    print("  erasure_threshold=0.0（无噪声）擦除集合 =", forced.erased,
          "，48 位全对 =", forced.bits == expected)

    # 3a) 实际错误：FIR 帧在第二个 OFDM 符号内截断，必须拒绝而非输出半结果。
    bad = R.transmit(bits, time_offset=10, taps=taps, truncate=70)
    print("\n[错误 1] FIR 帧起点 10、需", O.FRAME_LEN,
          "样本，但该起点后仅保留 70 样本（总长度", len(bad), "）")
    try:
        O.decode(bad)
    except O.TruncatedFrameError as exc:
        print(f"  已拒绝: {type(exc).__name__}")
        print(f"  原因: {exc}")
    else:
        print("  错误：截断帧未被拒绝")
        return 1

    # 3b) 实际错误：非法抽头数（契约上限 3），输入即拒绝，无任何部分状态。
    print("\n[错误 2] channel_taps=4 超出契约上限 [1, 3]")
    try:
        O.decode(signal, channel_taps=4)
    except O.ParameterError as exc:
        print(f"  已拒绝: {type(exc).__name__}: {exc}")
    else:
        print("  错误：非法抽头数未被拒绝")
        return 1

    print("\n演示完成：1 个正常恢复（含擦除）+ 门限边界 + 2 个实际拒绝，"
          "均为实时计算。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
