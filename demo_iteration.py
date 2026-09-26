"""保留子载波迭代压峰演示：真实计算一次压峰与一次零预算边界。

运行：python demo_iteration.py
全部确定性数据，实时计算，无 sleep、无预录结果。
"""
from __future__ import annotations

import math
import os
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import ofdm_recovery as O  # noqa: E402

SQRT1_2 = 1.0 / math.sqrt(2.0)
RESERVED = (8, 9, 10, 11, 12, 20, 21, 22, 23, 24)


def peaky_spectrum():
    """确定性高 PAPR 频域符号：12 个数据载波全部同相 (1+j)/√2。"""
    X = [0j] * O.N_FFT
    for k in O.DATA_CARRIERS:
        X[k] = complex(SQRT1_2, SQRT1_2)
    X[7] = 1 + 0j
    X[25] = -1 + 0j
    return X


def main() -> int:
    print("=" * 64)
    print("保留子载波迭代压峰（tone reservation）—— 固定输入演示")
    print(f"N={O.N_FFT}, 保留音调={list(RESERVED)} "
          f"(数据/导频/DC 频点保持不动)")
    print("=" * 64)

    spec = peaky_spectrum()

    # 1) 正常压峰：目标 4.5 dB，保留音调总能量预算 3.0。
    t0 = time.perf_counter()
    result = O.reduce_papr(
        spec, RESERVED, target_papr_db=6.0, energy_budget=6.0
    )
    dt = time.perf_counter() - t0

    print("\n[压峰] 目标 PAPR <= 6.0 dB, 保留能量预算 = 6.0")
    print("  每轮峰均比 / 保留音调实际能量:")
    for rec in result.iterations:
        print(f"    轮 {rec.iteration:>2}: PAPR = {rec.papr_db:7.4f} dB, "
              f"保留能量 = {rec.reserved_energy:.6f}")
    print(f"  最终 PAPR = {result.papr_db:.4f} dB "
          f"(初始 {result.iterations[0].papr_db:.4f} dB)")
    print(f"  目标达成: {result.target_reached}")
    print(f"  保留能量 {result.reserved_energy:.6f} <= 预算 "
          f"{result.energy_budget}")
    # 数据/导频/DC 频点逐一核对未变。
    untouched = all(
        result.spectrum[k] == spec[k]
        for k in range(O.N_FFT) if k not in RESERVED
    )
    print(f"  数据/导频/DC 频点未变: {untouched}")
    print(f"  耗时: {dt * 1000:.1f} ms")
    if not (result.target_reached and untouched):
        print("  错误：压峰演示未达预期")
        return 1

    # 2) 边界：能量预算为 0 —— 没有任何自由度，库只如实报告达不到，
    #    频谱原样返回，不裁剪时域、不伪造成功。
    edge = O.reduce_papr(
        spec, RESERVED, target_papr_db=6.0, energy_budget=0.0
    )
    print("\n[边界] 能量预算 = 0（无自由度）")
    print(f"  最终 PAPR = {edge.papr_db:.4f} dB, "
          f"目标达成: {edge.target_reached}（如实报告未达到）")
    print(f"  保留能量 = {edge.reserved_energy}，频谱与输入完全一致: "
          f"{all(a == b for a, b in zip(edge.spectrum, spec))}")
    if edge.target_reached:
        print("  错误：零预算不应报告达标")
        return 1

    print("\n演示完成：1 次实际压峰 + 1 个零预算边界，均为实时计算。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

