"""保留子载波 PAPR 压低演示：真实迭代计算 + 一个边界情况。

运行：python demo_iteration.py
全部为确定性输入、实时计算，无 sleep、无预录结果。
"""
from __future__ import annotations

import math
import os
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import ofdm_recovery as O  # noqa: E402

Q = 1.0 / math.sqrt(2.0)


def db(papr_linear: float) -> float:
    return 10.0 * math.log10(papr_linear)


def main() -> int:
    print("=" * 64)
    print("保留子载波迭代投影压低 OFDM 峰均比 —— 实时计算演示")
    print(f"N={O.N_FFT}，可选保留频点（除数据/导频/DC 外）: "
          f"{list(O.FREE_CARRIERS)}")
    print("=" * 64)

    # 确定性符号串：多数数据载波同相，时域在 n=0 附近叠出明显尖峰。
    signs = [1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1]
    symbols = [complex(Q * s, Q * s) for s in signs]

    # 1) 正常用例：17 个保留音调、能量预算 8.0、目标 4 dB。
    reserved = list(O.FREE_CARRIERS)
    t0 = time.perf_counter()
    res = O.reduce_papr_tone_reservation(
        symbols, reserved, target_papr_db=4.0, max_reserved_energy=8.0
    )
    dt = time.perf_counter() - t0
    print("\n[正常用例] 保留音调 17 个，能量预算 8.0，目标 PAPR 4.0 dB")
    print(f"  初始 PAPR = {db(res.papr_history[0]):.3f} dB")
    print(f"  最终 PAPR = {db(res.papr_history[-1]):.3f} dB  "
          f"(达标: {res.achieved}, 迭代 {res.iterations} 轮, 耗时 {dt*1000:.1f} ms)")
    print(f"  保留音调实际总能量 = {res.reserved_energy:.4f} (预算 8.0)")
    print("  每轮 PAPR (dB):")
    hist = [f"{db(p):.3f}" for p in res.papr_history]
    for i in range(0, len(hist), 8):
        print("   ", "  ".join(hist[i:i + 8]))
    print("  每轮保留能量:")
    eh = [f"{e:.3f}" for e in res.reserved_energy_history]
    for i in range(0, len(eh), 8):
        print("   ", "  ".join(eh[i:i + 8]))
    # 独立复核：重新 DFT 检查数据/导频/DC 未被破坏。
    check = O.dft(res.time_signal)
    signed = list(O.SIGNED_DATA_CARRIERS)
    worst = 0.0
    for k in list(O.DATA_CARRIERS) + list(O.PILOT_CARRIERS) + [0]:
        if k == 0:
            expected = 0j
        elif k in O.PILOT_VALUES:
            expected = O.PILOT_VALUES[k]
        else:
            kc = k if k < 16 else k - 32
            expected = symbols[signed.index(kc)]
        worst = max(worst, abs(check[k] - expected))
    print(f"  复核：数据/导频/DC 频点最大偏差 = {worst:.2e}（应为 0 量级）")

    # 2) 边界情况：能量预算为 0 —— 保留音调被强制为 0，
    #    峰均比不可能改善，必须如实报告未达标，且载荷完全不变。
    t0 = time.perf_counter()
    res0 = O.reduce_papr_tone_reservation(
        symbols, [8, 12, 16, 20, 24],
        target_papr_db=3.0, max_reserved_energy=0.0,
    )
    dt = time.perf_counter() - t0
    print("\n[边界情况] 保留音调 5 个，但能量预算 = 0，目标 3.0 dB")
    print(f"  最终 PAPR = {db(res0.papr_history[-1]):.3f} dB  "
          f"(达标: {res0.achieved}, 迭代 {res0.iterations} 轮, "
          f"耗时 {dt*1000:.1f} ms)")
    print(f"  保留音调实际总能量 = {res0.reserved_energy}（预算 0，必须为 0）")
    if res0.achieved or res0.reserved_energy != 0.0:
        print("  错误：零预算下不应达标或动用能量")
        return 1
    print("  如实报告：预算为 0 时没有自由度，未达标，数据频点保持不变。")

    print("\n演示完成：1 个正常压低 + 1 个零预算边界，均为实时计算。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
