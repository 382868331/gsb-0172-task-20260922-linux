"""独立参考合成器（测试专用）。

本文件用与 ``ofdm_recovery.core`` 互相独立的代码直接构造合成帧，
并施加已知时偏（前置样本数）、频偏（载波间隔为单位）与复增益。
测试期望值（原始位串、起点、频偏、增益）来自这里的**输入参数**，
不是从被测核心反推得到的。本文件只使用 Python 标准库。
"""

from __future__ import annotations

import cmath
import math
import random

N = 32
CP = 8
BLOCK = 16
N_SYM = 2
FRAME_LEN = 2 * BLOCK + N_SYM * (N + CP)  # 112

SQRT1_2 = 1.0 / math.sqrt(2.0)


def qpsk_modulate(bits):
    """独立 QPSK 映射：位对 (b0,b1) -> (±1±j)/√2（0 为正，1 为负）。"""
    out = []
    for i in range(0, len(bits), 2):
        re = SQRT1_2 if bits[i] == 0 else -SQRT1_2
        im = SQRT1_2 if bits[i + 1] == 0 else -SQRT1_2
        out.append(complex(re, im))
    return out


def _idft(X):
    """朴素 IDFT：x[n] = (1/N) Σ_k X[k] exp(+j2πkn/N)。"""
    n = len(X)
    return [
        sum((X[k] * cmath.exp(2j * math.pi * k * m / n) for k in range(n)), 0j)
        / n
        for m in range(n)
    ]


def fixed_bits():
    """48 个固定测试位（确定性，不依赖随机源）。"""
    return [
        0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1,
        1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0,
        1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0,
        0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1,
    ]


def _preamble_block():
    """独立确定的 16 点复前导块（0.8 幅 chirp-like 序列）。"""
    return [
        0.8 * cmath.exp(2j * math.pi * (m * (m + 1) // 2) / BLOCK)
        for m in range(BLOCK)
    ]


def build_frame(bits, *, zero_pilots=False):
    """按题面布局直接合成一帧（112 样本），未经信道、无时偏。"""
    assert len(bits) == 48
    block = _preamble_block()
    frame = list(block) + list(block)

    syms = qpsk_modulate(bits)
    for s in range(N_SYM):
        X = [0j] * N
        carriers = [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
        for j, kc in enumerate(carriers):
            X[kc % N] = syms[s * 12 + j]
        if not zero_pilots:
            X[7] = 1 + 0j    # +7 导频
            X[-7 % N] = -1 + 0j  # -7 导频（索引 25）
        time = _idft(X)
        frame.extend(time[-CP:])   # 循环前缀 = 末尾 8 点
        frame.extend(time)
    assert len(frame) == FRAME_LEN
    return frame


def _noise(seed, length, scale):
    rng = random.Random(seed)
    out = []
    for _ in range(length):
        # 两个均匀分布之和 -> 近似三角分布的复噪声
        re = (rng.random() + rng.random() - 1.0) * scale
        im = (rng.random() + rng.random() - 1.0) * scale
        out.append(complex(re, im))
    return out


def transmit(
    bits,
    *,
    time_offset=0,
    cfo=0.0,
    gain=1.0 + 0j,
    front_scale=0.0,
    inband_scale=0.0,
    trailing=0,
    zero_pilots=False,
    seed=1234,
    truncate=None,
):
    """施加已知信道并返回接收信号。

    time_offset: 帧前噪声样本数（时偏=该整数）。
    cfo: 频偏，单位载波间隔，乘子 exp(j2π f n/32)（n 为帧内序号）。
    gain: 单一复增益。
    front_scale: 前置噪声幅度；inband_scale: 叠加在帧上的噪声幅度。
    trailing: 帧后噪声样本数；truncate: 从帧起点起保留的总样本上限。
    """
    frame = build_frame(bits, zero_pilots=zero_pilots)
    gained = []
    for n, x in enumerate(frame):
        y = x * complex(gain) * cmath.exp(2j * math.pi * cfo * n / N)
        gained.append(y)

    front = _noise(seed, time_offset, front_scale) if front_scale else [
        0j
    ] * time_offset
    sig = list(front)
    if inband_scale:
        nb = _noise(seed + 7, len(gained), inband_scale)
        gained = [y + z for y, z in zip(gained, nb)]
    sig.extend(gained)
    if trailing:
        sig.extend(_noise(seed + 13, trailing, front_scale or inband_scale or 0.1))
    if truncate is not None:
        sig = sig[:time_offset + truncate]
    return sig
