"""Fixed-input demo for ofdm_recovery.

Synthesizes a frame with an independent, self-contained transmitter (direct
summation DFT -- it does not call the module's make_frame), applies a known
time offset, CFO and complex gain, recovers the bit string, then triggers a
real rejection boundary (truncated frame). Everything is computed live.
"""

import cmath
import math
import time

import ofdm_recovery as ofdm

N, CP, BLOCK = 32, 8, 16
DATA_K = list(range(-6, 0)) + list(range(1, 7))
INV_SQRT2 = 1.0 / math.sqrt(2.0)


def synth_frame(bits):
    """Independent transmitter: preamble + 2 OFDM symbols, from scratch."""
    preamble = [cmath.exp(1j * math.pi * (m * m % 8) / 4.0) for m in range(BLOCK)]
    frame = preamble + preamble
    pos = 0
    for _ in range(2):
        bins = [0j] * N
        bins[(-7) % N] = 1.0
        bins[7 % N] = 1.0
        for k in DATA_K:
            re = 1.0 if bits[pos] == 0 else -1.0
            im = 1.0 if bits[pos + 1] == 0 else -1.0
            bins[k % N] = complex(re * INV_SQRT2, im * INV_SQRT2)
            pos += 2
        td = [sum(bins[k] * cmath.exp(2j * math.pi * k * n / N)
                  for k in range(N)) / N for n in range(N)]
        frame.extend(td[-CP:])
        frame.extend(td)
    return frame


def main():
    t0 = time.perf_counter()

    bits = "010010000110010101101100011011000110111100101111"  # 48 bits
    frame = synth_frame([int(b) for b in bits])

    # Known channel: time offset 41, CFO 0.17 subcarrier spacings, complex gain.
    offset, cfo, gain = 41, 0.17, 1.8 * cmath.exp(1j * 0.9)
    sig = [0j] * offset + [
        gain * s * cmath.exp(2j * math.pi * cfo * i / N)
        for i, s in enumerate(frame)
    ]

    res = ofdm.recover(sig)
    print("=== case 1: clean frame with known offsets ===")
    print(f"start        : {res.start} (true {offset})")
    print(f"cfo          : {res.cfo:+.6f} subcarrier spacings (true {cfo:+.2f})")
    print(f"gain         : {res.gain:.4f} (true {gain:.4f})")
    print(f"residual     : {res.residual:.3e}")
    print(f"candidates   : {len(res.candidates)} (best first: start "
          f"{res.candidates[0].start}, coherence {res.candidates[0].coherence:.6f})")
    print(f"symbols[0:4] : {[complex(round(s.real, 4), round(s.imag, 4)) for s in res.symbols[:4]]}")
    print(f"bits         : {res.bits}")
    print(f"bits match   : {res.bits == bits}")

    print()
    print("=== case 2: truncated frame (rejection boundary) ===")
    bad = sig[:offset + 80]  # preamble intact, second OFDM symbol cut off
    try:
        ofdm.recover(bad)
        print("ERROR: truncated frame was accepted")
    except ofdm.TruncatedFrameError as exc:
        print(f"rejected with TruncatedFrameError: {exc}")
        print(f"  start={exc.start} available={exc.available} required={exc.required}")

    print(f"\nelapsed: {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    main()
