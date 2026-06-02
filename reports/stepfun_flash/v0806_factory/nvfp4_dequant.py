"""
NVFP4 E2M1 block dequantization in pure numpy.

packed  : uint8 ndarray [N, K//2]
           each byte packs two 4-bit E2M1 codes
           LOW  nibble (bits 0-3)  -> even column  (index 2*j)
           HIGH nibble (bits 4-7)  -> odd  column  (index 2*j+1)

4-bit code layout (bits 3..0):
  bit 3 = sign  (1 -> negative)
  bits 0..2 = magnitude index into E2M1 table

E2M1 magnitude table (8 entries, indexed 0..7):
  [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

scale      : float32 ndarray [N, K//16]  — one scale per 16-element block along K
global_scale: python float

Returns float32 [N, K]:
  value = code_magnitude * (-1 if sign) * block_scale * global_scale
  where block_scale[n, k] = scale[n, k // 16]
"""

import numpy as np

# E2M1 magnitude table — index = bits 0..2 of the 4-bit code
E2M1_TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def dequant(packed: np.ndarray, scale: np.ndarray, global_scale: float) -> np.ndarray:
    """Dequantize NVFP4 packed data back to float32.

    Parameters
    ----------
    packed : uint8 ndarray of shape [N, K//2]
    scale  : float32 ndarray of shape [N, K//16]
    global_scale : float

    Returns
    -------
    float32 ndarray of shape [N, K]
    """
    packed = np.asarray(packed, dtype=np.uint8)
    scale = np.asarray(scale, dtype=np.float32)

    N, packed_cols = packed.shape
    K = packed_cols * 2  # total columns after unpacking

    # --- unpack 4-bit codes ---
    # low nibble  -> even column (2*j)
    # high nibble -> odd  column (2*j+1)
    low_codes  = (packed        ) & 0x0F   # shape [N, K//2]
    high_codes = (packed >> 4   ) & 0x0F   # shape [N, K//2]

    # interleave: [low_0, high_0, low_1, high_1, ...]
    codes = np.empty((N, K), dtype=np.uint8)
    codes[:, 0::2] = low_codes
    codes[:, 1::2] = high_codes   # shape [N, K]

    # --- decode sign and magnitude ---
    sign_mask  = codes >> 3          # 1 if negative, 0 if positive  [N, K]
    mag_index  = codes & 0x07        # magnitude index 0..7           [N, K]

    magnitude  = E2M1_TABLE[mag_index]          # [N, K]  float32
    sign       = np.where(sign_mask == 1, -1.0, 1.0).astype(np.float32)  # [N, K]

    # --- block scale: one scale per 16-element block along K ---
    block_idx  = np.arange(K, dtype=np.int32) // 16          # [K]
    block_scale = scale[:, block_idx]                          # [N, K]

    # --- final value ---
    result = magnitude * sign * block_scale * global_scale     # [N, K]
    return result.astype(np.float32)


# ── self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Tiny deterministic case: N=1, K=16
    # packed shape [1, 8], scale shape [1, 1]
    N, K = 1, 16

    # packed bytes: each byte = (odd_col << 4) | even_col
    # We choose codes so the expected output is easy to verify by hand.
    #
    # codes (4-bit) for columns 0..15:
    #   col  0: code=0x1  -> sign=0 mag_idx=1 -> +0.5
    #   col  1: code=0x9  -> sign=1 mag_idx=1 -> -0.5
    #   col  2: code=0x2  -> sign=0 mag_idx=2 -> +1.0
    #   col  3: code=0xA  -> sign=1 mag_idx=2 -> -1.0
    #   col  4: code=0x3  -> sign=0 mag_idx=3 -> +1.5
    #   col  5: code=0xB  -> sign=1 mag_idx=3 -> -1.5
    #   col  6: code=0x4  -> sign=0 mag_idx=4 -> +2.0
    #   col  7: code=0xC  -> sign=1 mag_idx=4 -> -2.0
    #   col  8: code=0x5  -> sign=0 mag_idx=5 -> +3.0
    #   col  9: code=0xD  -> sign=1 mag_idx=5 -> -3.0
    #   col 10: code=0x6  -> sign=0 mag_idx=6 -> +4.0
    #   col 11: code=0xE  -> sign=1 mag_idx=6 -> -4.0
    #   col 12: code=0x7  -> sign=0 mag_idx=7 -> +6.0
    #   col 13: code=0xF  -> sign=1 mag_idx=7 -> -6.0
    #   col 14: code=0x0  -> sign=0 mag_idx=0 -> +0.0
    #   col 15: code=0x8  -> sign=1 mag_idx=0 -> -0.0  (=0.0)
    #
    # Packed bytes (low nibble = even, high nibble = odd):
    #   j=0: even=0x1 odd=0x9  -> 0x91
    #   j=1: even=0x2 odd=0xA  -> 0xA2
    #   j=2: even=0x3 odd=0xB  -> 0xB3
    #   j=3: even=0x4 odd=0xC  -> 0xC4
    #   j=4: even=0x5 odd=0xD  -> 0xD5
    #   j=5: even=0x6 odd=0xE  -> 0xE6
    #   j=6: even=0x7 odd=0xF  -> 0xF7
    #   j=7: even=0x0 odd=0x8  -> 0x80

    packed = np.array([[0x91, 0xA2, 0xB3, 0xC4, 0xD5, 0xE6, 0xF7, 0x80]],
                      dtype=np.uint8)

    # One block scale for all 16 columns (K//16 = 1 block)
    scale = np.array([[2.0]], dtype=np.float32)

    global_scale = 3.0

    # ── compute expected output by hand ──────────────────────────────────────
    # For each column k:
    #   code  = 4-bit value
    #   sign  = -1 if bit3==1 else +1
    #   mag   = E2M1_TABLE[code & 0x7]
    #   block = scale[0, k//16] = scale[0,0] = 2.0
    #   value = sign * mag * block * global_scale
    #
    # col  0: code=0x1 sign=+1 mag=0.5  -> +0.5  * 2.0 * 3.0 = +3.0
    # col  1: code=0x9 sign=-1 mag=0.5  -> -0.5  * 2.0 * 3.0 = -3.0
    # col  2: code=0x2 sign=+1 mag=1.0  -> +1.0  * 2.0 * 3.0 = +6.0
    # col  3: code=0xA sign=-1 mag=1.0  -> -1.0  * 2.0 * 3.0 = -6.0
    # col  4: code=0x3 sign=+1 mag=1.5  -> +1.5  * 2.0 * 3.0 = +9.0
    # col  5: code=0xB sign=-1 mag=1.5  -> -1.5  * 2.0 * 3.0 = -9.0
    # col  6: code=0x4 sign=+1 mag=2.0  -> +2.0  * 2.0 * 3.0 = +12.0
    # col  7: code=0xC sign=-1 mag=2.0  -> -2.0  * 2.0 * 3.0 = -12.0
    # col  8: code=0x5 sign=+1 mag=3.0  -> +3.0  * 2.0 * 3.0 = +18.0
    # col  9: code=0xD sign=-1 mag=3.0  -> -3.0  * 2.0 * 3.0 = -18.0
    # col 10: code=0x6 sign=+1 mag=4.0  -> +4.0  * 2.0 * 3.0 = +24.0
    # col 11: code=0xE sign=-1 mag=4.0  -> -4.0  * 2.0 * 3.0 = -24.0
    # col 12: code=0x7 sign=+1 mag=6.0  -> +6.0  * 2.0 * 3.0 = +36.0
    # col 13: code=0xF sign=-1 mag=6.0  -> -6.0  * 2.0 * 3.0 = -36.0
    # col 14: code=0x0 sign=+1 mag=0.0  -> +0.0  * 2.0 * 3.0 = +0.0
    # col 15: code=0x8 sign=-1 mag=0.0  -> -0.0  * 2.0 * 3.0 = -0.0  (=0.0)
    expected = np.array([[+3.0, -3.0, +6.0, -6.0,
                          +9.0, -9.0, +12.0, -12.0,
                          +18.0, -18.0, +24.0, -24.0,
                          +36.0, -36.0, +0.0, +0.0]], dtype=np.float32)

    result = dequant(packed, scale, global_scale)

    ok = np.allclose(result, expected, atol=1e-6)
    if ok:
        print("PASS")
        raise SystemExit(0)
    else:
        diff = np.abs(result - expected)
        print("FAIL")
        print("result :", result)
        print("expected:", expected)
        print("max diff:", diff.max())
        raise SystemExit(1)
