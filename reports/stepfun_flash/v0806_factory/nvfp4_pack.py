import numpy as np

# E2M1 magnitude table (positive magnitudes only)
MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _snap_to_e2m1(x: np.ndarray):
    """Snap each float value to the nearest signed E2M1 grid value."""
    sign = np.sign(x)
    ax = np.abs(x)
    # Find nearest magnitude index for each element
    mag_idx = np.argmin(np.abs(ax[:, None] - MAG[None, :]), axis=1)
    snapped = MAG[mag_idx] * sign
    return snapped, mag_idx


def pack(values: np.ndarray):
    """
    Pack float32 [N, K] (K multiple of 16) into NVFP4.

    Returns:
        packed  : uint8 [N, K//2]  – two 4-bit codes per byte
        scale   : float32 [N, K//16] – one block-scale per 16-element block
    """
    v = values.astype(np.float32)
    N, K = v.shape
    assert K % 16 == 0, "K must be a multiple of 16"

    n_blocks = K // 16
    packed = np.zeros((N, K // 2), dtype=np.uint8)
    scale = np.zeros((N, n_blocks), dtype=np.float32)

    for b in range(n_blocks):
        block = v[:, b * 16:(b + 1) * 16]       # [N, 16]
        blk_max = np.max(np.abs(block), axis=1)  # [N]
        blk_scale = np.where(blk_max == 0.0, 1.0, blk_max / 6.0).astype(np.float32)

        # Normalise and snap
        normed = block / blk_scale[:, None]          # [N, 16]
        snapped, mag_idx = _snap_to_e2m1(normed.reshape(-1))
        snapped = snapped.reshape(N, 16)
        mag_idx = mag_idx.reshape(N, 16)

        # Encode: sign bit (bit 3) | magnitude index (bits 0-2)
        sign_bit = (snapped < 0).astype(np.uint8)
        codes = (sign_bit << 3) | mag_idx.astype(np.uint8)   # [N, 16]

        # Pack two 4-bit codes per byte: LOW nibble = even col, HIGH nibble = odd col
        even = codes[:, 0::2]   # columns 0,2,4,...
        odd  = codes[:, 1::2]   # columns 1,3,5,...
        packed[:, b * 8:(b + 1) * 8] = (odd << 4) | even

        scale[:, b] = blk_scale

    return packed, scale


def dequant(packed: np.ndarray, scale: np.ndarray, global_scale: float = 1.0) -> np.ndarray:
    """
    Inverse of pack: reconstruct float32 [N, K] from packed uint8 and scale.
    """
    N, K2 = packed.shape
    K = K2 * 2
    n_blocks = K // 16

    out = np.zeros((N, K), dtype=np.float32)

    for b in range(n_blocks):
        pb = packed[:, b * 8:(b + 1) * 8]          # [N, 8] uint8 bytes
        even = (pb & 0x0F).astype(np.int32)         # low nibble
        odd  = ((pb >> 4) & 0x0F).astype(np.int32)  # high nibble

        # Interleave back to 16 columns
        codes = np.empty((N, 16), dtype=np.int32)
        codes[:, 0::2] = even
        codes[:, 1::2] = odd

        sign_bit = (codes >> 3) & 1
        mag_idx  = codes & 0x07

        values = MAG[mag_idx] * np.where(sign_bit == 1, -1.0, 1.0)  # [N, 16]
        out[:, b * 16:(b + 1) * 16] = values * scale[:, b:b + 1] * global_scale

    return out


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    N, K = 2, 32
    n_blocks = K // 16

    # Pick a per-block scale for every (row, block) pair
    block_scales = rng.uniform(0.3, 5.0, size=(N, n_blocks)).astype(np.float32)

    v = np.zeros((N, K), dtype=np.float32)
    for b in range(n_blocks):
        signs   = rng.choice([-1.0, 1.0], size=16)
        mag_idxs = rng.integers(0, 8, size=16)
        v[:, b * 16:(b + 1) * 16] = signs * MAG[mag_idxs] * block_scales[:, b:b + 1]

    # Verify round-trip
    packed, scale = pack(v)
    reconstructed = dequant(packed, scale)

    ok = np.allclose(reconstructed, v, atol=1e-4)
    if ok:
        print("PASS")
        exit(0)
    else:
        diff = np.max(np.abs(reconstructed - v))
        print(f"FAIL  maxdiff={diff:.6e}")
        exit(1)
