def kv_bytes(n_layers, n_kv_heads, head_dim, seq_len, dtype_bytes):
    """Return total KV-cache bytes.

    Formula: 2 * n_layers * n_kv_heads * head_dim * seq_len * dtype_bytes
    (factor 2 accounts for both K and V caches).
    """
    if any(x < 0 for x in (n_layers, n_kv_heads, head_dim, seq_len, dtype_bytes)):
        raise ValueError("all arguments must be non-negative")
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * dtype_bytes
