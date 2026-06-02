from kvcalc import kv_bytes

def main():
    # (1) basic calculation
    expected = 2 * 2 * 4 * 128 * 1000 * 2
    result = kv_bytes(2, 4, 128, 1000, 2)
    assert result == expected, f"expected {expected}, got {result}"

    # (2) zero seq_len → 0
    assert kv_bytes(2, 4, 128, 0, 2) == 0

    # (3) negative argument raises ValueError
    try:
        kv_bytes(-1, 4, 128, 1000, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative n_layers")

    print("PASS")

if __name__ == "__main__":
    main()
