import threading
import random
from lru import LRUCache


def test_single_thread_eviction():
    """Put 3 items into cap-2 cache; oldest should be evicted."""
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)          # should evict "a"

    assert cache.get("a") is None, "Expected 'a' to be evicted"
    assert cache.get("b") == 2, "Expected 'b' still present"
    assert cache.get("c") == 3, "Expected 'c' still present"
    assert len(cache) == 2, f"Expected len=2, got {len(cache)}"
    print("  [OK] single-thread eviction")


def test_get_refreshes_recency():
    """Accessing a key should make it most-recently-used."""
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    _ = cache.get("a")          # touch 'a' -> 'a' is now MRU, 'b' is LRU
    cache.put("c", 3)           # should evict 'b'

    assert cache.get("b") is None, "Expected 'b' to be evicted after refresh"
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert len(cache) == 2
    print("  [OK] get refreshes recency")


def test_concurrency():
    """8 threads, 3000 random put/get each, shared cache capacity=64."""
    cache = LRUCache(capacity=64)
    errors = []
    N_THREADS = 8
    OPS = 3000

    def worker(thread_id):
        rng = random.Random(thread_id)
        try:
            for _ in range(OPS):
                key = rng.randint(0, 1023)
                if rng.random() < 0.5:
                    cache.put(key, thread_id)
                else:
                    cache.get(key)
        except Exception as exc:
            errors.append((thread_id, exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        for tid, exc in errors:
            print(f"  [FAIL] thread {tid} raised: {exc}")
        return False

    if len(cache) > 64:
        print(f"  [FAIL] cache size {len(cache)} exceeds capacity 64")
        return False

    print("  [OK] concurrency: no errors, size <= 64")
    return True


def main():
    print("=== LRU Cache Tests ===")
    try:
        test_single_thread_eviction()
        test_get_refreshes_recency()
        ok = test_concurrency()
    except AssertionError as exc:
        print(f"  [FAIL] {exc}")
        ok = False
    except Exception as exc:
        print(f"  [FAIL] unexpected error: {exc}")
        ok = False

    if ok:
        print("PASS")
        raise SystemExit(0)
    else:
        print("FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
