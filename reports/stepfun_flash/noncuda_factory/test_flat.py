import sys
sys.path.insert(0, ".")
from flat import flatten

passed = 0
failed = 0

def check(label, got, expected):
    global passed, failed
    if got == expected:
        print(f"  PASS  {label}")
        passed += 1
    else:
        print(f"  FAIL  {label}")
        print(f"    expected: {expected}")
        print(f"    got:      {got}")
        failed += 1

# ── tests ────────────────────────────────────────────────────────────────
print("flat.py / test_flat.py")

# 1. empty list
check("empty list", flatten([]), [])

# 2. flat list
check("flat list", flatten([1, 2, 3]), [1, 2, 3])

# 3. single nested list
check("one level nesting", flatten([1, [2, 3], 4]), [1, 2, 3, 4])

# 4. deep nesting
check("deep nesting", flatten([1, [2, [3, [4, 5]], 6], 7]), [1, 2, 3, 4, 5, 6, 7])

# 5. tuples mixed with lists
check("mixed lists and tuples", flatten([1, (2, [3, (4,)]), 5]), [1, 2, 3, 4, 5])

# 6. all empty containers
check("all empty containers", flatten([[], (), [(), []]]), [])

# 7. single scalar
check("single scalar", flatten(42), [42])

# 8. strings are scalars (not iterated)
check("string scalar", flatten(["hello", [1, 2]]), ["hello", 1, 2])

# 9. heterogeneous types
check("heterogeneous", flatten([1, "a", [2.5, (None, True)], False]), [1, "a", 2.5, None, True, False])

# 10. deeply mixed
check("deeply mixed", flatten(((1, [2, ((3,), [4, 5])], 6), 7)), [1, 2, 3, 4, 5, 6, 7])

# ── summary ──────────────────────────────────────────────────────────────
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
