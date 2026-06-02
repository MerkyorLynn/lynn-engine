from csvparse import parse_csv

# 1. Simple "a,b,c"
r = parse_csv("a,b,c")
assert r == [["a", "b", "c"]], f"Test 1 failed: {r}"

# 2. Quoted field with comma '"a,b",c'
r = parse_csv('"a,b",c')
assert r == [["a,b", "c"]], f"Test 2 failed: {r}"

# 3. Embedded newline '"line1\nline2",x'
r = parse_csv('"line1\nline2",x')
assert r == [["line1\nline2", "x"]], f"Test 3 failed: {r}"

# 4. Escaped quote '"she said ""hi""",y'
r = parse_csv('"she said ""hi""",y')
assert r == [["she said \"hi\"", "y"]], f"Test 4 failed: {r}"

# 5. Empty fields "a,,c"
r = parse_csv("a,,c")
assert r == [["a", "", "c"]], f"Test 5 failed: {r}"

print("PASS")
