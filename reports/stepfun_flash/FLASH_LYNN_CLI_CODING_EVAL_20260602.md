# Step 3.7 Flash on Lynn CLI — coding evaluation (2026-06-02)

**Setup:** Lynn CLI (`@lynn/cli 0.80.6`, Mac) → SSH tunnel → Step 3.7 Flash
**Q3_K_M (~88GB, split GGUF)** served by llama.cpp on Spark GB10 `:18099`
(`-ngl 99`, ~26 TPS single-stream bench). CLI provider via env
`LYNN_CLI_BASE_URL`/`LYNN_CLI_MODEL`. Code generation tested with `Lynn -p
"..." --json`; agent mode with `Lynn code -p "..." --json --approval yolo
--sandbox workspace-write`. Each result compiled/run against hidden asserts.

## Results

### Code generation — 9/9 pass@1
| round | tasks | pass |
|---|---|---|
| classic | edit-distance, coin-change, balanced-parens, **dijkstra**, two-sum | 5/5 |
| harder | **regex-match (LeetCode Hard)**, LRU cache, longest-palindrome, JS curry | 4/4 |

Code is idiomatic/optimal, not boilerplate: `edit_distance` used a
space-optimized 1-D rolling DP; `two_sum` a hashmap O(n); `dijkstra` heapq with
early-exit; `regex-match` correct `.`/`*` DP. 1.2–2.4s/task; reasoning scaled
with difficulty (964 chars on dijkstra vs 66 on two-sum).

### Agent mode (`Lynn code`, file edits + shell) — 3/3
- **fizzbuzz** (create file + test + run): wrote both files, ran the test
  (`All tests passed!`), summarized. Notably **self-corrected** a sandbox error
  (absolute path rejected → retried with relative path via the tool ledger).
- **bugfix-quicksort** (fix "drops elements == pivot" bug so a failing test
  passes): final test PASS after 9 tool steps (agent didn't cleanly self-report
  `finished.ok`, took a few extra steps, but the fix was correct).
- **multifile-mathlib** (implement gcd/lcm/is_prime to pass a test): clean,
  5 tool steps, test PASS.

## Conclusion
On algorithm coding + basic agent coding, **Step 3.7 Flash via Lynn CLI is
genuinely strong** (9/9 generation incl. a LeetCode-Hard, 3/3 agent incl.
bug-fix + multi-file, with tool use + self-correction). This **corrects an
earlier data-free guess** that Flash was the weakest coder of the team — on
these tasks it is solid. As Lynn's local product-serving model, its coding
ability is well above "good enough."

**Caveats (honest):** small sample (9 gen + 3 agent), classic/self-contained
tasks (training-data-rich → easier than novel work), Q3_K_M quant. Large
real-repo multi-file refactors, long-horizon, and ambiguous-spec tasks were not
stressed and are the higher bar. Raw artifacts: `reports/stepfun_flash/*.json`.
