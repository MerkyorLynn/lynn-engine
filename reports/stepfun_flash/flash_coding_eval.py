#!/usr/bin/env python3
"""Lynn CLI + Step 3.7 Flash (local Q3_K_M @ Spark:18099 via tunnel) coding battery.
Each task: Lynn -p (NDJSON) -> extract code block -> compile/run hidden asserts -> pass@1."""
import os, subprocess, json, re, pathlib, time

ENV = {**os.environ,
       "LYNN_CLI_BASE_URL": "http://127.0.0.1:18099/v1",
       "LYNN_CLI_MODEL": "step-3.7-flash",
       "LYNN_CLI_API_KEY": "dummy",
       "LYNN_CLI_PROVIDER": "openai-compatible"}
BIN = "/Users/lynn/Downloads/Lynn/cli/bin/lynn.mjs"

TASKS = [
 {"id":"py-edit-distance","lang":"python",
  "prompt":"用 Python 实现 `def edit_distance(a: str, b: str) -> int`,返回两字符串的 Levenshtein 编辑距离(插入/删除/替换各代价1)。只返回该函数,用 ```python 代码块包裹,不要解释。",
  "test":'assert edit_distance("kitten","sitting")==3\nassert edit_distance("","abc")==3\nassert edit_distance("abc","abc")==0\nassert edit_distance("sunday","saturday")==3\nprint("__PASS__")'},
 {"id":"py-coin-change","lang":"python",
  "prompt":"用 Python 实现 `def min_coins(coins: list[int], amount: int) -> int`,返回凑出 amount 的最少硬币数,无法凑出返回 -1。只返回该函数,```python 包裹。",
  "test":'assert min_coins([1,2,5],11)==3\nassert min_coins([2],3)==-1\nassert min_coins([1],0)==0\nassert min_coins([186,419,83,408],6249)==20\nprint("__PASS__")'},
 {"id":"py-balanced","lang":"python",
  "prompt":"用 Python 实现 `def is_balanced(s: str) -> bool`,判断仅含 ()[]{} 的字符串括号是否正确匹配。只返回该函数,```python 包裹。",
  "test":'assert is_balanced("([]{})")==True\nassert is_balanced("([)]")==False\nassert is_balanced("")==True\nassert is_balanced("(((")==False\nassert is_balanced("{[()()]}")==True\nprint("__PASS__")'},
 {"id":"py-dijkstra","lang":"python",
  "prompt":"用 Python 实现 `def shortest(n: int, edges: list, src: int, dst: int) -> int`,n 个节点(0..n-1),edges 是无向带权边 (u,v,w),返回 src 到 dst 最短距离,不可达返回 -1。只返回该函数,```python 包裹。",
  "test":'assert shortest(5,[(0,1,4),(0,2,1),(2,1,2),(1,3,1),(2,3,5)],0,3)==4\nassert shortest(3,[(0,1,1)],0,2)==-1\nassert shortest(1,[],0,0)==0\nprint("__PASS__")'},
 {"id":"js-two-sum","lang":"js",
  "prompt":"用 JavaScript 实现 `function twoSum(nums, target)`,返回两个下标 [i,j](i<j)使 nums[i]+nums[j]===target(保证恰有一解)。只返回代码,```js 包裹。",
  "test":'const a=JSON.stringify(twoSum([2,7,11,15],9));if(a!=="[0,1]")throw a;const b=JSON.stringify(twoSum([3,2,4],6));if(b!=="[1,2]")throw b;console.log("__PASS__")'},
]

def run_cli(prompt):
    t0=time.time()
    p=subprocess.run(["node",BIN,"-p",prompt,"--json"],env=ENV,capture_output=True,text=True,timeout=360)
    asst=[]; reason=0
    for line in p.stdout.splitlines():
        line=line.strip()
        if not line: continue
        try: ev=json.loads(line)
        except Exception: continue
        ty=ev.get("type")
        if ty=="assistant.delta": asst.append(ev.get("text",""))
        elif ty=="reasoning.delta": reason+=len(ev.get("text",""))
    return "".join(asst), reason, round(time.time()-t0,1), p.stdout[-400:]

def extract(text):
    m=re.search(r"```(?:python|py|js|javascript)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()

results=[]
for t in TASKS:
    try:
        resp,reason,secs,tail=run_cli(t["prompt"])
    except subprocess.TimeoutExpired:
        results.append({"id":t["id"],"pass":False,"err":"CLI_timeout_360s"}); print(f"[{t['id']}] TIMEOUT",flush=True); continue
    code=extract(resp)
    src=code+"\n"+t["test"]
    runner=["python3","-c",src] if t["lang"]=="python" else ["node","-e",src]
    try:
        rp=subprocess.run(runner,capture_output=True,text=True,timeout=30)
        passed="__PASS__" in rp.stdout
        err="" if passed else ((rp.stderr or rp.stdout)[:240])
    except Exception as e:
        passed=False; err=str(e)[:240]
    results.append({"id":t["id"],"lang":t["lang"],"pass":passed,"secs":secs,"reason_chars":reason,"code":code[:500],"err":err})
    print(f"[{t['id']}] pass={passed} secs={secs} reason_chars={reason} err={err[:80]}",flush=True)

n=len(results); pp=sum(1 for r in results if r.get("pass"))
print(f"\n=== Step 3.7 Flash (Q3_K_M) via Lynn CLI: PASS@1 = {pp}/{n} ===",flush=True)
pathlib.Path("/tmp/flash_coding_results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2))
