import os, subprocess, json, re, pathlib, time
ENV={**os.environ,"LYNN_CLI_BASE_URL":"http://127.0.0.1:18099/v1","LYNN_CLI_MODEL":"step-3.7-flash","LYNN_CLI_API_KEY":"dummy","LYNN_CLI_PROVIDER":"openai-compatible"}
BIN="/Users/lynn/Downloads/Lynn/cli/bin/lynn.mjs"
TASKS=[
 {"id":"py-regex-match","lang":"python",
  "prompt":"用 Python 实现 `def is_match(s: str, p: str) -> bool`,支持正则 . (任意单字符)和 * (前一字符0次或多次),要求 p 完整匹配 s。只返回该函数,```python 包裹。",
  "test":'assert is_match("aa","a")==False\nassert is_match("aa","a*")==True\nassert is_match("ab",".*")==True\nassert is_match("mississippi","mis*is*p*.")==False\nassert is_match("aab","c*a*b")==True\nprint("__PASS__")'},
 {"id":"py-lru","lang":"python",
  "prompt":"用 Python 实现 `class LRUCache`,构造 `__init__(self, capacity: int)`,`get(self, key: int) -> int`(不存在返回 -1),`put(self, key: int, value: int)`,超容量淘汰最久未用。只返回该 class,```python 包裹。",
  "test":'c=LRUCache(2)\nc.put(1,1);c.put(2,2)\nassert c.get(1)==1\nc.put(3,3)\nassert c.get(2)==-1\nc.put(4,4)\nassert c.get(1)==-1\nassert c.get(3)==3\nassert c.get(4)==4\nprint("__PASS__")'},
 {"id":"py-longest-palindrome","lang":"python",
  "prompt":"用 Python 实现 `def longest_palindrome(s: str) -> str`,返回最长回文子串(有多个返回任意一个)。只返回该函数,```python 包裹。",
  "test":'r=longest_palindrome("babad")\nassert r in ("bab","aba"),r\nassert longest_palindrome("cbbd")=="bb"\nassert longest_palindrome("a")=="a"\nassert longest_palindrome("")==""\nprint("__PASS__")'},
 {"id":"js-curry","lang":"js",
  "prompt":"用 JavaScript 实现 `function curry(fn)`,返回 fn 的柯里化版本:可分次传参,凑满 fn.length 个参数后求值。只返回代码,```js 包裹。",
  "test":'function add(a,b,c){return a+b+c}\nconst c=curry(add);\nif(c(1)(2)(3)!==6)throw "a";\nif(c(1,2)(3)!==6)throw "b";\nif(c(1)(2,3)!==6)throw "c";\nif(c(1,2,3)!==6)throw "d";\nconsole.log("__PASS__")'},
]
def run_cli(prompt):
    t0=time.time()
    p=subprocess.run(["node",BIN,"-p",prompt,"--json"],env=ENV,capture_output=True,text=True,timeout=360)
    asst=[];reason=0
    for line in p.stdout.splitlines():
        line=line.strip()
        if not line:continue
        try:ev=json.loads(line)
        except Exception:continue
        if ev.get("type")=="assistant.delta":asst.append(ev.get("text",""))
        elif ev.get("type")=="reasoning.delta":reason+=len(ev.get("text",""))
    return "".join(asst),reason,round(time.time()-t0,1)
def extract(t):
    m=re.search(r"```(?:python|py|js|javascript)?\s*\n(.*?)```",t,re.S)
    return (m.group(1) if m else t).strip()
res=[]
for t in TASKS:
    try:resp,reason,secs=run_cli(t["prompt"])
    except subprocess.TimeoutExpired:res.append({"id":t["id"],"pass":False,"err":"timeout"});print(f"[{t['id']}] TIMEOUT",flush=True);continue
    code=extract(resp);src=code+"\n"+t["test"]
    runner=["python3","-c",src] if t["lang"]=="python" else ["node","-e",src]
    try:
        rp=subprocess.run(runner,capture_output=True,text=True,timeout=30)
        passed="__PASS__" in rp.stdout;err="" if passed else (rp.stderr or rp.stdout)[:200]
    except Exception as e:passed=False;err=str(e)[:200]
    res.append({"id":t["id"],"pass":passed,"secs":secs,"reason_chars":reason,"code":code[:450],"err":err})
    print(f"[{t['id']}] pass={passed} secs={secs} reason={reason}ch err={err[:90]}",flush=True)
pp=sum(1 for r in res if r.get("pass"))
print(f"\n=== round2 (harder) PASS@1 = {pp}/{len(res)} ===",flush=True)
pathlib.Path("/tmp/flash_coding_results2.json").write_text(json.dumps(res,ensure_ascii=False,indent=2))
