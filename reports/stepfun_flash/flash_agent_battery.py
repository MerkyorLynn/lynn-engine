#!/usr/bin/env python3
"""Lynn CLI `code` agent-mode harder battery on Step 3.7 Flash: real bug-fix +
multi-file implement, each with a sandbox + failing test the agent must make pass."""
import os, subprocess, json, pathlib, shutil, time
ENV={**os.environ,"LYNN_CLI_BASE_URL":"http://127.0.0.1:18099/v1","LYNN_CLI_MODEL":"step-3.7-flash","LYNN_CLI_API_KEY":"dummy","LYNN_CLI_PROVIDER":"openai-compatible"}
BIN="/Users/lynn/Downloads/Lynn/cli/bin/lynn.mjs"

# scenario files: {dir: {filename: content}}, task prompt, verify cmd (must print PASS / exit0 on success)
SCEN=[
 {"id":"bugfix-quicksort","dir":"/tmp/fab_bug",
  "files":{
    "quicksort.py":"def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr if x < pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)\n",
    "test_quicksort.py":"from quicksort import quicksort\nassert quicksort([3,1,2,3,1])==[1,1,2,3,3], quicksort([3,1,2,3,1])\nassert quicksort([5,5,5])==[5,5,5]\nassert quicksort([])==[]\nassert quicksort([2,1])==[1,2]\nprint('PASS')\n"},
  "prompt":"quicksort.py 有个 bug:当数组里有等于 pivot 的重复元素时会丢失它们(test_quicksort.py 会失败)。修复 quicksort.py 让 test_quicksort.py 通过,然后用 python3 运行 test_quicksort.py 确认通过。",
  "verify":"python3 test_quicksort.py"},
 {"id":"multifile-mathlib","dir":"/tmp/fab_feat",
  "files":{
    "mathlib.py":"# implement gcd(a,b), lcm(a,b), is_prime(n) here\n",
    "test_mathlib.py":"from mathlib import gcd, lcm, is_prime\nassert gcd(12,18)==6\nassert gcd(17,5)==1\nassert lcm(4,6)==12\nassert lcm(7,3)==21\nassert is_prime(2)==True\nassert is_prime(1)==False\nassert is_prime(97)==True\nassert is_prime(100)==False\nprint('PASS')\n"},
  "prompt":"在 mathlib.py 里实现 gcd(a,b)、lcm(a,b)、is_prime(n) 三个函数,让 test_mathlib.py 全部通过,然后用 python3 运行 test_mathlib.py 确认。",
  "verify":"python3 test_mathlib.py"},
]

def run_agent(d, prompt):
    t0=time.time()
    p=subprocess.run(["node",BIN,"code","-p",prompt,"--json","--cwd",d,"--approval","yolo","--sandbox","workspace-write"],
                     env=ENV,capture_output=True,text=True,timeout=420)
    tools=0; finished=False
    for line in p.stdout.splitlines():
        line=line.strip()
        if not line: continue
        try: ev=json.loads(line)
        except Exception: continue
        ty=ev.get("type","")
        if ty=="code.tool.result": tools+=1
        if ty=="code.task.finished": finished=bool(ev.get("ok"))
    return tools, finished, round(time.time()-t0,1)

res=[]
for s in SCEN:
    shutil.rmtree(s["dir"],ignore_errors=True); pathlib.Path(s["dir"]).mkdir(parents=True)
    for fn,ct in s["files"].items(): (pathlib.Path(s["dir"])/fn).write_text(ct)
    # pre-check: test should FAIL before
    pre=subprocess.run(s["verify"].split(),cwd=s["dir"],capture_output=True,text=True)
    pre_pass="PASS" in pre.stdout
    try:
        tools,finished,secs=run_agent(s["dir"],s["prompt"])
    except subprocess.TimeoutExpired:
        res.append({"id":s["id"],"pass":False,"err":"agent_timeout_420s"}); print(f"[{s['id']}] TIMEOUT",flush=True); continue
    post=subprocess.run(s["verify"].split(),cwd=s["dir"],capture_output=True,text=True,timeout=30)
    post_pass="PASS" in post.stdout
    res.append({"id":s["id"],"pre_pass":pre_pass,"agent_finished_ok":finished,"tool_calls":tools,"secs":secs,
                "pass":post_pass,"err":"" if post_pass else (post.stderr or post.stdout)[:200]})
    print(f"[{s['id']}] pre_pass={pre_pass} agent_ok={finished} tools={tools} secs={secs} FINAL_PASS={post_pass}",flush=True)
pp=sum(1 for r in res if r.get("pass"))
print(f"\n=== Flash agent-mode (bugfix+multifile) PASS = {pp}/{len(res)} ===",flush=True)
pathlib.Path("/tmp/flash_agent_battery_results.json").write_text(json.dumps(res,ensure_ascii=False,indent=2))
