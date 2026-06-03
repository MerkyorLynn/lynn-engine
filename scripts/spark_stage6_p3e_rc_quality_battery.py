#!/usr/bin/env python3
"""Stage 6 Phase 3-E: RC quality-battery smoke for the opt-in server path.

P3-D proves the OpenAI server surface and release/reload health counters. P3-E
keeps the same candidate server alive and runs a compact quality battery:

* MMLU sample through the existing OpenAI-compatible evaluator;
* GPQA Diamond sample through the existing OpenAI-compatible evaluator;
* structured JSON smoke;
* tool-call smoke;
* V8/V9-shaped prompt-format smoke;
* long-context needle smoke.

This banks only `banked_rc_quality_smoke=true`. It does not claim full
leaderboard quality or default promotion.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spark_stage6_p3d_server_rc_gate import _prefill_updates, _server_env  # noqa: E402
from scripts.spark_stage6_p2o_packed_prefill_rc_smoke import DEFAULT_MODEL, _degenerate, _norm  # noqa: E402


def _post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {text}") from exc


def _get_json(url: str, *, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {text}") from exc


def _start_server(
    *,
    model: str,
    served_name: str,
    port: int,
    max_seq_len: int,
    work_dir: Path,
) -> tuple[subprocess.Popen[bytes], Path]:
    log_path = work_dir / "candidate_server.log"
    log_fh = log_path.open("wb")
    cmd = [
        sys.executable,
        "-m",
        "server.openai_http",
        "--model",
        model,
        "--served-name",
        served_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-seq-len",
        str(max_seq_len),
    ]
    env = _server_env(True)
    env["LYNN_SKIP_RELOAD_IF_PACKED_PREFILL"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_fh.close()
    return proc, log_path


def _stop_server(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=20)


def _wait_health(base_url: str, proc: subprocess.Popen[bytes], *, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before ready with code {proc.returncode}")
        try:
            data = _get_json(f"{base_url}/health", timeout=5.0)
            if data.get("status") == "ok":
                return data
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
        time.sleep(2.0)
    raise TimeoutError(f"server not ready after {timeout_s}s; last_error={last_error}")


def _chat(base_url: str, model: str, messages: list[dict[str, str]], *, max_tokens: int, timeout: float, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    payload.update(extra)
    return _post_json(f"{base_url}/v1/chat/completions", payload, timeout=timeout)


def _content(resp: dict[str, Any]) -> str:
    choices = resp.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def _run_eval_cmd(cmd: list[str], *, cwd: Path, timeout: float) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _run_mmlu_gpqa(
    *,
    base_url: str,
    model: str,
    out_dir: Path,
    mmlu_data_dir: Path,
    gpqa_csv: Path,
    mmlu_sample: int,
    gpqa_sample: int,
    timeout: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mmlu_jsonl = out_dir / "mmlu_sample.jsonl"
    gpqa_jsonl = out_dir / "gpqa_sample.jsonl"
    out: dict[str, Any] = {
        "mmlu": {
            "available": mmlu_data_dir.is_dir(),
            "data_dir": str(mmlu_data_dir),
            "jsonl": str(mmlu_jsonl),
            "summary": str(mmlu_jsonl.with_suffix(".summary.json")),
        },
        "gpqa": {
            "available": gpqa_csv.is_file(),
            "csv": str(gpqa_csv),
            "jsonl": str(gpqa_jsonl),
            "summary": str(gpqa_jsonl.with_suffix(".summary.json")),
        },
    }
    if out["mmlu"]["available"]:
        out["mmlu"]["command"] = _run_eval_cmd(
            [
                sys.executable,
                "scripts/openai_mmlu_500_5shot_eval.py",
                "--data-dir",
                str(mmlu_data_dir),
                "--base-url",
                f"{base_url}/v1",
                "--model",
                model,
                "--out",
                str(mmlu_jsonl),
                "--concurrency",
                "1",
                "--shots",
                "5",
                "--sample",
                str(mmlu_sample),
                "--disable-thinking",
                "--append-no-think",
            ],
            cwd=ROOT,
            timeout=timeout,
        )
        out["mmlu"]["summary_data"] = _read_json(mmlu_jsonl.with_suffix(".summary.json"))
    if out["gpqa"]["available"]:
        out["gpqa"]["command"] = _run_eval_cmd(
            [
                sys.executable,
                "scripts/openai_gpqa_diamond_eval.py",
                "--csv",
                str(gpqa_csv),
                "--base-url",
                f"{base_url}/v1",
                "--model",
                model,
                "--out",
                str(gpqa_jsonl),
                "--concurrency",
                "1",
                "--sample",
                str(gpqa_sample),
                "--disable-thinking",
                "--append-no-think",
            ],
            cwd=ROOT,
            timeout=timeout,
        )
        out["gpqa"]["summary_data"] = _read_json(gpqa_jsonl.with_suffix(".summary.json"))
    return out


def _structured_smoke(base_url: str, model: str, *, timeout: float) -> dict[str, Any]:
    resp = _chat(
        base_url,
        model,
        [{"role": "user", "content": "Return one JSON object with keys city and unit for Tokyo in celsius. No markdown."}],
        max_tokens=96,
        timeout=timeout,
        response_format={"type": "json_object"},
    )
    text = _content(resp)
    parsed: Any = None
    error = None
    try:
        parsed = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    return {
        "text": text,
        "ok": isinstance(parsed, dict) and "city" in parsed and "unit" in parsed,
        "json_error": error,
        "usage": resp.get("usage") or {},
    }


def _tool_smoke(base_url: str, model: str, *, timeout: float) -> dict[str, Any]:
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    resp = _chat(
        base_url,
        model,
        [{"role": "user", "content": "Call get_weather for Tokyo using celsius."}],
        max_tokens=128,
        timeout=timeout,
        tools=tools,
        tool_choice="auto",
    )
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = str(message.get("content") or "")
    tool_calls = message.get("tool_calls") or []
    ok = bool(tool_calls) or "get_weather" in content
    return {
        "content": content,
        "tool_calls": tool_calls,
        "ok": ok,
        "finish_reason": choice.get("finish_reason"),
        "usage": resp.get("usage") or {},
    }


def _prompt_smoke(base_url: str, model: str, *, timeout: float) -> dict[str, Any]:
    specs = [
        {
            "id": "v8_format_smoke",
            "prompt": "Write exactly three bullet points about why linear attention helps long context. No markdown title.",
            "must": ["linear"],
            "min_chars": 80,
        },
        {
            "id": "v9_reasoning_smoke",
            "prompt": "A router sends each token to 8 of 256 experts. Explain in two concise sentences why top-k routing is useful.",
            "must": ["expert"],
            "min_chars": 80,
        },
    ]
    rows = []
    for spec in specs:
        resp = _chat(
            base_url,
            model,
            [{"role": "user", "content": spec["prompt"]}],
            max_tokens=192,
            timeout=timeout,
            chat_template_kwargs={"enable_thinking": False},
        )
        text = _content(resp)
        rows.append({
            "id": spec["id"],
            "text": text,
            "ok": (
                len(text.strip()) >= int(spec["min_chars"])
                and not _degenerate(text)
                and all(item in _norm(text) for item in spec["must"])
            ),
            "usage": resp.get("usage") or {},
        })
    return {"rows": rows, "ok": all(row["ok"] for row in rows)}


def _longctx_smoke(base_url: str, model: str, *, timeout: float, target_tokens: int) -> dict[str, Any]:
    filler = "The quick brown fox jumps over the lazy dog. " * max(80, target_tokens // 9)
    needle = "P3E-LONGCTX-NEEDLE-7749"
    prompt = (
        filler[: len(filler) // 2]
        + f"\n\nSecret needle: {needle}\n\n"
        + filler[len(filler) // 2 :]
        + "\n\nReturn only the secret needle string."
    )
    resp = _chat(
        base_url,
        model,
        [{"role": "user", "content": prompt}],
        max_tokens=64,
        timeout=timeout,
        chat_template_kwargs={"enable_thinking": False},
    )
    text = _content(resp)
    return {
        "target_tokens": target_tokens,
        "prompt_chars": len(prompt),
        "needle": needle,
        "text": text,
        "ok": needle in text,
        "usage": resp.get("usage") or {},
    }


def _summary_ok(summary: dict[str, Any] | None, *, min_n: int, min_accuracy: float, max_parse_fail_rate: float) -> bool:
    if not isinstance(summary, dict):
        return False
    n = int(summary.get("n") or 0)
    if n < min_n:
        return False
    if float(summary.get("accuracy") or 0.0) < min_accuracy:
        return False
    parse_fail = int(summary.get("parse_fail") or 0)
    if n and (parse_fail / n) > max_parse_fail_rate:
        return False
    if int(summary.get("errors") or 0) > 0:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--served-name", default="Lynn-P3E-candidate")
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--port", type=int, default=18371)
    ap.add_argument("--startup-timeout", type=float, default=900.0)
    ap.add_argument("--request-timeout", type=float, default=240.0)
    ap.add_argument("--eval-timeout", type=float, default=3600.0)
    ap.add_argument("--mmlu-data-dir", default="/home/merkyor/lynn-nemotron-eval/mmlu_csv")
    ap.add_argument("--gpqa-csv", default="/home/merkyor/quality-eval-20260517/datasets/gpqa/gpqa_diamond.csv")
    ap.add_argument("--mmlu-sample", type=int, default=100)
    ap.add_argument("--gpqa-sample", type=int, default=50)
    ap.add_argument("--mmlu-floor", type=float, default=0.70)
    ap.add_argument("--gpqa-floor", type=float, default=0.30)
    ap.add_argument("--max-parse-fail-rate", type=float, default=0.10)
    ap.add_argument("--longctx-target-tokens", type=int, default=8192)
    ap.add_argument("--work-dir", default="")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--p3d-pass", action="store_true", help="Assert P3-D predecessor passed.")
    args = ap.parse_args()

    work_dir = Path(args.work_dir or ".").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    proc: subprocess.Popen[bytes] | None = None
    base_url = f"http://127.0.0.1:{args.port}"
    started = time.time()
    error: str | None = None
    result: dict[str, Any]
    try:
        proc, server_log = _start_server(
            model=args.model,
            served_name=args.served_name,
            port=args.port,
            max_seq_len=args.max_seq_len,
            work_dir=work_dir,
        )
        health_before = _wait_health(base_url, proc, timeout_s=args.startup_timeout)
        models = _get_json(f"{base_url}/v1/models", timeout=args.request_timeout)
        structured = _structured_smoke(base_url, args.served_name, timeout=args.request_timeout)
        tool = _tool_smoke(base_url, args.served_name, timeout=args.request_timeout)
        prompt_smoke = _prompt_smoke(base_url, args.served_name, timeout=args.request_timeout)
        longctx = _longctx_smoke(
            base_url,
            args.served_name,
            timeout=args.eval_timeout,
            target_tokens=args.longctx_target_tokens,
        )
        mcq = _run_mmlu_gpqa(
            base_url=base_url,
            model=args.served_name,
            out_dir=work_dir,
            mmlu_data_dir=Path(args.mmlu_data_dir),
            gpqa_csv=Path(args.gpqa_csv),
            mmlu_sample=args.mmlu_sample,
            gpqa_sample=args.gpqa_sample,
            timeout=args.eval_timeout,
        )
        health_after = _get_json(f"{base_url}/health", timeout=args.request_timeout)
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
        health_before = {}
        health_after = {}
        models = {}
        structured = {"ok": False}
        tool = {"ok": False}
        prompt_smoke = {"ok": False}
        longctx = {"ok": False}
        mcq = {"mmlu": {"available": False}, "gpqa": {"available": False}}
        server_log = work_dir / "candidate_server.log"
    finally:
        _stop_server(proc)

    mmlu_summary = ((mcq.get("mmlu") or {}).get("summary_data") or None)
    gpqa_summary = ((mcq.get("gpqa") or {}).get("summary_data") or None)
    passes = {
        "p3d_pass": bool(args.p3d_pass),
        "server_ready": bool((health_before or {}).get("status") == "ok"),
        "models_surface": any(row.get("id") == args.served_name for row in (models.get("data") or [])),
        "release_enabled": bool((health_after or health_before).get("release_decode_shadows_after_prefill") is True),
        "skip_reload_enabled": bool((health_after or health_before).get("skip_reload_when_packed_prefill") is True),
        "zero_reload_observed": int((health_after or {}).get("release_reload_count") or 0) == 0,
        "structured_json": bool(structured.get("ok")),
        "tool_call": bool(tool.get("ok")),
        "v8_v9_prompt_smoke": bool(prompt_smoke.get("ok")),
        "longctx_needle": bool(longctx.get("ok")),
        "mmlu_available": bool((mcq.get("mmlu") or {}).get("available")),
        "gpqa_available": bool((mcq.get("gpqa") or {}).get("available")),
        "mmlu_score": _summary_ok(
            mmlu_summary,
            min_n=max(1, args.mmlu_sample),
            min_accuracy=args.mmlu_floor,
            max_parse_fail_rate=args.max_parse_fail_rate,
        ),
        "gpqa_score": _summary_ok(
            gpqa_summary,
            min_n=max(1, args.gpqa_sample),
            min_accuracy=args.gpqa_floor,
            max_parse_fail_rate=args.max_parse_fail_rate,
        ),
        "no_runner_error": error is None,
    }
    all_pass = all(passes.values())
    result = {
        "schema": "lynn-stage6-p3e-rc-quality-battery-v1",
        "verdict": "PASS" if all_pass else "FAIL",
        "banked_rc_quality_smoke": bool(all_pass),
        "banked_default_promotion": False,
        "banked_full_leaderboard_quality": False,
        "model": args.model,
        "served_name": args.served_name,
        "max_seq_len": args.max_seq_len,
        "base_url": base_url,
        "wall_seconds": round(time.time() - started, 3),
        "thresholds": {
            "mmlu_floor": args.mmlu_floor,
            "gpqa_floor": args.gpqa_floor,
            "max_parse_fail_rate": args.max_parse_fail_rate,
            "mmlu_sample": args.mmlu_sample,
            "gpqa_sample": args.gpqa_sample,
            "longctx_target_tokens": args.longctx_target_tokens,
        },
        "env": {"candidate": {**_prefill_updates(True), "LYNN_SKIP_RELOAD_IF_PACKED_PREFILL": "1"}},
        "health_before": health_before,
        "health_after": health_after,
        "models": models,
        "structured": structured,
        "tool": tool,
        "prompt_smoke": prompt_smoke,
        "longctx": longctx,
        "mcq": mcq,
        "passes": {**passes, "all": bool(all_pass)},
        "server_log": str(server_log),
        "error": error,
        "notes": [
            "P3-E banks only RC quality smoke for the opt-in P3 path.",
            "Default promotion remains false until a full release battery passes.",
            "V8/V9 here are repo-local prompt-format smoke checks, not the full external V8/V9 harness.",
            "MMLU/GPQA are samples with explicit floors; record sample size and floors with every artifact.",
        ],
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
