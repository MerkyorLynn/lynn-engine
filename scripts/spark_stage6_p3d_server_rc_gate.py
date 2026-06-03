#!/usr/bin/env python3
"""Stage 6 Phase 3-D: OpenAI server smoke gate for the P3 zero-shadow path.

This is intentionally a service-level smoke, not a default-promotion claim. It
launches two local HTTP servers inside the Spark docker container:

* baseline: normal BF16 prefill path;
* candidate: packed prefill opt-in + release-after-prefill serving cycle.

The gate checks OpenAI endpoints, greedy text parity, non-degenerate responses,
and /health release/reload counters. Full MMLU/GPQA/long-context RC quality is a
separate publication gate.
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

from scripts.spark_stage6_p2o_packed_prefill_rc_smoke import (  # noqa: E402
    CANONICAL_ENV,
    DEFAULT_MODEL,
    _degenerate,
    _norm,
    _prompt_preset,
    _restore_env,
)


def _prefill_updates(enabled: bool) -> dict[str, str]:
    return {
        "LYNN_PACKED_PREFILL_SLOW": "1" if enabled else "0",
        "LYNN_PACKED_PREFILL_SLOW_MODE": "p3a_grouped",
        "LYNN_PACKED_PREFILL_P2E_LAYERS": "all",
        "LYNN_PACKED_PREFILL_P2E_BLOCK_T": "32",
        "LYNN_PACKED_PREFILL_P2E_BLOCK_INTER": "8",
        "LYNN_PACKED_PREFILL_P2E_BLOCK_HIDDEN": "128",
        "LYNN_PACKED_PREFILL_P2E_NUM_WARPS": "4",
        "LYNN_PACKED_PREFILL_P2E_DOWN_BLOCK_HIDDEN": "8",
        "LYNN_PACKED_PREFILL_P2E_DOWN_BLOCK_INTER": "512",
        "LYNN_PACKED_PREFILL_P2E_DOWN_NUM_WARPS": "8",
        "LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA": "1" if enabled else "0",
        "LYNN_RELEASE_DECODE_SHADOWS_AFTER_PREFILL": "1" if enabled else "0",
    }


def _server_env(candidate: bool) -> dict[str, str]:
    env = dict(os.environ)
    env.update(CANONICAL_ENV)
    env.update(_prefill_updates(candidate))
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def _post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
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
    label: str,
    model: str,
    served_name: str,
    port: int,
    max_seq_len: int,
    env: dict[str, str],
    work_dir: Path,
) -> tuple[subprocess.Popen[bytes], Path]:
    log_path = work_dir / f"{label}_server.log"
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
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    # Keep the descriptor owned by the child now.
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


def _models_ok(base_url: str, served_name: str, *, timeout: float) -> bool:
    data = _get_json(f"{base_url}/v1/models", timeout=timeout)
    return any(row.get("id") == served_name for row in data.get("data", []))


def _completion_text(resp: dict[str, Any]) -> str:
    choices = resp.get("choices") or []
    if not choices:
        return ""
    return str(choices[0].get("text", ""))


def _chat_text(resp: dict[str, Any]) -> str:
    choices = resp.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content", ""))


def _run_requests(
    *,
    base_url: str,
    served_name: str,
    prompts: list[str],
    chat_prompts: list[str],
    max_new: int,
    timeout: float,
) -> dict[str, Any]:
    completion_rows: list[dict[str, Any]] = []
    chat_rows: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        t0 = time.time()
        resp = _post_json(
            f"{base_url}/v1/completions",
            {
                "model": served_name,
                "prompt": prompt,
                "max_tokens": max_new,
                "temperature": 0,
            },
            timeout=timeout,
        )
        text = _completion_text(resp)
        completion_rows.append({
            "index": idx,
            "prompt": prompt,
            "text": text,
            "normalized_text": _norm(text),
            "degenerate": _degenerate(text),
            "finish_reason": (resp.get("choices") or [{}])[0].get("finish_reason"),
            "usage": resp.get("usage") or {},
            "metrics": resp.get("_lynn_engine_metrics") or {},
            "elapsed_seconds": time.time() - t0,
        })

    for idx, prompt in enumerate(chat_prompts):
        t0 = time.time()
        resp = _post_json(
            f"{base_url}/v1/chat/completions",
            {
                "model": served_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new,
                "temperature": 0,
            },
            timeout=timeout,
        )
        text = _chat_text(resp)
        chat_rows.append({
            "index": idx,
            "prompt": prompt,
            "text": text,
            "normalized_text": _norm(text),
            "degenerate": _degenerate(text),
            "finish_reason": (resp.get("choices") or [{}])[0].get("finish_reason"),
            "usage": resp.get("usage") or {},
            "metrics": resp.get("_lynn_engine_metrics") or {},
            "elapsed_seconds": time.time() - t0,
        })
    return {"completions": completion_rows, "chat": chat_rows}


def _serve_once(
    *,
    label: str,
    model: str,
    served_name: str,
    port: int,
    max_seq_len: int,
    prompts: list[str],
    chat_prompts: list[str],
    max_new: int,
    timeout: float,
    startup_timeout: float,
    candidate: bool,
    work_dir: Path,
) -> dict[str, Any]:
    proc: subprocess.Popen[bytes] | None = None
    log_path: Path | None = None
    base_url = f"http://127.0.0.1:{port}"
    started = time.time()
    try:
        proc, log_path = _start_server(
            label=label,
            model=model,
            served_name=served_name,
            port=port,
            max_seq_len=max_seq_len,
            env=_server_env(candidate),
            work_dir=work_dir,
        )
        health_before = _wait_health(base_url, proc, timeout_s=startup_timeout)
        models_ok = _models_ok(base_url, served_name, timeout=timeout)
        rows = _run_requests(
            base_url=base_url,
            served_name=served_name,
            prompts=prompts,
            chat_prompts=chat_prompts,
            max_new=max_new,
            timeout=timeout,
        )
        health_after = _get_json(f"{base_url}/health", timeout=timeout)
        return {
            "label": label,
            "served_name": served_name,
            "port": port,
            "candidate": candidate,
            "server_log": str(log_path) if log_path else "",
            "startup_seconds": health_before.get("elapsed_s"),
            "wall_seconds": time.time() - started,
            "models_ok": models_ok,
            "health_before": health_before,
            "health_after": health_after,
            **rows,
        }
    finally:
        _stop_server(proc)


def _compare_rows(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base, cand in zip(baseline, candidate):
        rows.append({
            "index": base.get("index"),
            "text_exact": base.get("normalized_text") == cand.get("normalized_text"),
            "baseline_text": base.get("text", ""),
            "candidate_text": cand.get("text", ""),
            "baseline_finish_reason": base.get("finish_reason"),
            "candidate_finish_reason": cand.get("finish_reason"),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--preset", choices=("basic", "rc-mini"), default="basic")
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--port", type=int, default=18361)
    ap.add_argument("--startup-timeout", type=float, default=900.0)
    ap.add_argument("--request-timeout", type=float, default=180.0)
    ap.add_argument("--min-release-gib", type=float, default=30.0)
    ap.add_argument("--chat-prompts", type=int, default=2)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--work-dir", default="")
    ap.add_argument("--prompts-json", default="")
    ap.add_argument("--p3c-pass", action="store_true", help="Assert P3-C predecessor passed.")
    args = ap.parse_args()

    if args.max_new <= 0:
        raise ValueError("--max-new must be positive")
    if args.chat_prompts < 0:
        raise ValueError("--chat-prompts must be non-negative")

    prompts = _prompt_preset(args.preset)
    if args.prompts_json:
        prompts = json.loads(Path(args.prompts_json).read_text())
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompt list must be non-empty")
    prompts = [str(prompt) for prompt in prompts]
    if any(not prompt.strip() for prompt in prompts):
        raise ValueError("all prompts must be non-empty strings")
    chat_prompts = prompts[: min(args.chat_prompts, len(prompts))]
    work_dir = Path(args.work_dir or ".").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    old_env = {key: os.environ.get(key) for key in set(CANONICAL_ENV) | set(_prefill_updates(True))}
    try:
        baseline = _serve_once(
            label="baseline",
            model=args.model,
            served_name="Lynn-P3D-baseline",
            port=args.port,
            max_seq_len=args.max_seq_len,
            prompts=prompts,
            chat_prompts=chat_prompts,
            max_new=args.max_new,
            timeout=args.request_timeout,
            startup_timeout=args.startup_timeout,
            candidate=False,
            work_dir=work_dir,
        )
        candidate = _serve_once(
            label="candidate",
            model=args.model,
            served_name="Lynn-P3D-candidate",
            port=args.port,
            max_seq_len=args.max_seq_len,
            prompts=prompts,
            chat_prompts=chat_prompts,
            max_new=args.max_new,
            timeout=args.request_timeout,
            startup_timeout=args.startup_timeout,
            candidate=True,
            work_dir=work_dir,
        )
    finally:
        _restore_env(old_env)

    completion_cmp = _compare_rows(baseline["completions"], candidate["completions"])
    chat_cmp = _compare_rows(baseline["chat"], candidate["chat"])
    all_baseline_rows = baseline["completions"] + baseline["chat"]
    all_candidate_rows = candidate["completions"] + candidate["chat"]
    all_cmp = completion_cmp + chat_cmp

    health = candidate.get("health_after") or {}
    total_candidate_requests = len(all_candidate_rows)
    reload_count = int(health.get("release_reload_count") or 0)
    last_release_gib = float(health.get("last_release_gib") or 0.0)
    last_reload_seconds = health.get("last_reload_seconds")
    release_enabled = health.get("release_decode_shadows_after_prefill") is True
    release_consumed = health.get("release_decode_shadows_consumed") is True
    currently_released = health.get("decode_shadows_currently_released") is True

    prompt_count = (
        len(baseline["completions"]) == len(candidate["completions"]) == len(prompts)
        and len(baseline["chat"]) == len(candidate["chat"]) == len(chat_prompts)
        and len(all_cmp) == total_candidate_requests
    )
    functional_non_degenerate = all(not row["degenerate"] for row in all_baseline_rows + all_candidate_rows)
    text_exact = prompt_count and all(row["text_exact"] for row in all_cmp)
    server_surface = bool(baseline.get("models_ok") and candidate.get("models_ok"))
    release_meaningful = last_release_gib >= args.min_release_gib
    reload_expected = max(0, total_candidate_requests - 1)
    reload_observed = reload_count >= reload_expected and (
        total_candidate_requests <= 1 or last_reload_seconds is not None
    )

    passes = {
        "p3c_pass": bool(args.p3c_pass),
        "server_surface": bool(server_surface),
        "prompt_count": bool(prompt_count),
        "functional_non_degenerate": bool(functional_non_degenerate),
        "server_text_exact": bool(text_exact),
        "release_enabled": bool(release_enabled),
        "release_consumed": bool(release_consumed),
        "decode_shadows_currently_released": bool(currently_released),
        "release_meaningful": bool(release_meaningful),
        "reload_observed": bool(reload_observed),
    }
    all_pass = all(passes.values())
    result = {
        "schema": "lynn-stage6-p3d-server-rc-gate-v1",
        "verdict": "PASS" if all_pass else "FAIL",
        "banked_server_smoke": bool(all_pass),
        "banked_default_promotion": False,
        "banked_full_rc_quality": False,
        "model": args.model,
        "preset": args.preset,
        "max_new": args.max_new,
        "max_seq_len": args.max_seq_len,
        "min_release_gib": args.min_release_gib,
        "env": {
            "baseline": _prefill_updates(False),
            "candidate": _prefill_updates(True),
        },
        "baseline": baseline,
        "candidate": candidate,
        "comparisons": {
            "completions": completion_cmp,
            "chat": chat_cmp,
        },
        "candidate_health": {
            "release_reload_count": reload_count,
            "reload_expected_min": reload_expected,
            "last_release_gib": last_release_gib,
            "last_reload_seconds": last_reload_seconds,
            "release_enabled": release_enabled,
            "release_consumed": release_consumed,
            "decode_shadows_currently_released": currently_released,
        },
        "passes": {**passes, "all": bool(all_pass)},
        "notes": [
            "P3-D launches OpenAI-compatible baseline and candidate servers.",
            "A PASS banks only server smoke for the opt-in zero-shadow prefill path.",
            "Default promotion remains false until full RC quality batteries pass.",
            "Full MMLU/GPQA/tool/long-context release gates are not run here.",
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
