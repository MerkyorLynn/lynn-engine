#!/usr/bin/env python3
"""Stage 6 P4C tile=2 OpenAI server smoke gate.

This is a service-level gate for the opt-in active-reuse native MoE path. It
does not claim default promotion or full RC quality. The candidate server runs:

* ``LYNN_NATIVE_ACTIVE_MOE_BACKEND=fused_zero_shadow_active_reuse_contract``
* ``LYNN_NATIVE_GATEUP_TILE_INTER=2`` (or the supplied candidate tile)
* ``LYNN_RELEASE_DECODE_SHADOWS_AFTER_PREFILL=1``

The gate requires text parity against a Triton baseline and /health evidence
that the P4C native backend was called while decode shadows are released.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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
from scripts.spark_stage6_p3d_server_rc_gate import (  # noqa: E402
    _compare_rows,
    _get_json,
    _models_ok,
    _run_requests,
    _start_server,
    _stop_server,
    _wait_health,
)


SCHEMA = "lynn-stage6-p4c-tile2-server-smoke-v1"
EXPECTED_BACKEND = "fused_zero_shadow_active_reuse_contract"
COUNTER_NAME = "p4c_active_reuse_contract"
PASS_DECISION = "PASS_P4C_TILE2_SERVER_SMOKE"
FAIL_DECISION = "FAIL_P4C_TILE2_SERVER_SMOKE"
DEFAULT_NATIVE_ACTIVE_MOE_LAYERS = ",".join(str(i) for i in range(40))


def _p4c_updates(*, candidate: bool, gateup_tile_inter: int, native_active_moe_layers: str) -> dict[str, str]:
    return {
        "LYNN_MOE_IMPL": "packed_nvfp4",
        "LYNN_MOE_FAST_FIXED": "0",
        "LYNN_MOE_ACTIVE_SCRATCH": "1",
        "LYNN_NATIVE_ACTIVE_MOE_BACKEND": EXPECTED_BACKEND if candidate else "triton",
        "LYNN_NATIVE_ACTIVE_MOE_LAYERS": native_active_moe_layers,
        "LYNN_NATIVE_GATEUP_BACKEND": "triton_fast_decode",
        "LYNN_NATIVE_DOWN_BACKEND": "triton",
        "LYNN_NATIVE_GATEUP_TILE_INTER": str(gateup_tile_inter),
        "LYNN_NATIVE_FUSED_ZERO_SHADOW_TILE_TOKENS": "1",
        "LYNN_NATIVE_DOWN_TILE_HIDDEN": "8",
        "LYNN_RELEASE_DECODE_SHADOWS_AFTER_PREFILL": "1" if candidate else "0",
        "LYNN_SKIP_RELOAD_IF_PACKED_PREFILL": "0",
        "LYNN_PACKED_PREFILL_SLOW": "0",
    }


def _server_env(
    *,
    label: str,
    candidate: bool,
    gateup_tile_inter: int,
    native_active_moe_layers: str,
    work_dir: Path,
) -> dict[str, str]:
    env = dict(os.environ)
    env.update(CANONICAL_ENV)
    env.update(_p4c_updates(
        candidate=candidate,
        gateup_tile_inter=gateup_tile_inter,
        native_active_moe_layers=native_active_moe_layers,
    ))
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("LYNN_NATIVE_CUDA_BUILD_DIR", str(work_dir / f"native_build_{label}"))
    return env


def _counter(health: dict[str, Any]) -> dict[str, Any]:
    runtime = health.get("runtime") or {}
    counters = runtime.get("native_moe_counters") or {}
    return counters.get(COUNTER_NAME) or {}


def _counter_total(health: dict[str, Any]) -> int:
    return int((_counter(health).get("total_calls") or 0))


def _last_shapes(health: dict[str, Any]) -> dict[str, Any]:
    shapes = _counter(health).get("last_shapes") or {}
    return shapes if isinstance(shapes, dict) else {}


def _runtime_env_ok(health: dict[str, Any], *, gateup_tile_inter: int, native_active_moe_layers: str) -> bool:
    runtime = health.get("runtime") or {}
    return (
        runtime.get("native_active_moe_backend") == EXPECTED_BACKEND
        and runtime.get("native_active_moe_layers") == native_active_moe_layers
        and str(runtime.get("native_gateup_tile_inter")) == str(gateup_tile_inter)
    )


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
    gateup_tile_inter: int,
    native_active_moe_layers: str,
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
            env=_server_env(
                label=label,
                candidate=candidate,
                gateup_tile_inter=gateup_tile_inter,
                native_active_moe_layers=native_active_moe_layers,
                work_dir=work_dir,
            ),
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--preset", choices=("basic", "rc-mini"), default="basic")
    ap.add_argument("--prompt-limit", type=int, default=2)
    ap.add_argument("--chat-prompts", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--port", type=int, default=18371)
    ap.add_argument("--startup-timeout", type=float, default=900.0)
    ap.add_argument("--request-timeout", type=float, default=240.0)
    ap.add_argument("--min-release-gib", type=float, default=30.0)
    ap.add_argument("--min-p4c-call-delta", type=int, default=1)
    ap.add_argument("--gateup-tile-inter", type=int, default=2)
    ap.add_argument("--native-active-moe-layers", default=DEFAULT_NATIVE_ACTIVE_MOE_LAYERS)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--work-dir", default="")
    ap.add_argument("--prompts-json", default="")
    ap.add_argument("--p4c-runtime-pass", action="store_true", help="Assert P4C resident-runner predecessor passed.")
    args = ap.parse_args()

    if args.max_new <= 0:
        raise ValueError("--max-new must be positive")
    if args.prompt_limit <= 0:
        raise ValueError("--prompt-limit must be positive")
    if args.chat_prompts < 0:
        raise ValueError("--chat-prompts must be non-negative")
    if args.gateup_tile_inter <= 0:
        raise ValueError("--gateup-tile-inter must be positive")
    if not args.native_active_moe_layers.strip():
        raise ValueError("--native-active-moe-layers must be non-empty")

    prompts = _prompt_preset(args.preset)
    if args.prompts_json:
        prompts = json.loads(Path(args.prompts_json).read_text())
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompt list must be non-empty")
    prompts = [str(prompt) for prompt in prompts[: args.prompt_limit]]
    if any(not prompt.strip() for prompt in prompts):
        raise ValueError("all prompts must be non-empty strings")
    chat_prompts = prompts[: min(args.chat_prompts, len(prompts))]
    work_dir = Path(args.work_dir or ".").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    env_keys = set(CANONICAL_ENV) | set(_p4c_updates(
        candidate=True,
        gateup_tile_inter=args.gateup_tile_inter,
        native_active_moe_layers=args.native_active_moe_layers,
    ))
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        baseline = _serve_once(
            label="baseline",
            model=args.model,
            served_name="Lynn-P4C-baseline",
            port=args.port,
            max_seq_len=args.max_seq_len,
            prompts=prompts,
            chat_prompts=chat_prompts,
            max_new=args.max_new,
            timeout=args.request_timeout,
            startup_timeout=args.startup_timeout,
            candidate=False,
            gateup_tile_inter=args.gateup_tile_inter,
            native_active_moe_layers=args.native_active_moe_layers,
            work_dir=work_dir,
        )
        candidate = _serve_once(
            label="candidate",
            model=args.model,
            served_name="Lynn-P4C-tile2-candidate",
            port=args.port,
            max_seq_len=args.max_seq_len,
            prompts=prompts,
            chat_prompts=chat_prompts,
            max_new=args.max_new,
            timeout=args.request_timeout,
            startup_timeout=args.startup_timeout,
            candidate=True,
            gateup_tile_inter=args.gateup_tile_inter,
            native_active_moe_layers=args.native_active_moe_layers,
            work_dir=work_dir,
        )
    finally:
        _restore_env(old_env)

    completion_cmp = _compare_rows(baseline["completions"], candidate["completions"])
    chat_cmp = _compare_rows(baseline["chat"], candidate["chat"])
    all_baseline_rows = baseline["completions"] + baseline["chat"]
    all_candidate_rows = candidate["completions"] + candidate["chat"]
    all_cmp = completion_cmp + chat_cmp

    candidate_health_before = candidate.get("health_before") or {}
    candidate_health_after = candidate.get("health_after") or {}
    total_candidate_requests = len(all_candidate_rows)
    p4c_before = _counter_total(candidate_health_before)
    p4c_after = _counter_total(candidate_health_after)
    p4c_delta = p4c_after - p4c_before
    shapes = _last_shapes(candidate_health_after)
    release_reload_count = int(candidate_health_after.get("release_reload_count") or 0)
    last_release_gib = float(candidate_health_after.get("last_release_gib") or 0.0)
    last_reload_seconds = candidate_health_after.get("last_reload_seconds")

    prompt_count = (
        len(baseline["completions"]) == len(candidate["completions"]) == len(prompts)
        and len(baseline["chat"]) == len(candidate["chat"]) == len(chat_prompts)
        and len(all_cmp) == total_candidate_requests
    )
    functional_non_degenerate = all(not row["degenerate"] for row in all_baseline_rows + all_candidate_rows)
    text_exact = prompt_count and all(row["text_exact"] for row in all_cmp)
    server_surface = bool(baseline.get("models_ok") and candidate.get("models_ok"))
    runtime_env_ok = _runtime_env_ok(
        candidate_health_after,
        gateup_tile_inter=args.gateup_tile_inter,
        native_active_moe_layers=args.native_active_moe_layers,
    )
    p4c_called = p4c_delta >= args.min_p4c_call_delta
    tile_recorded = int(shapes.get("gateup_tile_inter") or -1) == args.gateup_tile_inter
    active_reuse_shapes = all(key in shapes for key in ("inter_scratch", "out", "expert_ids"))
    release_enabled = candidate_health_after.get("release_decode_shadows_after_prefill") is True
    release_consumed = candidate_health_after.get("release_decode_shadows_consumed") is True
    currently_released = candidate_health_after.get("decode_shadows_currently_released") is True
    release_meaningful = last_release_gib >= args.min_release_gib
    reload_expected = max(0, total_candidate_requests - 1)
    reload_observed = release_reload_count >= reload_expected and (
        total_candidate_requests <= 1 or last_reload_seconds is not None
    )

    passes = {
        "p4c_runtime_predecessor_pass": bool(args.p4c_runtime_pass),
        "server_surface": bool(server_surface),
        "prompt_count": bool(prompt_count),
        "functional_non_degenerate": bool(functional_non_degenerate),
        "server_text_exact": bool(text_exact),
        "candidate_runtime_env": bool(runtime_env_ok),
        "p4c_native_backend_called": bool(p4c_called),
        "p4c_tile_recorded": bool(tile_recorded),
        "p4c_active_reuse_shapes_recorded": bool(active_reuse_shapes),
        "release_enabled": bool(release_enabled),
        "release_consumed": bool(release_consumed),
        "decode_shadows_currently_released": bool(currently_released),
        "release_meaningful": bool(release_meaningful),
        "reload_observed": bool(reload_observed),
        "default_promotion_closed": True,
        "full_rc_quality_unbanked": True,
    }
    all_pass = all(passes.values())
    result = {
        "schema": SCHEMA,
        "decision": PASS_DECISION if all_pass else FAIL_DECISION,
        "verdict": "PASS" if all_pass else "FAIL",
        "banked_p4c_tile2_server_smoke": bool(all_pass),
        "banked_default_promotion": False,
        "banked_full_rc_quality": False,
        "model": args.model,
        "preset": args.preset,
        "prompt_limit": args.prompt_limit,
        "chat_prompts": len(chat_prompts),
        "max_new": args.max_new,
        "max_seq_len": args.max_seq_len,
        "gateup_tile_inter": args.gateup_tile_inter,
        "native_active_moe_layers": args.native_active_moe_layers,
        "min_release_gib": args.min_release_gib,
        "min_p4c_call_delta": args.min_p4c_call_delta,
        "env": {
            "baseline": _p4c_updates(
                candidate=False,
                gateup_tile_inter=args.gateup_tile_inter,
                native_active_moe_layers=args.native_active_moe_layers,
            ),
            "candidate": _p4c_updates(
                candidate=True,
                gateup_tile_inter=args.gateup_tile_inter,
                native_active_moe_layers=args.native_active_moe_layers,
            ),
        },
        "baseline": baseline,
        "candidate": candidate,
        "comparisons": {
            "completions": completion_cmp,
            "chat": chat_cmp,
        },
        "candidate_native_counter": {
            "counter_name": COUNTER_NAME,
            "before_total_calls": p4c_before,
            "after_total_calls": p4c_after,
            "delta_total_calls": p4c_delta,
            "after": _counter(candidate_health_after),
            "last_shapes": shapes,
        },
        "candidate_health": {
            "release_reload_count": release_reload_count,
            "reload_expected_min": reload_expected,
            "last_release_gib": last_release_gib,
            "last_reload_seconds": last_reload_seconds,
            "release_enabled": release_enabled,
            "release_consumed": release_consumed,
            "decode_shadows_currently_released": currently_released,
        },
        "passes": {**passes, "all": bool(all_pass)},
        "notes": [
            "P4C tile2 server smoke launches OpenAI-compatible baseline and candidate servers.",
            "A PASS banks only opt-in server evidence for fused_zero_shadow_active_reuse_contract.",
            "Default promotion remains false until full RC quality and sustained server speed gates pass.",
            "This gate checks native call counters and tile recording, not MMLU/GPQA/tool/long-context quality.",
        ],
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
