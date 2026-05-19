#!/usr/bin/env python3
"""P171: emit identity candidate outputs for the P169 linear-core contract.

This is a smoke helper for candidate-output-dir plumbing.  It reads P169
fixtures, mirrors the fixture reference tensors into a candidate output
directory, and optionally invokes P169 with --candidate-output-dir to prove the
emitted safetensors are accepted by the contract.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]

P169_CHECK_KEYS = [
    "proj_all",
    "out_conv",
    "conv_state_out",
    "core_attn_out",
    "recurrent_state_out",
    "gated_norm_out",
    "linear_core_out",
]
P169_FINAL_KEYS = [
    "linear_core_out",
    "recurrent_state_out",
    "conv_state_out",
]
P169_REFERENCE_KEY_ORDER = [
    "proj_all",
    "mixed_new",
    "z",
    "b",
    "a",
    "out_conv",
    "conv_state_out",
    "q_for_recurrent",
    "k_for_recurrent",
    "v_for_recurrent",
    "beta",
    "g",
    "core_attn_out",
    "recurrent_state_out",
    "gated_norm_out",
    "linear_core_out",
]
P169_INPUT_KEYS = {
    "h_norm",
    "recurrent_state_in",
    "conv_state_in",
}


@dataclass
class CandidateRow:
    fixture_file: str
    output_file: str
    layer_id: int
    prompt_id: int
    keys_written: list[str]
    tensor_shapes: dict[str, list[int]]
    tensor_dtypes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_manifest(fixtures_dir: Path) -> dict[str, Any]:
    manifest_path = fixtures_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"P169 fixture manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest.get("schema_version") or manifest.get("schema")
    if schema != "lynn-qwen36-linear-core-fixture-v1":
        raise ValueError(f"unexpected P169 fixture schema: {schema!r}")
    return manifest


def _requested_keys(args: argparse.Namespace, tensors: dict[str, Any]) -> list[str]:
    if args.only_final:
        keys = list(P169_FINAL_KEYS)
    elif args.keys:
        keys = [part.strip() for part in args.keys.split(",") if part.strip()]
    else:
        ordered = [key for key in P169_REFERENCE_KEY_ORDER if key in tensors]
        extras = sorted(k for k in tensors if k not in P169_INPUT_KEYS and k not in ordered)
        keys = ordered + extras

    missing = [key for key in keys if key not in tensors]
    if missing:
        raise KeyError(f"fixture is missing requested reference tensors: {missing}")
    return keys


def _emit_identity_candidates(args: argparse.Namespace) -> dict[str, Any]:
    fixtures_dir = Path(args.fixtures)
    out_dir = Path(args.candidate_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(fixtures_dir)
    rows: list[CandidateRow] = []
    t0 = time.time()

    for item in manifest["fixtures"]:
        fixture_file = str(item["file"])
        fixture_path = fixtures_dir / fixture_file
        tensors = load_file(str(fixture_path), device=args.device)
        keys = _requested_keys(args, tensors)
        candidate_tensors = {
            key: tensors[key].detach().contiguous().cpu()
            for key in keys
        }
        output_path = out_dir / fixture_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(candidate_tensors, str(output_path))
        rows.append(
            CandidateRow(
                fixture_file=fixture_file,
                output_file=fixture_file,
                layer_id=int(item["layer_id"]),
                prompt_id=int(item["prompt_id"]),
                keys_written=keys,
                tensor_shapes={key: list(candidate_tensors[key].shape) for key in keys},
                tensor_dtypes={key: str(candidate_tensors[key].dtype) for key in keys},
            )
        )

    summary = {
        "total_fixtures": len(rows),
        "total_tensors_written": sum(len(row.keys_written) for row in rows),
        "keys_mode": "only-final" if args.only_final else ("custom" if args.keys else "identity-reference"),
        "all_checked_keys_written": all(
            all(key in row.keys_written for key in P169_CHECK_KEYS)
            for row in rows
        ),
    }
    report = {
        "schema": "lynn-qwen36-linear-core-candidate-output-smoke-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fixtures": str(fixtures_dir),
        "fixture_manifest": str(fixtures_dir / "manifest.json"),
        "candidate_output_dir": str(out_dir),
        "device": args.device,
        "only_final": bool(args.only_final),
        "custom_keys": args.keys,
        "elapsed_seconds": time.time() - t0,
        "summary": summary,
        "p169_candidate_output_dir": str(out_dir),
        "results": [row.to_dict() for row in rows],
    }
    (out_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _p169_check_command(args: argparse.Namespace) -> list[str]:
    p169_script = Path(args.p169_script)
    if not p169_script.is_absolute():
        p169_script = ROOT / p169_script
    cmd = [
        sys.executable,
        str(p169_script),
        "--fixtures",
        str(Path(args.fixtures)),
        "--out",
        str(Path(args.p169_out)),
        "--check",
        "--candidate-output-dir",
        str(Path(args.candidate_output_dir)),
        "--device",
        args.device,
        "--max-abs-threshold",
        str(args.max_abs_threshold),
        "--cosine-threshold",
        str(args.cosine_threshold),
    ]
    if args.require_all_keys:
        cmd.append("--require-all-keys")
    return cmd


def _run_p169_check(args: argparse.Namespace) -> dict[str, Any]:
    p169_out = Path(args.p169_out)
    p169_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = _p169_check_command(args)
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    result: dict[str, Any] = {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "report": str(p169_out),
    }
    if p169_out.exists():
        try:
            p169_report = json.loads(p169_out.read_text(encoding="utf-8"))
            result["summary"] = p169_report.get("contract", {}).get("summary")
        except json.JSONDecodeError:
            result["summary"] = None
    if completed.returncode != 0:
        raise RuntimeError(f"P169 candidate-output-dir check failed with exit code {completed.returncode}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit P169 identity candidate outputs and smoke-check them.")
    parser.add_argument("--fixtures", required=True, help="P169 fixture directory containing manifest.json.")
    parser.add_argument(
        "--candidate-output-dir",
        "--out",
        dest="candidate_output_dir",
        required=True,
        help="Directory to write P169-compatible candidate safetensors.",
    )
    parser.add_argument("--report", default="", help="Optional copy of this helper's JSON report.")
    parser.add_argument("--only-final", action="store_true", help="Write only linear_core_out/recurrent_state_out/conv_state_out.")
    parser.add_argument("--keys", default="", help="Optional comma-separated tensor keys to mirror instead of the default set.")
    parser.add_argument("--device", default="cpu", help="Device used to load fixture tensors before writing CPU safetensors.")
    parser.add_argument("--skip-p169-check", action="store_true", help="Only write candidate outputs and print the P169 check command.")
    parser.add_argument("--p169-script", default="benchmarks/p169_qwen36_linear_core_fixture_contract.py")
    parser.add_argument("--p169-out", default="", help="P169 check report path. Defaults under candidate-output-dir.")
    parser.add_argument("--require-all-keys", action="store_true", help="Forward --require-all-keys to P169.")
    parser.add_argument("--max-abs-threshold", type=float, default=0.0)
    parser.add_argument("--cosine-threshold", type=float, default=0.999999)
    args = parser.parse_args()

    if args.only_final and args.keys:
        raise ValueError("--only-final and --keys are mutually exclusive")
    if args.only_final and args.require_all_keys:
        raise ValueError("--only-final cannot be combined with --require-all-keys")
    if not args.p169_out:
        args.p169_out = str(Path(args.candidate_output_dir) / "p169_candidate_output_smoke_check.json")

    report = _emit_identity_candidates(args)
    report["p169_check_command"] = _p169_check_command(args)
    if args.skip_p169_check:
        report["p169_check"] = {"skipped": True}
    else:
        report["p169_check"] = _run_p169_check(args)

    candidate_manifest = Path(args.candidate_output_dir) / "manifest.json"
    candidate_manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "candidate_output_dir": report["candidate_output_dir"],
        "summary": report["summary"],
        "p169_check_report": args.p169_out,
        "p169_check_summary": report.get("p169_check", {}).get("summary"),
        "report": args.report or str(candidate_manifest),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
