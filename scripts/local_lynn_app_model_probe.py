#!/usr/bin/env python3
"""Recommend local model install options for the Lynn desktop app.

The app should remain usable with a cloud/MIMO fallback even when no local model
is installed. This probe only decides which local model offers are reasonable to
show during first-run setup.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


GIB = 1024**3


def _darwin_mem_bytes() -> int | None:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(out)
    except Exception:
        return None


def _posix_mem_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except Exception:
        return None


def _memory_bytes() -> int | None:
    if platform.system() == "Darwin":
        return _darwin_mem_bytes()
    return _posix_mem_bytes()


def _round_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / GIB, 2)


def _disk_free_bytes(path: Path) -> int | None:
    probe_path = path
    while not probe_path.exists() and probe_path != probe_path.parent:
        probe_path = probe_path.parent
    try:
        return shutil.disk_usage(probe_path).free
    except Exception:
        return None


def _offer(
    *,
    artifact_id: str,
    model: str,
    runtime: str,
    priority: str,
    reason: str,
    download_gib: float,
    min_memory_gib: int,
    min_disk_gib: int,
    download_url_hint: str,
    smoke_required: list[str],
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "model": model,
        "runtime": runtime,
        "priority": priority,
        "reason": reason,
        "download_gib": download_gib,
        "download_url_hint": download_url_hint,
        "min_unified_memory_gib": min_memory_gib,
        "min_free_disk_gib": min_disk_gib,
        "smoke_required": smoke_required,
    }


def build_recommendation(model_root: Path) -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine()
    mem_bytes = _memory_bytes()
    free_bytes = _disk_free_bytes(model_root)
    mem_gib = _round_gib(mem_bytes)
    free_gib = _round_gib(free_bytes)

    is_macos_arm = system == "Darwin" and machine in {"arm64", "aarch64"}
    mem_for_offer = mem_gib if mem_gib is not None else 0.0
    disk_for_offer = free_gib if free_gib is not None else 0.0

    offers: list[dict[str, Any]] = []
    offer_excluded_reasons: list[dict[str, Any]] = []

    if is_macos_arm and mem_for_offer >= 8 and disk_for_offer >= 8:
        offers.append(
            _offer(
                artifact_id="qwen35-9b-q4km-imatrix-gguf",
                model="qwen35-9b-q4km",
                runtime="llama.cpp-metal",
                priority="recommended",
                reason="Apple Silicon with enough unified memory and disk for the stable 9B local-agent path.",
                download_gib=5.49,
                min_memory_gib=8,
                min_disk_gib=8,
                download_url_hint="https://dl.merkyorlynn.com/models/qwen35-9b/q4_k_m/",
                smoke_required=[
                    "download_manifest",
                    "sha256",
                    "llama_cpp_v1_models",
                    "chat_32_tokens",
                ],
            )
        )
    else:
        offer_excluded_reasons.append(
            {
                "artifact_id": "qwen35-9b-q4km-imatrix-gguf",
                "required": {
                    "is_macos_apple_silicon": True,
                    "unified_memory_gib": 8,
                    "free_disk_gib": 8,
                },
                "observed": {
                    "is_macos_apple_silicon": is_macos_arm,
                    "unified_memory_gib": mem_gib,
                    "free_disk_gib": free_gib,
                },
            }
        )

    if is_macos_arm and mem_for_offer >= 32 and disk_for_offer >= 30:
        offers.append(
            _offer(
                artifact_id="qwen36-35b-a3b-q4km-imatrix-gguf",
                model="qwen36-35b-a3b-q4km",
                runtime="llama.cpp-metal",
                priority="optional",
                reason="Large-memory Mac can try the 35B quality path, but it stays opt-in.",
                download_gib=20.0,
                min_memory_gib=32,
                min_disk_gib=30,
                download_url_hint="https://dl.merkyorlynn.com/models/qwen36-35b-a3b/q4_k_m/",
                smoke_required=[
                    "download_manifest",
                    "sha256",
                    "llama_cpp_v1_models",
                    "chat_32_tokens",
                    "short_structured_smoke",
                ],
            )
        )
    else:
        offer_excluded_reasons.append(
            {
                "artifact_id": "qwen36-35b-a3b-q4km-imatrix-gguf",
                "required": {
                    "is_macos_apple_silicon": True,
                    "unified_memory_gib": 32,
                    "free_disk_gib": 30,
                },
                "observed": {
                    "is_macos_apple_silicon": is_macos_arm,
                    "unified_memory_gib": mem_gib,
                    "free_disk_gib": free_gib,
                },
            }
        )

    local_first_allowed = bool(offers)
    return {
        "schema_version": "lynn-app-local-model-probe-v1",
        "platform": {
            "system": system,
            "machine": machine,
            "is_macos_apple_silicon": is_macos_arm,
            "unified_memory_gib": mem_gib,
            "model_root": str(model_root),
            "free_disk_gib": free_gib,
        },
        "default_provider": "mimo",
        "fallback_provider": "mimo",
        "local_first_allowed_after_smoke": local_first_allowed,
        "offers": offers,
        "offer_excluded_reasons": offer_excluded_reasons,
        "decision": (
            "offer_local_models_after_app_setup"
            if offers
            else "keep_mimo_first_no_local_offer"
        ),
        "notes": [
            "Never block first app launch on local model setup.",
            "Only switch local_provider.priority=first after all smoke_required checks pass.",
            "If any local runtime check fails, keep MIMO as the active provider.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        default=str(Path.home() / "Models" / "Lynn"),
        help="Directory where local model artifacts would be stored.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    args = parser.parse_args()

    payload = build_recommendation(Path(args.model_root).expanduser())
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
