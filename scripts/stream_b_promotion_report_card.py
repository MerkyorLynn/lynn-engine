#!/usr/bin/env python3
"""Stream B promotion-decision report-card emitter.

Single source of truth for the 2026-05-18 hand-off promotion discipline:

* every candidate report must carry **P37 exact + hard structured + P25 512
  decode TPS** together;
* if any one of those three is missing, the candidate is a
  ``research_artifact_only`` — it cannot promote to DEFAULT or AMBER on
  microbench latency alone.

Reads the summary JSON emitted by
``scripts/r6000_qwen36_candidate_promotion_gate.sh`` (Codex Stream C
wrapper) and prints a deterministic markdown decision card to stdout.
The card structure is fixed so future readers and dashboards do not have
to re-derive the bar from scratch.

Promotion bar (2026-05-18 hand-off, fixed):

* DEFAULT promote: P37 3/3 exact AND structured 40/40 AND P25 512 ≥ 108 TPS
* AMBER promote: P37 may drift AND structured 70/70 AND P25 512 ≥ 118 TPS
                  (and the candidate ships opt-in, not as default route)
* sprint target Stream B: 118 TPS (AMBER)
* 122 TPS only after Stream A + Stream B candidates each pass their
  own DEFAULT gate, then are stacked through the full ladder.

Usage::

    scripts/stream_b_promotion_report_card.py \\
        --gate-json reports/promotion-gates/rope_cache_default.json \\
        --safe-default-tps 107 \\
        --out reports/promotion-gates/rope_cache_default.card.md

Exit codes:

* ``0`` — card emitted (decision may be DEFAULT_promote, AMBER_promote,
  AMBER_only, closed, or research_artifact_only).
* ``2`` — argument error (missing --gate-json or unparseable JSON).
* ``3`` — gate-json structure missing required Stream C wrapper fields
  (different schema than the wrapper emits).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_SAFE_TPS_FALLBACK = 107.0
DEFAULT_PROMOTE_DELTA_PCT = 1.0  # P25 512 ≥ safe + 1% (and ≥ 108 absolute)
AMBER_PROMOTE_DELTA_PCT = 5.0  # P25 512 ≥ safe + 5% (and ≥ 118 absolute)
DEFAULT_PROMOTE_TPS_FLOOR = 108.0
AMBER_PROMOTE_TPS_FLOOR = 118.0
STRUCTURED_DEFAULT_FRACTION = 1.0  # 40/40 exact
STRUCTURED_AMBER_FRACTION_HARD = 70.0 / 70.0  # placeholder; structured 70/70 = 1.0


def _get(d: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k)
        elif isinstance(cur, list):
            try:
                cur = cur[int(k)]
            except Exception:  # noqa: BLE001
                return default
        else:
            return default
    return cur if cur is not None else default


def _extract_gate_fields(report: dict[str, Any]) -> dict[str, Any]:
    """Pull the 3 mandatory fields from the Stream C wrapper schema.

    The Codex wrapper writes ``gates`` with:
      - ``p37_exact_pass``        (bool)
      - ``structured_pass_rate``  (float in [0,1] or None)
      - ``structured_pass``       (bool)
      - ``p25_512_mean_tps``      (float or None)
      - ``p25_delta_pct_vs_safe`` (float or None)

    Older runs may use slightly different keys; tolerate a couple of
    aliases so a Codex schema bump does not break this tool.
    """
    gates = report.get("gates") or {}
    out: dict[str, Any] = {}

    # P37
    p37 = gates.get("p37_exact_pass")
    if p37 is None:
        # tolerate alias paths
        exact_count = _get(report, "p37", "exact_count")
        if exact_count is not None:
            p37 = int(exact_count) >= 3
    out["p37_exact_pass"] = p37

    # Structured
    rate = gates.get("structured_pass_rate")
    if rate is None:
        passed = _get(report, "structured", "passed")
        total = _get(report, "structured", "total")
        if passed is not None and total:
            rate = float(passed) / float(total)
    out["structured_pass_rate"] = rate
    out["structured_pass"] = gates.get("structured_pass")
    if out["structured_pass"] is None and rate is not None:
        out["structured_pass"] = float(rate) >= STRUCTURED_DEFAULT_FRACTION

    # P25 @ 512
    tps_512 = gates.get("p25_512_mean_tps")
    if tps_512 is None:
        tps_512 = _get(report, "p25", "summary", "mean")
    out["p25_512_mean_tps"] = float(tps_512) if tps_512 is not None else None
    out["p25_delta_pct_vs_safe"] = gates.get("p25_delta_pct_vs_safe")

    return out


def _missing_required(fields: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if fields.get("p37_exact_pass") is None:
        missing.append("P37 exact-greedy")
    if fields.get("structured_pass_rate") is None and fields.get("structured_pass") is None:
        missing.append("hard structured pass")
    if fields.get("p25_512_mean_tps") is None:
        missing.append("P25 512-token decode TPS")
    return missing


def _decide(
    fields: dict[str, Any], safe_default_tps: float
) -> tuple[str, list[str]]:
    """Apply 2026-05-18 hand-off promotion bar; return (decision, reasons)."""
    p37 = bool(fields.get("p37_exact_pass"))
    rate = fields.get("structured_pass_rate")
    rate_f = float(rate) if rate is not None else None
    tps_512 = fields.get("p25_512_mean_tps")
    tps_f = float(tps_512) if tps_512 is not None else None
    delta_pct = fields.get("p25_delta_pct_vs_safe")
    if delta_pct is None and tps_f is not None and safe_default_tps > 0:
        delta_pct = (tps_f - safe_default_tps) / safe_default_tps * 100.0

    reasons: list[str] = []

    if tps_f is None:
        reasons.append("P25 512 decode TPS missing")
        return "research_artifact_only", reasons

    if rate_f is None:
        reasons.append("structured pass rate missing")
        return "research_artifact_only", reasons

    # Closed by hard regression
    if tps_f < safe_default_tps:
        reasons.append(
            f"P25 512 ({tps_f:.2f}) below safe default ({safe_default_tps:.2f}) — closed"
        )
        return "closed", reasons

    # DEFAULT?
    default_ok = (
        p37
        and rate_f >= STRUCTURED_DEFAULT_FRACTION
        and tps_f >= DEFAULT_PROMOTE_TPS_FLOOR
        and (delta_pct is not None and delta_pct >= DEFAULT_PROMOTE_DELTA_PCT)
    )

    # AMBER?
    amber_ok = (
        rate_f >= STRUCTURED_AMBER_FRACTION_HARD
        and tps_f >= AMBER_PROMOTE_TPS_FLOOR
        and (delta_pct is not None and delta_pct >= AMBER_PROMOTE_DELTA_PCT)
    )

    if default_ok:
        reasons.append(
            f"P37 3/3 + structured {rate_f * 100:.0f}% + P25 {tps_f:.2f} ≥ "
            f"{DEFAULT_PROMOTE_TPS_FLOOR} & ≥ safe + {DEFAULT_PROMOTE_DELTA_PCT}%"
        )
        if amber_ok:
            reasons.append("(also clears AMBER bar; ship as DEFAULT — stricter wins)")
        return "DEFAULT_promote", reasons

    if amber_ok:
        if not p37:
            reasons.append(
                f"P37 drift accepted; structured {rate_f * 100:.0f}% (need 70/70 hard) + "
                f"P25 {tps_f:.2f} ≥ {AMBER_PROMOTE_TPS_FLOOR}"
            )
        else:
            reasons.append(
                f"P37 3/3 but DEFAULT bar not met; falling to AMBER: structured "
                f"{rate_f * 100:.0f}% + P25 {tps_f:.2f} ≥ {AMBER_PROMOTE_TPS_FLOOR}"
            )
        return "AMBER_promote", reasons

    # Neither
    if not p37:
        reasons.append("P37 exact-greedy failed (DEFAULT requires 3/3)")
    if rate_f < STRUCTURED_DEFAULT_FRACTION:
        reasons.append(
            f"structured {rate_f * 100:.1f}% below DEFAULT bar 100%"
        )
    if tps_f < DEFAULT_PROMOTE_TPS_FLOOR:
        reasons.append(
            f"P25 512 ({tps_f:.2f}) below DEFAULT floor {DEFAULT_PROMOTE_TPS_FLOOR}"
        )
    if tps_f < AMBER_PROMOTE_TPS_FLOOR:
        reasons.append(
            f"P25 512 ({tps_f:.2f}) below AMBER floor {AMBER_PROMOTE_TPS_FLOOR}"
        )
    return "AMBER_only" if (rate_f >= STRUCTURED_DEFAULT_FRACTION and tps_f >= safe_default_tps) else "closed", reasons


def _render_card(
    report: dict[str, Any], fields: dict[str, Any], decision: str,
    reasons: list[str], missing: list[str], safe_default_tps: float,
) -> str:
    label = report.get("label", "(unnamed)")
    model = report.get("model", "(unknown)")
    candidate_env = report.get("candidate_env") or {}
    delta_pct = fields.get("p25_delta_pct_vs_safe")
    if delta_pct is None and fields.get("p25_512_mean_tps") is not None and safe_default_tps > 0:
        delta_pct = (fields["p25_512_mean_tps"] - safe_default_tps) / safe_default_tps * 100.0

    p37 = fields.get("p37_exact_pass")
    p37_str = "3/3 ✓" if p37 else ("漂 ✗" if p37 is False else "missing")
    rate = fields.get("structured_pass_rate")
    rate_str = f"{rate * 100:.1f}%" if rate is not None else "missing"
    tps = fields.get("p25_512_mean_tps")
    tps_str = f"{tps:.2f} TPS" if tps is not None else "missing"
    delta_str = f"{delta_pct:+.2f}%" if delta_pct is not None else "n/a"

    icon = {
        "DEFAULT_promote": "🟢 DEFAULT_promote",
        "AMBER_promote": "🟡 AMBER_promote",
        "AMBER_only": "🟡 AMBER_only (no DEFAULT path)",
        "closed": "🔴 closed",
        "research_artifact_only": "⚪ research_artifact_only",
    }.get(decision, decision)

    env_lines = "\n".join(f"- `{k}={v}`" for k, v in sorted(candidate_env.items())) or "- (delta from safe default empty)"

    reasons_block = "\n".join(f"- {r}" for r in reasons) if reasons else "- (no reasons recorded)"
    missing_block = (
        "\n".join(f"- {m}" for m in missing) if missing else "- (all three gates present)"
    )

    return f"""# Candidate: `{label}` — promotion decision

**Model**: `{model}`
**Safe-default reference**: `{safe_default_tps:.2f} TPS`

## Three required gates

| Gate | Result | DEFAULT bar | AMBER bar |
|---|---|---|---|
| P37 exact-greedy | **{p37_str}** | 3/3 required | drift OK |
| Hard structured | **{rate_str}** | 100% (40/40) | 100% on the hard 70-prompt set |
| P25 512 decode TPS | **{tps_str}** ({delta_str} vs safe) | ≥ {DEFAULT_PROMOTE_TPS_FLOOR} & ≥ safe + {DEFAULT_PROMOTE_DELTA_PCT}% | ≥ {AMBER_PROMOTE_TPS_FLOOR} & ≥ safe + {AMBER_PROMOTE_DELTA_PCT}% |

## Decision: {icon}

### Reasoning
{reasons_block}

### Required fields present
{missing_block}

## Candidate env delta

{env_lines}

## Discipline reminder

Per 2026-05-18 hand-off, no candidate is promoted on microbench latency
alone. Every promote-eligible candidate must carry **all three** gates:
P37 exact-greedy, hard structured pass, and P25 512-token decode TPS.

* DEFAULT route ships if and only if P37 3/3 + structured 40/40 + P25 512 ≥ {DEFAULT_PROMOTE_TPS_FLOOR}.
* AMBER ships opt-in only with structured 70/70 + P25 512 ≥ {AMBER_PROMOTE_TPS_FLOOR}.
* 122 TPS is reachable only after Stream A + Stream B candidates each
  pass DEFAULT in isolation, then are stacked through the full ladder.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-json", required=True, help="Stream C wrapper output JSON")
    ap.add_argument("--safe-default-tps", type=float, default=None,
                    help=f"override safe-default reference (else read from gate-json, "
                         f"else fallback {DEFAULT_SAFE_TPS_FALLBACK})")
    ap.add_argument("--out", default=None,
                    help="write markdown card here in addition to stdout")
    args = ap.parse_args()

    path = Path(args.gate_json)
    if not path.exists():
        print(f"[report-card] gate-json not found: {path}", file=sys.stderr)
        return 2
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[report-card] failed to parse {path}: {e}", file=sys.stderr)
        return 2

    safe_default_tps = args.safe_default_tps
    if safe_default_tps is None:
        safe_default_tps = report.get("safe_default_tps")
    if safe_default_tps is None:
        safe_default_tps = DEFAULT_SAFE_TPS_FALLBACK
    safe_default_tps = float(safe_default_tps)

    fields = _extract_gate_fields(report)
    missing = _missing_required(fields)
    if missing:
        decision = "research_artifact_only"
        reasons = [f"missing required gate: {m}" for m in missing] + [
            "no microbench-only number can earn DEFAULT or AMBER promote",
        ]
    else:
        decision, reasons = _decide(fields, safe_default_tps)

    card = _render_card(report, fields, decision, reasons, missing, safe_default_tps)
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(card, encoding="utf-8")
    print(card)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
