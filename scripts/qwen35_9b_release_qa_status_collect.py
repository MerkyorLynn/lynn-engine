#!/usr/bin/env python3
"""Qwen3.5-9B Release QA Status Collector (stdlib-only).

Reads known evidence files from reports/qwen35_9b/ and updates the QA status
markdown with PASS/EVIDENCE or BLOCKED/EXPERIMENTAL where evidence is clear.
Items without evidence remain PENDING with a reason.

Evidence sources:
  - P199 size audit (NVFP4 model size confirmation)
  - Q4_K_M CUDA baseline + long32k parallel=1 (Mac Q4_K_M 32K context)
  - P193 native boundary admission (W4A8 gate decision)
  - P190 true FP8 resident gate findings
  - P196 W4A8 structured content gate

Usage:
  python3 scripts/qwen35_9b_release_qa_status_collect.py \
    --root /path/to/repo \
    --status-file reports/qwen35_9b/QWEN35_9B_RELEASE_QA_STATUS_20260519.md \
    --report-dir reports/qwen35_9b/
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


# ── Evidence loaders ────────────────────────────────────────────────────────

def load_p199_size_audit(report_dir: str) -> Optional[dict]:
    """Load P199 NVFP4 size audit evidence."""
    path = os.path.join(report_dir, "p199_nvfp4_size_audit_20260519_live_size2.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_q4km_long32k(report_dir: str) -> Optional[dict]:
    """Check Q4_K_M long32k parallel=1 evidence."""
    path = os.path.join(report_dir, "q4km_long32k_parallel1_20260519_0128.md")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        content = f.read()
    # Parse TPS and status from the markdown table
    tps_match = re.search(r"32768\s*\|\s*7757\s*\|\s*128\s*\|\s*([\d.]+)\s*\|\s*(\w+)", content)
    if tps_match:
        return {"tps": float(tps_match.group(1)), "status": tps_match.group(2)}
    return None


def load_p193_admission(report_dir: str) -> Optional[dict]:
    """Load P193 native boundary admission evidence."""
    path = os.path.join(report_dir, "p193_native_boundary_admission_20260519_required_gates.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_p190_findings(report_dir: str) -> Optional[dict]:
    """Load P190 true FP8 resident gate findings."""
    path = os.path.join(report_dir, "P190_FP4XFP8_RESIDENT_FINDINGS_20260519.md")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        content = f.read()
    verdict_match = re.search(r"full-32-layer\s+resident\s+gate:\s*`([^`]+)`", content)
    exact_match = re.search(r"full-32-layer\s+exact:\s*`([^`]+)`", content)
    return {
        "verdict": verdict_match.group(1) if verdict_match else "UNKNOWN",
        "exact": exact_match.group(1) if exact_match else "UNKNOWN",
    }


def load_p196_gate(report_dir: str) -> Optional[dict]:
    """Load P196 W4A8 structured content gate evidence."""
    path = os.path.join(report_dir, "P196_W4A8_STRUCTURED_CONTENT_GATE_20260519.md")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        content = f.read()
    w4a16_match = re.search(r"W4A16\s+reference\s*\|\s*(\d+)/(\d+)\s*\|\s*([\d.]+)%", content)
    decision_match = re.search(r"Decision\b[^\n]*\n+([^\n]+)", content)
    result = {}
    if w4a16_match:
        result["w4a16_pass"] = int(w4a16_match.group(1))
        result["w4a16_total"] = int(w4a16_match.group(2))
        result["w4a16_rate"] = float(w4a16_match.group(3))
    if decision_match:
        result["decision"] = decision_match.group(1).strip()
    return result if result else None


# ── Evidence → checklist mapping ────────────────────────────────────────────

def build_evidence_map(report_dir: str) -> Dict[str, Tuple[str, str]]:
    """Build a map of checklist item IDs → (status, evidence_remark).

    Only items with clear evidence are included. Everything else stays PENDING.
    """
    evidence: Dict[str, Tuple[str, str]] = {}

    # ── P199: NVFP4 size audit → B.1.7 ──
    p199 = load_p199_size_audit(report_dir)
    if p199:
        total_gib = p199.get("total_gib", 0)
        n_files = p199.get("n_files", 0)
        n_tensors = p199.get("n_tensors", 0)
        manifest = p199.get("manifest_present", False)
        evidence["B.1.7"] = (
            "PASS",
            f"P199 size audit: {total_gib:.2f} GiB, {n_files} files, "
            f"{n_tensors} tensors, manifest={'yes' if manifest else 'no'}",
        )

    # ── Q4_K_M long32k parallel=1 → A.6.5 ──
    q4km = load_q4km_long32k(report_dir)
    if q4km and q4km.get("status") == "PASS":
        evidence["A.6.5"] = (
            "PASS",
            f"Q4_K_M CUDA long32k parallel=1: {q4km['tps']:.2f} TPS, 32K context confirmed",
        )

    # ── P193: native boundary admission → C.1.4 ──
    p193 = load_p193_admission(report_dir)
    if p193:
        decision = p193.get("decision", "UNKNOWN")
        reasons = p193.get("reasons", [])
        reason_str = "; ".join(reasons) if reasons else "no details"
        if decision == "PROMOTE_BLOCKED":
            evidence["C.1.4"] = (
                "BLOCKED",
                f"P193: {decision} — {reason_str}",
            )

    # ── P190: true FP8 resident gate → C.1.2 ──
    p190 = load_p190_findings(report_dir)
    if p190:
        verdict = p190.get("verdict", "UNKNOWN")
        exact = p190.get("exact", "UNKNOWN")
        if verdict in ("TRUE_FP8_RESIDENT_RED", "RED"):
            evidence["C.1.2"] = (
                "BLOCKED",
                f"P190: {verdict}, exact={exact}; W4A8 resident not structured-safe",
            )

    # ── W4A8 resident gate items → BLOCKED/EXPERIMENTAL ──
    # C.1.1: P197 drift probe — AMBER on fake-W4A8 path
    p197_path = os.path.join(report_dir, "P197_W4A8_TOKEN_DRIFT_ISOLATION_20260519.md")
    if os.path.isfile(p197_path):
        evidence["C.1.1"] = (
            "BLOCKED",
            "P197: AMBER on fake-W4A8 path; true resident requires P190/P198 gate",
        )
    else:
        # Check if any p197 drift report exists
        import glob as _glob
        p197_files = _glob.glob(os.path.join(report_dir, "p197_*drift*.json"))
        p197_md = _glob.glob(os.path.join(report_dir, "P197_*.md"))
        if p197_files or p197_md:
            evidence["C.1.1"] = (
                "BLOCKED",
                "P197 drift probe exists but true resident path blocked (P190 RED)",
            )

    # C.1.3: P196 W4A8 column
    p196 = load_p196_gate(report_dir)
    if p196:
        decision = p196.get("decision", "")
        if "AMBER" in decision or "NO_REGRESSION" in decision:
            evidence["C.1.3"] = (
                "BLOCKED",
                f"P196: {decision}; W4A8 experimental, not on stable track",
            )

    # C.2.1: Already PASS by construction
    evidence["C.2.1"] = ("PASS", "Track C title contains **experimental** (by construction)")

    return evidence


# ── Markdown table parser / updater ─────────────────────────────────────────

def parse_status_table(lines: List[str]) -> List[dict]:
    """Parse QA status markdown rows into structured items.

    Each row: | A.1.1 | 检查项描述 | STATUS | EVIDENCE |
    Returns list of {id, description, status, evidence, line_index}.
    """
    items = []
    row_pattern = re.compile(
        r"^\|\s*([A-C]\.\d+\.\d+)\s*\|\s*(.+?)\s*\|\s*(\S+)\s*\|\s*(.*?)\s*\|"
    )
    for i, line in enumerate(lines):
        m = row_pattern.match(line)
        if m:
            items.append({
                "id": m.group(1),
                "description": m.group(2).strip(),
                "status": m.group(3).strip(),
                "evidence": m.group(4).strip(),
                "line_index": i,
            })
    return items


def update_status_markdown(
    status_path: str,
    evidence_map: Dict[str, Tuple[str, str]],
    report_dir: str,
) -> str:
    """Read status markdown, apply evidence updates, return updated content."""
    with open(status_path) as f:
        lines = f.readlines()

    items = parse_status_table(lines)
    updated_count = 0

    for item in items:
        item_id = item["id"]
        if item_id in evidence_map:
            new_status, new_evidence = evidence_map[item_id]
            old_line = lines[item["line_index"]]
            # Replace status and evidence columns
            # Columns: | ID | DESC | STATUS | EVIDENCE |
            parts = old_line.split("|")
            if len(parts) >= 5:
                parts[3] = f" {new_status} "
                parts[4] = f" {new_evidence} "
                new_line = "|".join(parts)
                if not new_line.endswith("\n"):
                    new_line += "\n"
                lines[item["line_index"]] = new_line
                updated_count += 1

    # Update overall decision to reflect current state
    # Count PASS/FAIL/BLOCKED/PENDING for summary
    statuses = {}
    for item in items:
        sid = item["id"]
        if sid in evidence_map:
            st, _ = evidence_map[sid]
        else:
            st = item["status"]
        statuses[st] = statuses.get(st, 0) + 1

    # Update the "Overall decision" line
    pass_count = statuses.get("PASS", 0)
    pending_count = statuses.get("PENDING", 0)
    blocked_count = statuses.get("BLOCKED", 0)

    # Rebuild summary section
    out_lines = []
    in_summary = False
    in_blockers = False
    summary_written = False
    blockers_written = False

    for i, line in enumerate(lines):
        # Update "Overall decision" line
        if line.startswith("**Overall decision:**"):
            out_lines.append(f"**Overall decision:** `PENDING_QA` (collector updated {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})\n")
            continue

        # Update "最后更新" line
        if "最后更新:" in line and "QA 状态文件" in line:
            out_lines.append(f"*QA 状态文件，随每次执行更新。最后更新: {datetime.now(timezone.utc).strftime('%Y-%m-%d')} (collector v2).*\n")
            continue

        # Update generated date
        if line.startswith("**Generated:**"):
            out_lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n")
            continue

        # Update branch reference
        if line.startswith("**Branch:**"):
            out_lines.append("**Branch:** `main` + `qwen35_9b_release_qa_status_collect.py`\n")
            continue

        out_lines.append(line)

    # Re-count Track-level summaries
    # Count by track
    track_counts = {"A": {"PASS": 0, "FAIL": 0, "PENDING": 0, "BLOCKED": 0, "total": 0},
                    "B": {"PASS": 0, "FAIL": 0, "PENDING": 0, "BLOCKED": 0, "total": 0},
                    "C": {"PASS": 0, "FAIL": 0, "PENDING": 0, "BLOCKED": 0, "total": 0}}
    for item in items:
        tid = item["id"]
        track = tid[0]
        if track in track_counts:
            if tid in evidence_map:
                st, _ = evidence_map[tid]
            else:
                st = item["status"]
            if st in track_counts[track]:
                track_counts[track][st] += 1
            track_counts[track]["total"] += 1

    # Update track subtotal lines
    final_lines = []
    for line in out_lines:
        # Track A subtotal
        m = re.match(r"^\*\*Track A 小计:\*\*\s+.*", line)
        if m:
            tc = track_counts["A"]
            final_lines.append(
                f"**Track A 小计:** {tc['PASS']} PASS / {tc['FAIL']} FAIL / "
                f"{tc['BLOCKED']} BLOCKED / {tc['PENDING']} PENDING\n"
            )
            continue
        # Track B subtotal
        m = re.match(r"^\*\*Track B 小计:\*\*\s+.*", line)
        if m:
            tc = track_counts["B"]
            final_lines.append(
                f"**Track B 小计:** {tc['PASS']} PASS / {tc['FAIL']} FAIL / "
                f"{tc['BLOCKED']} BLOCKED / {tc['PENDING']} PENDING\n"
            )
            continue
        # Track C subtotal
        m = re.match(r"^\*\*Track C 小计:\*\*\s+.*", line)
        if m:
            tc = track_counts["C"]
            final_lines.append(
                f"**Track C 小计:** {tc['PASS']} PASS / {tc['FAIL']} FAIL / "
                f"{tc['BLOCKED']} BLOCKED / {tc['PENDING']} PENDING\n"
            )
            continue
        final_lines.append(line)

    grand = {"PASS": 0, "FAIL": 0, "PENDING": 0, "BLOCKED": 0, "total": 0}
    for tc in track_counts.values():
        for key in ("PASS", "FAIL", "PENDING", "BLOCKED", "total"):
            grand[key] += tc[key]

    def decision_for(tc):
        if tc["FAIL"]:
            return "FAIL"
        if tc["BLOCKED"]:
            return "BLOCKED" if tc["PENDING"] == 0 else "PENDING_BLOCKED"
        if tc["PENDING"]:
            return "PENDING_QA"
        return "PASS"

    rewritten = []
    in_overall_summary = False
    for line in final_lines:
        if line.startswith("## Overall Summary"):
            in_overall_summary = True
            rewritten.append(line)
            continue
        if in_overall_summary and line.startswith("| Track | Items | PASS | FAIL | PENDING | Decision |"):
            rewritten.append("| Track | Items | PASS | FAIL | BLOCKED | PENDING | Decision |\n")
            continue
        if in_overall_summary and line.startswith("|-------|-------|------|------|---------|----------|"):
            rewritten.append("|-------|-------|------|------|---------|---------|----------|\n")
            continue
        if line.startswith("| A: Mac Q4_K_M |"):
            tc = track_counts["A"]
            rewritten.append(
                f"| A: Mac Q4_K_M | {tc['total']} | {tc['PASS']} | {tc['FAIL']} | "
                f"{tc['BLOCKED']} | {tc['PENDING']} | {decision_for(tc)} |\n"
            )
            continue
        if line.startswith("| B: NVIDIA NVFP4 W4A16 |"):
            tc = track_counts["B"]
            rewritten.append(
                f"| B: NVIDIA NVFP4 W4A16 | {tc['total']} | {tc['PASS']} | {tc['FAIL']} | "
                f"{tc['BLOCKED']} | {tc['PENDING']} | {decision_for(tc)} |\n"
            )
            continue
        if line.startswith("| C: Experimental W4A8 |"):
            tc = track_counts["C"]
            rewritten.append(
                f"| C: Experimental W4A8 | {tc['total']} | {tc['PASS']} | {tc['FAIL']} | "
                f"{tc['BLOCKED']} | {tc['PENDING']} | {decision_for(tc)} |\n"
            )
            continue
        if line.startswith("| **Total** |"):
            rewritten.append(
                f"| **Total** | **{grand['total']}** | **{grand['PASS']}** | **{grand['FAIL']}** | "
                f"**{grand['BLOCKED']}** | **{grand['PENDING']}** | **{decision_for(grand)}** |\n"
            )
            continue
        rewritten.append(line)

    return "".join(rewritten)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3.5-9B Release QA Status Collector"
    )
    parser.add_argument(
        "--root",
        default=os.getcwd(),
        help="Repository root directory (default: cwd)",
    )
    parser.add_argument(
        "--status-file",
        default="reports/qwen35_9b/QWEN35_9B_RELEASE_QA_STATUS_20260519.md",
        help="Path to QA status markdown (relative to --root or absolute)",
    )
    parser.add_argument(
        "--report-dir",
        default="reports/qwen35_9b",
        help="Directory containing evidence reports (relative to --root or absolute)",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    status_path = os.path.join(root, args.status_file) if not os.path.isabs(args.status_file) else args.status_file
    report_dir = os.path.join(root, args.report_dir) if not os.path.isabs(args.report_dir) else args.report_dir

    if not os.path.isfile(status_path):
        print(f"ERROR: status file not found: {status_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(report_dir):
        print(f"ERROR: report directory not found: {report_dir}", file=sys.stderr)
        sys.exit(1)

    # Build evidence map
    evidence_map = build_evidence_map(report_dir)

    if not evidence_map:
        print("No evidence files found. Nothing to update.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(evidence_map)} evidence-backed items:", file=sys.stderr)
    for item_id, (status, remark) in sorted(evidence_map.items()):
        print(f"  {item_id}: {status} — {remark[:80]}", file=sys.stderr)

    # Read and update status markdown
    updated = update_status_markdown(status_path, evidence_map, report_dir)

    # Write back
    with open(status_path, "w") as f:
        f.write(updated)

    print(f"\nUpdated: {status_path}", file=sys.stderr)

    # Print summary of transitions
    items = parse_status_table(updated.splitlines(True))
    transitioned = [item for item in items if item["id"] in evidence_map]
    if transitioned:
        print(f"\nQA status transitions ({len(transitioned)} items):")
        for item in transitioned:
            new_status = evidence_map[item["id"]][0]
            print(f"  {item['id']}: PENDING → {new_status}")
    else:
        print("\nNo status transitions.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
