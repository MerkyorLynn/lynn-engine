"""
Parity report JSON schema for Lynn engine vs SGLang production token-for-token verification.

All P2 / P3 parity tests emit JSON in this shape. Daily regression detector (P3.daily)
consumes the same shape and diffs day-over-day.

Schema is intentionally explicit + versioned. Any breaking change → bump SCHEMA_VERSION.

See docs/PHASE4_REFERENCE_WORKLOAD.md for context.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import hashlib
import json
import platform


SCHEMA_VERSION = "1.0.0"


@dataclass
class TokenComparison:
    """Per-prompt token-level diff between Lynn engine and oracle (SGLang nightly)."""
    prompt_id: str                    # stable id, e.g. "v4_distill_short_03"
    prompt_text: str                  # the raw prompt text
    prompt_text_sha256: str           # hash for reproducibility
    max_new_tokens: int
    decoding: str = "greedy"          # currently P3 only supports greedy
    lynn_token_ids: list = field(default_factory=list)
    lynn_decoded_text: str = ""
    oracle_token_ids: list = field(default_factory=list)
    oracle_decoded_text: str = ""
    first_divergence_idx: Optional[int] = None  # None = no divergence
    matched_token_count: int = 0
    exact_match: bool = False
    notes: str = ""

    @classmethod
    def from_token_lists(cls, prompt_id: str, prompt_text: str, max_new_tokens: int,
                         lynn_ids: list, oracle_ids: list, tokenizer=None):
        """Construct + auto-compute divergence metrics."""
        n = min(len(lynn_ids), len(oracle_ids))
        first_div = None
        matched = 0
        for i in range(n):
            if lynn_ids[i] == oracle_ids[i]:
                matched += 1
            else:
                first_div = i
                break
        else:
            matched = n
            first_div = None if len(lynn_ids) == len(oracle_ids) else min(len(lynn_ids), len(oracle_ids))
        exact = first_div is None and len(lynn_ids) == len(oracle_ids)
        lynn_text = tokenizer.decode(lynn_ids, skip_special_tokens=True) if tokenizer else ""
        oracle_text = tokenizer.decode(oracle_ids, skip_special_tokens=True) if tokenizer else ""
        return cls(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            prompt_text_sha256=hashlib.sha256(prompt_text.encode()).hexdigest()[:16],
            max_new_tokens=max_new_tokens,
            lynn_token_ids=lynn_ids,
            lynn_decoded_text=lynn_text,
            oracle_token_ids=oracle_ids,
            oracle_decoded_text=oracle_text,
            first_divergence_idx=first_div,
            matched_token_count=matched,
            exact_match=exact,
        )


@dataclass
class ParityRunMeta:
    """Top-level metadata for one parity run (one full N≥20 invocation)."""
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""                  # ISO timestamp + git sha typically
    timestamp_utc: str = ""
    lynn_engine_git_sha: str = ""
    lynn_engine_branch: str = ""
    oracle_stack: str = "SGLang dev-cu13 nightly"
    oracle_endpoint: str = ""
    oracle_model_name: str = ""
    workload_name: str = "Lynn-V4-Distill-Qwen-35B-A3B"
    checkpoint_name: str = ""         # which ckpt: bf16_merged / nvfp4_v8rtn / etc
    quant_format: str = ""
    host: str = field(default_factory=platform.node)
    cuda_visible: str = ""
    notes: str = ""


@dataclass
class ParityRunSummary:
    """Aggregated metrics for the run."""
    n_prompts: int = 0
    n_exact_match: int = 0
    n_first_div_le_5_tokens: int = 0
    n_first_div_le_20_tokens: int = 0
    n_first_div_gt_20_tokens: int = 0
    avg_matched_tokens: float = 0.0
    verdict: str = ""                 # "PASS_BF16_20_OF_20" or "FAIL_DEQUANT_18_OF_20" etc
    pass_threshold_for_format: dict = field(default_factory=lambda: {
        "bf16": 20,                   # require all 20 exact match
        "compressed_tensors_nvfp4": 18,  # allow 2 prompts FP4 noise
        "modelopt_fp4": 18,
        "compressed_tensors_fp8": 19,    # FP8 should be very close
    })


@dataclass
class ParityRunReport:
    """Top-level JSON shape written to disk."""
    meta: ParityRunMeta = field(default_factory=ParityRunMeta)
    summary: ParityRunSummary = field(default_factory=ParityRunSummary)
    comparisons: list = field(default_factory=list)   # list[TokenComparison]
    # Per-day diff vs previous run (populated by daily regression runner only)
    delta_vs_previous: Optional[dict] = None

    def compute_summary(self):
        """Recompute summary from comparisons list."""
        n = len(self.comparisons)
        ex = sum(1 for c in self.comparisons if c.exact_match)
        le5 = sum(1 for c in self.comparisons if c.first_divergence_idx is not None and c.first_divergence_idx <= 5)
        le20 = sum(1 for c in self.comparisons if c.first_divergence_idx is not None and c.first_divergence_idx <= 20)
        gt20 = sum(1 for c in self.comparisons if c.first_divergence_idx is not None and c.first_divergence_idx > 20)
        avg_match = sum(c.matched_token_count for c in self.comparisons) / max(1, n)
        self.summary.n_prompts = n
        self.summary.n_exact_match = ex
        self.summary.n_first_div_le_5_tokens = le5
        self.summary.n_first_div_le_20_tokens = le20
        self.summary.n_first_div_gt_20_tokens = gt20
        self.summary.avg_matched_tokens = round(avg_match, 2)
        # Verdict
        threshold = self.summary.pass_threshold_for_format.get(self.meta.quant_format, n)
        passes = ex if self.meta.quant_format == "bf16" else (n - gt20)  # for non-bf16, "near-match" counts
        self.summary.verdict = f"{'PASS' if passes >= threshold else 'FAIL'}_{self.meta.quant_format.upper()}_{passes}_OF_{n}"

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)

    def save(self, path: str):
        self.compute_summary()
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


def _example():
    """Minimal example to validate the schema parses cleanly."""
    rep = ParityRunReport()
    rep.meta.run_id = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_test"
    rep.meta.timestamp_utc = datetime.utcnow().isoformat()
    rep.meta.checkpoint_name = "bf16_merged"
    rep.meta.quant_format = "bf16"
    rep.comparisons.append(TokenComparison(
        prompt_id="example_001",
        prompt_text="hello",
        prompt_text_sha256=hashlib.sha256(b"hello").hexdigest()[:16],
        max_new_tokens=32,
        lynn_token_ids=[1, 2, 3, 4, 5],
        oracle_token_ids=[1, 2, 3, 4, 5],
        first_divergence_idx=None,
        matched_token_count=5,
        exact_match=True,
    ))
    rep.compute_summary()
    print(rep.to_json())


if __name__ == "__main__":
    _example()
