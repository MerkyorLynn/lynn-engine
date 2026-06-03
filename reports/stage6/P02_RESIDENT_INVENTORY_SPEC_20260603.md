# Stage 6 P0.2 — resident inventory gate

**Purpose:** after P0.1 proved no-reload prefill can be token-exact with the
60 GiB grouped-MoE shadow non-resident, P0.2 records what BF16 tensors still
live and which ones are plausible packed-prefill candidates.

## Runner

```bash
python3 scripts/spark_stage6_p02_resident_inventory.py
```

Spark/docker form:

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 -u scripts/spark_stage6_p02_resident_inventory.py
```

Optional inventory-only alias scan:

```bash
LYNN_P02_ATTACH_PROJECTION_ALIASES=1 python3 scripts/spark_stage6_p02_resident_inventory.py
```

Do not treat alias-scan mode as a decode-quality run; it only answers which
weights have resident packed aliases.

## Gate

P0.2 passes only when the report includes:

- BF16 total before release.
- BF16 total after `release_decode_bf16_shadows(include_projection_aliases=False)`.
- Category table for remaining BF16 resident tensors.
- Top tensor table.
- Packed-alias candidate bytes.
- Explicit next decision: which residents move to P1 projection kernels, which
  stay because deleting them changes decode semantics, and which belong to P2
  grouped MoE.

No speed claim is allowed at P0.2. It is a resident-byte and semantics inventory
gate.
