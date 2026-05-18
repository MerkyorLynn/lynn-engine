# P133/P134 MoE Fixture Contract — R6000 Validation Report

Date: 2026-05-18
Machine: NVIDIA RTX PRO 6000 Blackwell Server Edition (98 GB VRAM)
Branch: `claude/moe-fixture-contract-20260518`

## Results

### P133 Export: 18/18 fixtures exported successfully

| Layer | Prompt 0 ("Hello") | Prompt 1 ("The capital of France is") |
|-------|--------------------|-----------------------------------------|
| L00 | experts=[238,112,106,127,157,66,56,120] h_norm=32.87 | experts=[127,49,112,11,153,85,146,18] h_norm=24.70 |
| L04 | experts=[88,195,172,206,154,233,87,27] h_norm=40.60 | experts=[12,25,154,204,215,217,110,80] h_norm=31.37 |
| L08 | experts=[123,175,17,217,127,207,223,141] h_norm=46.78 | experts=[120,183,52,82,191,32,232,72] h_norm=45.27 |
| L16 | experts=[89,245,88,166,203,16,107,141] h_norm=58.03 | experts=[130,51,233,132,98,244,49,30] h_norm=45.25 |
| L20 | experts=[17,127,175,217,223,18,11,8] h_norm=60.62 | experts=[120,82,183,191,232,72,186,56] h_norm=46.65 |
| L28 | experts=[158,107,88,166,93,218,18,89] h_norm=66.79 | experts=[233,98,16,30,130,137,132,113] h_norm=44.45 |
| L32 | experts=[191,185,112,237,106,134,61,68] h_norm=55.39 | experts=[221,171,41,232,102,254,130,13] h_norm=52.81 |
| L36 | experts=[14,73,45,120,148,69,55,106] h_norm=68.13 | experts=[49,61,195,92,14,73,210,12] h_norm=59.47 |
| L39 | experts=[75,15,201,108,9,239,219,241] h_norm=58.61 | experts=[244,72,146,182,183,14,36,213] h_norm=55.43 |

Timing: 59.2s load + 69.0s total export
Fixture size: 8.5 KB each, 252 KB total

### P134 Contract: 18/18 GREEN

```
FIXTURE                      MAX_ABS   MEAN_ABS     REL_L2        COS  EX  REF_MS STATUS
L00/P00                     0.00e+00   0.00e+00   0.00e+00 1.00000000   1   0.997 GREEN
L04/P00                     0.00e+00   0.00e+00   0.00e+00 1.00000000   1   1.248 GREEN
L08/P00                     0.00e+00   0.00e+00   0.00e+00 1.00000000   1   0.969 GREEN
L16/P00                     0.00e+00   0.00e+00   0.00e+00 1.00000000   1   0.960 GREEN
L20/P00                     0.00e+00   0.00e+00   0.00e+00 1.00000000   1   0.960 GREEN
L28/P00                     0.00e+00   0.00e+00   0.00e+00 1.00000000   1   0.946 GREEN
L32/P00                     0.00e+00   0.00e+00   0.00e+00 1.00000000   1   1.959 GREEN
L36/P00                     0.00e+00   0.00e+00   0.00e+00 1.00000000   1   0.954 GREEN
L39/P00                     0.00e+00   0.00e+00   0.00e+00 1.00000000   1   0.955 GREEN
L00/P01                     0.00e+00   0.00e+00   0.00e+00 1.00000000   1   2.763 GREEN
L04/P01                     0.00e+00   0.00e+00   0.00e+00 1.00000000   1   1.833 GREEN
L08/P01                     0.00e+00   0.00e+00   0.00e+00 1.00000000   1   1.813 GREEN
L16/P01                     0.00e+00   0.00e+00   0.00e+00 1.00000000   1   1.032 GREEN
L20/P01                     0.00e+00   0.00e+00   0.00e+00 1.00000000   1   1.000 GREEN
L28/P01                     0.00e+00   0.00e+00   0.00e+00 1.00000000   1   0.957 GREEN
L32/P01                     0.00e+00   0.00e+00   0.00e+00 1.00000000   1   0.955 GREEN
L36/P01                     0.00e+00   0.00e+00   0.00e+00 1.00000000   1   0.951 GREEN
L39/P01                     0.00e+00   0.00e+00   0.00e+00 1.00000000   1   0.960 GREEN
```

**VERDICT: ALL 18 GREEN. max_abs=0 for all fixtures. Bit-exact reproduction confirmed.**

## Fixture Specification

Each fixture safetensors file (8.5 KB) contains:

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `hidden_in` | [1, 2048] | bfloat16 | MoE sublayer input (post_attention_layernorm output) |
| `expert_ids` | [8] | int32 | Top-8 expert indices from router |
| `routing_weights` | [8] | float32 | Softmax routing weights |
| `moe_output` | [1, 2048] | bfloat16 | Ground truth MoE output (routed + shared expert) |

Manifest v2 additionally records per-fixture:
- `prompt_text`, `prompt_tokens`, `token_pos`
- `hidden_in_norm`, `moe_output_norm`
- `hidden_in_sha256`, `moe_output_sha256` (integrity check)
- `tensor_shapes`, `tensor_dtypes`
- `sidecar.layer_prefix` (for native kernel weight loading)
- `sidecar.folded_scale_model_dir` (for NVFP4 folded-scale path)

## How to Use (Stream A Native Kernel Developer)

### Quick validation (~10s per iteration):

```python
from safetensors.torch import load_file
from engine.loader import load_qwen36_layer

# Load one fixture
fixture = load_file("reports/qwen36_35b/p133_fixtures/layer_28_prompt_00.safetensors", device="cuda")
hidden_in = fixture["hidden_in"]        # [1, 2048] BF16
expert_ids = fixture["expert_ids"]      # [8] int32
routing_weights = fixture["routing_weights"]  # [8] float32
expected = fixture["moe_output"]        # [1, 2048] BF16 ground truth

# Load layer weights (or native packed weights)
w, _ = load_qwen36_layer(model_dir, layer_idx=28, ...)

# Run your native kernel
output = your_native_moe_kernel(hidden_in, expert_ids, routing_weights, w)

# Contract check
diff = (output - expected).abs().max().item()
print(f"max_abs: {diff:.2e}")  # Target: < 5e-3
```

### Full contract gate:

```bash
python benchmarks/p134_active_moe_fixture_contract.py \
    --fixtures reports/qwen36_35b/p133_fixtures \
    --model-dir /path/to/model \
    --candidate-output-dir /path/to/native_outputs \
    --max-abs-threshold 0.005 \
    --cosine-threshold 0.999
```

### Debug intermediates (when drift occurs):

```bash
python benchmarks/p133_export_active_moe_fixtures.py \
    --model-dir /path/to/model \
    --layers 28 \
    --export-intermediates \
    --out /tmp/debug_fixtures
```

This exports per-expert `gate_out`, `up_out`, `inter`, `down_out`, `weighted_sum` to isolate which stage drifts.

## Sidecar: Folded-Scale Native Weights

Manifest records the folded-scale sidecar model path:
```
/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-moe-repack-folded-scale-v0
```

Native kernel developers load expert weights from this sidecar using:
```
layer_prefix = f"model.language_model.layers.{layer_id}."
# Expert weight keys:
#   {layer_prefix}mlp.experts.{e}.gate_proj.weight_packed
#   {layer_prefix}mlp.experts.{e}.gate_proj.weight_scale
#   {layer_prefix}mlp.experts.{e}.gate_proj.weight_global_scale
#   (same for up_proj, down_proj)
```

## File Inventory

| File | Size | Purpose |
|------|------|---------|
| `benchmarks/p133_export_active_moe_fixtures.py` | 12 KB | Fixture export (memory-efficient, monkey-patch hook) |
| `benchmarks/p134_active_moe_fixture_contract.py` | 9 KB | Contract test (self-check + candidate comparison) |
| `scripts/r6000_export_qwen36_moe_fixtures.sh` | 5 KB | One-shot R6000 automation |
| `reports/qwen36_35b/p133_fixtures/manifest.json` | 30 KB | Full fixture metadata |
| `reports/qwen36_35b/p133_fixtures/*.safetensors` | 18 × 8.5 KB | Fixture tensors |
| `reports/qwen36_35b/p134_triton_selfcheck_report.json` | 7 KB | Contract test JSON output |
| `docs/QWEN36_W4A16_KERNEL_REFACTOR_PLAN_20260518.md` | 4 KB | Architecture + rationale |

## Acceptance Checklist

- [x] `python -m py_compile` passes for p133 and p134
- [x] R6000 p133 exports 18 fixtures (9 layers × 2 prompts)
- [x] R6000 p134 self-check: 18/18 GREEN, max_abs=0
- [x] Does not modify engine/csrc/triton_kernels/server
- [x] Does not conflict with Stream A/B
- [x] Fixtures validated as deterministic contract gate
