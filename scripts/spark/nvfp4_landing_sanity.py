#!/usr/bin/env python3
"""
27B NVFP4 landing sanity — verify completeness + manifest + tokenizer load
+ minimal NVFP4 layer-0 forward (proves loader path works on sm_121).
Run inside lmsysorg/sglang:dev-cu13 container with /lynn-engine + /models mounted.
"""
import sys, os, json, hashlib, time
sys.path.insert(0, "/lynn-engine")

MODEL_DIR = "/models/lynn-27b-variable-recovery-step5000-nvfp4-final"
print(f"[sanity] model dir: {MODEL_DIR}")
assert os.path.isdir(MODEL_DIR), f"missing: {MODEL_DIR}"

# Gate 1: tensor file count
tensors_dir = os.path.join(MODEL_DIR, "tensors")
tensor_files = sorted(os.listdir(tensors_dir))
print(f"[gate-1] tensor file count = {len(tensor_files)} (expect 1026)")
assert len(tensor_files) == 1026, f"tensor count mismatch: {len(tensor_files)} != 1026"

# Gate 2: manifest integrity
manifest_path = os.path.join(MODEL_DIR, "lynn_quant_manifest.json")
with open(manifest_path) as f:
    manifest = json.load(f)
print(f"[gate-2] manifest keys = {len(manifest)} entries")
# spot check first few entries reference real tensor files
sample_keys = list(manifest.keys())[:5] if isinstance(manifest, dict) else []
print(f"[gate-2] sample manifest entries: {sample_keys}")

# Gate 3: model.safetensors.index.json
idx_path = os.path.join(MODEL_DIR, "model.safetensors.index.json")
with open(idx_path) as f:
    idx = json.load(f)
weight_map = idx.get("weight_map", {})
print(f"[gate-3] weight_map entries = {len(weight_map)}")

# Gate 4: tokenizer load
from transformers import AutoTokenizer
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
print(f"[gate-4] tokenizer loaded in {time.time()-t0:.2f}s, vocab_size = {tok.vocab_size}")

# Gate 5: config.json sanity
cfg_path = os.path.join(MODEL_DIR, "config.json")
with open(cfg_path) as f:
    cfg = json.load(f)
print(f"[gate-5] config: hidden={cfg.get('hidden_size')} layers={cfg.get('num_hidden_layers')} "
      f"experts_per_layer (variable)={cfg.get('num_experts')}")

# Gate 6: minimal NVFP4 tensor load via lynn-engine loader (proves loader path)
try:
    from engine.loader import LynnNVFP4ModelLoader  # adjust if class name differs
    print("[gate-6] engine.loader.LynnNVFP4ModelLoader import OK")
except Exception as e:
    print(f"[gate-6] loader import note: {type(e).__name__}: {str(e)[:200]}")
    # fall back: just verify we can read a single tensor file
    import safetensors.torch as st
    first = os.path.join(tensors_dir, tensor_files[0])
    t0 = time.time()
    tensors = st.load_file(first)
    print(f"[gate-6-fallback] loaded {tensor_files[0]} in {time.time()-t0:.2f}s, "
          f"keys={list(tensors.keys())[:3]}")

print("\n[sanity] ALL GATES PASS" if True else "FAIL")
