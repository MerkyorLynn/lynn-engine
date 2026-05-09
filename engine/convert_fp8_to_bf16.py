"""
Lynn Engine — FP8 e4m3 block-scaled → BF16 offline converter.

Why: HF transformers loading of `qwen3_5_moe` FP8 fine-grained quantization
needs the `kernels` package + DeepGEMM, whose Hub metadata is malformed and
breaks the forward pass (DESIGN.md §13.5 P1.2 blocker). Loading the same
model in plain BF16 sidesteps the entire FP8 kernel path — HF transformers
runs cleanly because there is no quant-config to honor.

Trade-off: 35 GB FP8 → 70 GB BF16 on disk. Spark has 2.4 TB free, fine.

This script is CPU-only and does NOT touch CUDA — runs alongside live vLLM
on Spark without disturbing the GPU.

Usage:
    python convert_fp8_to_bf16.py \
        --src /home/merkyor/models/Qwen3.6-35B-A3B-FP8 \
        --dst /home/merkyor/models/Qwen3.6-35B-A3B-BF16
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def dequant_fp8_block(weight_fp8: torch.Tensor,
                      scale_inv: torch.Tensor,
                      block_size=(128, 128),
                      dtype=torch.bfloat16) -> torch.Tensor:
    """vLLM FP8 e4m3 block-scaled dequant.

    weight_fp8 : [out, in]  float8_e4m3fn
    scale_inv  : [out//bs0, in//bs1]  float32
    Output      : [out, in]  bf16
    """
    bs0, bs1 = block_size
    out_dim, in_dim = weight_fp8.shape
    # Broadcast scale to weight grid via repeat_interleave
    scale_full = (
        scale_inv
        .repeat_interleave(bs0, dim=0)
        .repeat_interleave(bs1, dim=1)
    )
    scale_full = scale_full[:out_dim, :in_dim]
    return (weight_fp8.to(torch.float32) * scale_full.to(torch.float32)).to(dtype)


def convert_one_file(src_st_path: Path, dst_st_path: Path,
                     block_size, target_dtype):
    """Read one .safetensors file, dequant FP8 tensors, write new .safetensors."""
    out_tensors = {}

    with safe_open(src_st_path, framework="pt", device="cpu") as st:
        keys = list(st.keys())
        # First pass: collect all weights + scales
        weights = {}
        scales = {}
        passthrough = {}
        for k in keys:
            t = st.get_tensor(k)
            if k.endswith(".weight_scale_inv"):
                base = k[: -len(".weight_scale_inv")]
                scales[base] = t
            elif t.dtype == torch.float8_e4m3fn:
                weights[k] = t
            else:
                passthrough[k] = t

    # Dequant each FP8 weight using its paired scale
    for w_key, w in weights.items():
        base = w_key[: -len(".weight")] if w_key.endswith(".weight") else w_key
        scale = scales.get(base)
        if scale is None:
            raise SystemExit(
                f"FP8 weight {w_key} has no matching .weight_scale_inv in {src_st_path.name}"
            )
        out_tensors[w_key] = dequant_fp8_block(w, scale, block_size, target_dtype).contiguous()

    # Pass-through tensors converted to target dtype where applicable
    for k, t in passthrough.items():
        if t.is_floating_point():
            out_tensors[k] = t.to(target_dtype).contiguous()
        else:
            out_tensors[k] = t.contiguous()

    save_file(out_tensors, str(dst_st_path), metadata={"format": "pt"})
    return len(out_tensors), sum(t.element_size() * t.numel() for t in out_tensors.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Qwen3.6-35B-A3B-FP8 model dir")
    ap.add_argument("--dst", required=True, help="output BF16 model dir")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--limit-files", type=int, default=0,
                    help="convert only the first N safetensors files (smoke test)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    target_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    # Parse config + block_size
    cfg_path = src / "config.json"
    cfg = json.loads(cfg_path.read_text())
    quant_cfg = cfg.get("quantization_config", {})
    block_size = quant_cfg.get("weight_block_size", [128, 128])
    print(f"FP8 block_size: {block_size}")

    # Write new config without quantization_config + force dtype to bf16
    cfg_out = {k: v for k, v in cfg.items() if k != "quantization_config"}
    if "text_config" in cfg_out:
        cfg_out["text_config"]["dtype"] = args.dtype
    cfg_out["dtype"] = args.dtype
    (dst / "config.json").write_text(json.dumps(cfg_out, indent=2, ensure_ascii=False))
    print(f"Wrote {dst}/config.json (no quantization_config, dtype={args.dtype})")

    # Copy non-weight files (tokenizer, chat template, etc.)
    skip_suffixes = (".safetensors", ".safetensors.index.json", ".bak")
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(skip_suffixes) and f.name != "config.json":
            shutil.copy2(f, dst / f.name)
            print(f"  copied {f.name}")

    # Find all safetensors files
    st_files = sorted(src.glob("*.safetensors"))
    if args.limit_files:
        st_files = st_files[: args.limit_files]
    print(f"\nConverting {len(st_files)} safetensors files...")

    # Build new index
    new_index = {"metadata": {"total_size": 0}, "weight_map": {}}
    total_t = 0
    total_b = 0
    t_start = time.time()

    for i, src_f in enumerate(st_files):
        dst_f = dst / src_f.name
        t0 = time.time()
        n_t, n_b = convert_one_file(src_f, dst_f, block_size, target_dtype)
        elapsed = time.time() - t0
        total_t += n_t
        total_b += n_b
        size_gb = n_b / 1e9
        rate = size_gb / elapsed if elapsed > 0 else 0
        print(
            f"  [{i+1:3}/{len(st_files)}] {src_f.name:<30} "
            f"{n_t:4} tensors, {size_gb:5.2f} GB BF16, {elapsed:5.1f}s ({rate:.2f} GB/s)",
            flush=True,
        )

        # Update index
        with safe_open(dst_f, framework="pt", device="cpu") as st:
            for k in st.keys():
                new_index["weight_map"][k] = src_f.name

    new_index["metadata"]["total_size"] = total_b
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(new_index, indent=2, ensure_ascii=False)
    )

    total_elapsed = time.time() - t_start
    print(
        f"\nDone. {total_t:,} tensors, {total_b/1e9:.2f} GB BF16 in {total_elapsed:.1f}s "
        f"({total_b/1e9/total_elapsed:.2f} GB/s)"
    )
    print(f"Output: {dst}")


if __name__ == "__main__":
    main()
