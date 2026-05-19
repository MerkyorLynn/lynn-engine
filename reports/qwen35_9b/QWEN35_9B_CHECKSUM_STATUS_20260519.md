# Qwen3.5-9B Checksum Status · 2026-05-19

## Scope

Release checksum generator and verifier for Qwen3.5-9B Q4_K_M GGUF files and Lynn-native NVFP4 W4A16 directories.

## Added files

- `scripts/qwen35_9b_release_checksums.py`
- `scripts/local_qwen35_9b_verify_checksums.sh`
- `docs/QWEN35_9B_INSTALL_QUICKSTART_20260519.md`
- `reports/qwen35_9b/QWEN35_9B_CHECKSUM_STATUS_20260519.md`

## Behavior

- Python tool is stdlib-only.
- `generate` accepts files and directories and emits stable sorted relative paths.
- Manifest format: `<sha256>\t<size_bytes>\t<relative_path>`.
- `verify` checks file existence, file size, and SHA256, writes JSON summary, and exits nonzero on missing or mismatched files.
- Shell wrapper defaults to `~/Models/Lynn/Qwen3.5-9B`, supports `--dry-run`, and exits nonzero when the root, manifest, or manifest-listed files are absent.
- No GPU commands and no model downloads are used.

## Validation commands

```bash
python3 -m py_compile scripts/qwen35_9b_release_checksums.py
bash -n scripts/local_qwen35_9b_verify_checksums.sh
```

Temporary two-file pass/fail validation:

```bash
python3 scripts/qwen35_9b_release_checksums.py generate \
  --paths /tmp/qwen35_checksum_test \
  --out /tmp/qwen35_checksum_test/checksums.sha256

python3 scripts/qwen35_9b_release_checksums.py verify \
  --manifest /tmp/qwen35_checksum_test/checksums.sha256 \
  --root /tmp \
  --out /tmp/qwen35_checksum_test/verify_pass.json

printf 'mutated' >> /tmp/qwen35_checksum_test/file_a.txt
python3 scripts/qwen35_9b_release_checksums.py verify \
  --manifest /tmp/qwen35_checksum_test/checksums.sha256 \
  --root /tmp \
  --out /tmp/qwen35_checksum_test/verify_fail.json
```

Expected result: first verify passes; second verify fails nonzero after mutation.
