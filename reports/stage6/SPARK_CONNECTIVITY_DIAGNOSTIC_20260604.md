# Spark Connectivity Diagnostic

Date: 2026-06-04

Verdict: **Spark GPU gates are blocked by SSH connectivity; no GPU runtime
artifact is banked by this diagnostic.**

## Findings

| Path | Result | Evidence |
|---|---|---|
| `dgx-spark` / `dgx` | blocked | SSH alias targets `tencent:127.0.0.1:2224`; jump host has no `2224` listener. |
| `dgx-via-ssh` | blocked | SSH alias targets `lynn-jump:127.0.0.1:2222`; jump host has no `2222` listener. |
| `frps` on Tencent jump | server alive, Spark client gone | `frps` listens on `:7000` and dashboard `127.0.0.1:7500`, but `ssh-dgx` proxy closed at `2026-06-04 04:45:23 CST` and did not re-register. |
| `dgx-via-n5` | blocked | N5 is reachable and can ping/connect TCP to Spark `192.168.100.26:22`, but SSH times out during banner exchange. |
| N5 -> Spark LAN | alive at TCP layer | `nc -vz 192.168.100.26 22` succeeds and ping replies with sub-millisecond RTT. |

## Interpretation

Spark is visible on the N5 LAN, but the SSH service path is not usable:

- the FRP client on Spark exited, taking `ssh-dgx` and service proxies down;
- the autossh fallback on `:2222` is not listening on the jump host;
- direct LAN SSH through N5 establishes TCP but does not receive an SSH banner.

That points to a Spark-side SSH daemon/session hang, firewall/session limit, or
Spark-side tunnel process failure. From the current workstation/jump access,
there is no safe way to run GPU gates until the Spark SSH path is restored.

## Unblock Checklist

From Spark console or any already-open shell on Spark:

```bash
sudo systemctl status ssh sshd || true
sudo systemctl restart ssh || sudo systemctl restart sshd
ps aux | grep -E 'frpc|autossh' | grep -v grep || true
```

Then from this repo:

```bash
ssh dgx-via-n5 'hostname; nvidia-smi'
scripts/run_spark_stage6_p4_runtime_bridge_preflight.sh --host dgx-via-n5
```

If FRP is restored instead:

```bash
ssh dgx-spark 'hostname; nvidia-smi'
scripts/run_spark_stage6_p4_runtime_bridge_preflight.sh --host dgx-spark
```

Keep `P4 fused kernel banked=false` until the runtime bridge summary returns
`PASS_TWO_STAGE_RUNTIME_BRIDGE` with `native_backend_call_count.delta == 1`.
