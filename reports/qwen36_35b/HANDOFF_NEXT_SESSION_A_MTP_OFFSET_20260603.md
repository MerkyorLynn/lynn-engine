# 新会话起跑手册 — 任务 A:MTP offset 对齐修复(2026-06-03 交接)

> 上个 session 上下文耗深,这里是干净交接。**立即目标 = 任务 A**:修 MTP draft-head 的 offset 对齐,
> 把 accept 从 2.4% 拉到 ~60%,decode ~45 → ~51 TPS。A 无论成败都转 B(Stage 6 零-shadow 内核)。

## 0. 一句话现状
NVFP4 专用引擎重启。Spark 单流 decode **36 → ~45 TPS(+26%)**,5 个 RC-validated 融合 cut,全提交。
README 6/3 banner 已 LIVE 在主页(main, CN+EN)。MTP 评估完:**wiring 正确(token-exact),但 accept 2.4%
= offset 对齐 bug,可修、不判负**。

## 1. 立即任务 A — MTP offset 修复(精确)
- **症状**:`scripts/spark_mtp_ab.py` 实测,两个 sidecar 都是 `TOKEN_EXACT=True` 但 accept = 2/82 ≈ **2.4%**
  → effective ~20 TPS(回归)。"基本随机"特征。
- **诊断(已收敛)**:不是引擎 correctness;是 draft-head ↔ serving 对齐。hidden 源 = pre-final-norm(**对**)。
  两个 sidecar 同 2.4%(系统性,非变体)→ **嫌疑 = offset 错位**(训练契约 offset=2,serving 可能当 offset-1 用)。
- **不判负硬依据**:llama.cpp APEX-MTP 跑**同一 head + 同模型**拿 **+13% / 60%+ accept**(79 vs 69.77)→ head 好、可达。
- **修复入口**:`engine/mtp_sidecar.py::mtp_logits`(head 的 offset 契约)vs serving 循环的 draft 定位
  (`engine/mtp_serving.py::speculative_step_kn_batched` + `engine/resident_runner.py::_mtp_draft_logits` @ ~1028)。
- **第一步(探针)**:加 draft-vs-actual 的 **±1 offset 探针** —— head 在步 N 的 draft 是否匹配实际输出的 N±1?
  确认 shift 方向,再改 offset。多半**配置级、不用重训**。交叉参考 llama.cpp APEX-MTP(`--spec-type draft-mtp`)的正确定位。
- **成功标准**:accept ≥ ~50-60% + `TOKEN_EXACT` 仍 True + effective TPS > 45(目标 ~51)。
- **复现**:`scripts/spark_mtp_ab.py`(toggle `LYNN_MTP_SPECULATIVE` 0/1 + `LYNN_MTP_SPECULATIVE_BATCHED=1`
  + `LYNN_MTP_SPECULATIVE_K=2`;sidecar 走 `LYNN_MTP_SIDECAR`)。看 accept True/False 比 + effective_token_tps + token-exact。
- **sidecars**:`/home/merkyor/models/mtp_sidecars/qwen36-35b-a3b-mtp/mtp.safetensors`(base)+ `-official-lynn-fused/`(都 2.4%)。

## 2. 怎么在 Spark 跑(fresh context 必读)
- 硬件:Spark GB10 **sm_121**,119GB unified,**240 GB/s BW,无 FP4 MMA**,CUDA 13 / torch 2.9.1。
- `ssh dgx`(frp 经 Tencent)—— **ssh 经常吞输出**:用 `docker run -d` 拿 container id + check-then-relaunch。
- 引擎跑在 **docker**:`lynn-eval-base:cu13`。**必须** `-e PYTHONNOUSERSITE=1`(否则挂载的 `~/.local` hf-hub 1.12
  盖掉容器 transformers 崩)。完整:`docker run -d --gpus all --ipc=host -e HOME=/home/merkyor -e PYTHONNOUSERSITE=1
  -e PYTHONUNBUFFERED=1 -v /home/merkyor:/home/merkyor -w /home/merkyor/lynn-engine lynn-eval-base:cu13 bash -lc "python3 -u scripts/X.py > reports/X.log 2>&1"`。
- 模型:`/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526`(load ~2.7min)。
- **跑前停 APEX**:`pkill -9 -f "APEX-MTP-I-Balanced"; sleep 12` → GPU free;**测完恢复 APEX**(setsid llama-server :18098,
  cmd 见战役文档/apex_manual_restart)。日志 `[N/A]` 是 GB10 unified mem 正常,不是错。
- **代码同步**:Spark repo `/home/merkyor/lynn-engine` 是 rsync 的(非 git),改完 `scp` 改动文件过去。
  本地 repo `/Users/lynn/Downloads/Lynn/lynn-engine`,branch `claude/fp8-9b-revival-graph-mtp-20260601`(HEAD `0a7be2e`)。
- 等长任务:用后台 waiter(`docker inspect ... Running` 轮询 → tail log),harness-tracked,完成自动通知。

## 3. 当前 RC-validated 栈(基线,别破)
7 flag 全 gated、默认关、RC battery 40/40 与 baseline 逐字一致、继承 **MMLU 84.40 / GPQA-Diamond 49.49**:
`LYNN_RMSNORM_FUSED` / `LYNN_FULL_ATTN_FUSED` / `LYNN_SHARED_EXPERT_FUSED` / `LYNN_LINEAR_ATTN_FUSE_GBETA` /
`LYNN_NVFP4_BF16_OUT` / `LYNN_DECODE_OPROJ_NOCOPY` + BASE_ENV(见任一 `scripts/spark_*_ab.py` 头部)。≈45 TPS。
- A/B 范式:同进程 toggle 一个 env、max-of-3、docker。质量关:`scripts/spark_rc_quality_regression.py`(把新 flag 加进 FUSION_FLAGS 重跑)。

## 4. A 之后 = B:Stage 6(真护城河)
fused **读 4-bit + 寄存器反量化 + bf16 GEMV + 零 shadow + 单 launch** NVFP4 内核。真墙是带宽(BF16 shadow 2× → ~40);
读 4-bit 把墙 ~40 → ~140(**70 活在这区间**)。同核挪 R6000 FP4 MMA = native 更快(跨设备 moat)。llama.cpp Q4_K_M = **MIT 蓝图**可 clean-room 参考。分阶段:单投影 PoC → 全 dense + 删 shadow(task #6)→ MoE grouped → 融合减 launch,每阶段 gate + RC。

## 5. 坑(别重蹈)
1. 只信 clean e2e A/B,**不信 profile section delta**(cuda-sync 灌水)。
2. `TOKEN_EXACT=False` ≠ 质量回归(reduction order);**RC battery 40/40 才是质量真验证**,单 prompt 相干不够。
3. 看到功能被禁用,**先查 git 历史 + 测试断言意图**,别假设是 bug(pseudo-tool-call 那次:v0.79.3 有意 pass-through)。
4. **ssh frp 吞输出** → `docker run -d` + check/relaunch。
5. **凭感觉估 launch 错一个量级**(估 140 实测 1527)→ census 实测。
6. **MoE/router 已 grouped**(census 证),别去啃高风险 router 减 launch。

## 6. 关键文件 / 协作
- 战役文档(全程 log + Stage 5 MTP 结果 + Stage 6 spec + offset 修复入口):
  `reports/qwen36_35b/DECODE_LAUNCH_OVERHEAD_CAMPAIGN_20260603.md`。
- 知乎稿:`reports/articles/NVFP4_ENGINE_RESTART_LAUNCH_OVERHEAD_CAMPAIGN_20260603.md`(已复制到 `~/Desktop/Claude/`)。
- main 有 README homepage(`a0ec00e`);branch HEAD `0a7be2e`(MTP 结果+article+doc)。
- 多-CLI:**claude-internal**(`claude -p "<task>" --dangerously-skip-permissions`,独立进程写核)→ lead 集成+GPU 验。
  **codex 仍 GATED**(上次 429 了用户的客户端,要用先问用户)。
