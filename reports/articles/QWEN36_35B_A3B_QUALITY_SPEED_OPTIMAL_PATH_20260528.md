# Qwen 3.6 35B-A3B: 质量与速度并存的最优解

如果只看参数规模,35B 模型很容易给人一种“本地跑不动”的印象。真正把它落到桌面或工作站时,关键其实不是参数量本身,而是三件事:

1. 量化以后质量还剩多少。
2. 单流响应能不能快到愿意日常使用。
3. 服务端并发时,吞吐是不是还能站住。

这几天我们在 Spark GB10 上围绕 Qwen 3.6 35B-A3B 做了一轮横向验证。结论很直接:

目前最稳的公开部署底座,仍然是 Q4_K_M-imatrix GGUF;而如果追求单流最快响应,新的 APEX-MTP I-Balanced GGUF 已经把 Spark 上的单流速度推到约 77 tok/s。

更重要的是,这不是用质量换速度的“少算专家”捷径。APEX-MTP 走的是 speculative decoding:先由 MTP draft 预测后续 token,再由主模型 verify。只要 verifier 路径正确,它追求的是更快得到同一模型 AR 路径会接受的 token,而不是把模型能力裁掉。

## 模型地址和 MTP 入口

这篇文章里说的 77 tok/s 版本,不是普通 Q4_K_M,而是 **APEX-MTP I-Balanced GGUF**。它的 MTP head 已经内嵌在同一个 GGUF 里,用 llama.cpp 时不需要再手动挂一个 draft sidecar。

推荐下载入口:

```text
Hugging Face(Lynn 镜像):
https://huggingface.co/nerkyor/Qwen3.6-35B-A3B-APEX-MTP-GGUF

ModelScope(国内镜像):
Merkyor/Qwen3.6-35B-A3B-APEX-MTP-GGUF

上游社区 APEX-MTP 源:
https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-MTP-GGUF
```

文件名和校验:

```text
Qwen3.6-35B-A3B-APEX-MTP-I-Balanced.gguf
size: 24.27 GiB / 26.06 GB
sha256: 9bf7d96bb3a9d363e645dd998aee9e9bff8e016a82aec7ff081e0e6cdb53419e
```

我们 Spark 当前生产本地路径:

```text
/home/merkyor/models/Qwen3.6-35B-A3B-APEX-MTP-GGUF/
  Qwen3.6-35B-A3B-APEX-MTP-I-Balanced.gguf
```

确认它真的带 MTP,看 GGUF metadata / tensor:

```text
qwen35moe.nextn_predict_layers = 1
blk.40.nextn.eh_proj.weight
blk.40.nextn.enorm.weight
blk.40.nextn.hnorm.weight
blk.40.nextn.shared_head_norm.weight
```

注意区分两条路线:

```text
llama.cpp 发布/部署路线:
  MTP head 已嵌入 GGUF,直接 --spec-type draft-mtp

Lynn engine 研究路线:
  仍可使用独立 MTP sidecar,例如
  /home/merkyor/models/mtp_sidecars/qwen36-35b-a3b-mtp-official-lynn-fused/mtp.safetensors
```

也就是说,用户下载上面的 APEX-MTP GGUF 以后,重点不是再找一个单独的 MTP 文件,而是确认自己拿到的是 **MTP 版 GGUF**。普通 APEX no-MTP 或普通 Q4_K_M base GGUF,即使能跑 Qwen 3.6 35B-A3B,也不能启用 `draft-mtp`。

## 先看质量:Q4_K_M 没有想象中脆

在同一台 Spark GB10 上,我们对 Qwen 3.6 35B-A3B 做过一组 thinking-off 质量锚点:

| 版本 | MMLU 500 5-shot | GPQA Diamond 198 0-shot |
|---|---:|---:|
| BF16 official | 86.40% | 45.45% |
| Q4_K_M-imatrix GGUF | 83.00% | 50.00% |
| Lynn-native W4A16 NVFP4 | 84.40% | 49.49% |

这个表有两个信息。

第一,Q4_K_M-imatrix 的 MMLU 相比 BF16 掉了 3.40 个百分点,但 GPQA 反而在这组样本上更高。不要把 GPQA 的小幅反超理解成量化“增强了模型”,更合理的解释是采样和解析噪声;但它足以说明,Q4_K_M-imatrix 在 35B-A3B 上不是“只能凑合用”的低质版本。

第二,W4A16 NVFP4 和 Q4_K_M-imatrix 的 GPQA 都在 49.5% 到 50.0% 这一档。也就是说,在这组足量 GPQA Diamond 上,我们没有看到某一种量化格式对 35B-A3B 产生决定性的质量优势。

另外还有一条 thinking-on 信号:Q4_K_M-imatrix 在 GPQA50、32K thinking budget 下跑到 78.00% naive accuracy,剔除 parse fail 后是 81.25%。这不是完整 198 题结果,不能直接和上表混算,但它说明 35B-A3B 在长思考模式下的上限很值得继续挖。

APEX-MTP I-Balanced 版本也已经有一组 32K thinking-on 正式质量数据:

| 版本 | 模式 | MMLU 500 5-shot | GPQA Diamond 198 | Tool-call |
|---|---|---:|---:|---:|
| APEX-MTP I-Balanced GGUF | thinking-on 32K | 90.00% | 78.79% naive / 83.87% excl parse fail | 12/15 |

这一组数据口径和 thinking-off 表不同,所以我不会把它们强行放在同一个 leaderboard 里比较。但它对使用路径很关键:APEX-MTP I-Balanced 不是只有速度的玩具包,它在 32K thinking-on 下已经表现出一个高质量 35B-A3B 本地模型该有的能力。

## 再看速度:Q4_K_M 是旧基线,APEX-MTP 是新单流上限

之前的 Spark 单流基线如下:

| 路线 | 栈 | 单流 TPS mean | Median | Peak |
|---|---|---:|---:|---:|
| BF16 official | SGLang dev-cu13 | 30.14 | 30.19 | 30.30 |
| Q4_K_M-imatrix GGUF | llama.cpp server-cuda | 69.77 | 69.76 | 70.08 |
| W4A16 NVFP4 | Lynn-native Config D | 38.96 | 38.85 | 39.18 |

所以在 5 月 18 日那轮测试里,Q4_K_M-imatrix 已经是 Spark 上最实用的 35B-A3B 路线:20GB 左右模型文件,加载快,质量不塌,单流接近 70 tok/s。

今天的新结果来自另一个包:Qwen 3.6 35B-A3B APEX-MTP I-Balanced GGUF。这个包不是普通 Q4_K_M,它是带 imatrix 的 mixed/balanced GGUF,并且内嵌了 MTP 相关 tensor:

```text
qwen35moe.nextn_predict_layers = 1
tensor mix: Q6_K 218, Q8_0 131, Q5_K 94, F32 308, BF16 2
```

在 llama.cpp server 里打开:

```bash
--spec-type draft-mtp
--spec-draft-n-max 4
```

短 HTTP benchmark 结果:

| 模式 | 单流 wall TPS | 单流 server TPS | draft accept | 4 路并发 wall TPS |
|---|---:|---:|---:|---:|
| AR baseline | 60.65 | 66.05 | n/a | 124.80 |
| APEX-MTP n_max=1 | 48.55 | 54.15 | 92.86% | 59.45 |
| APEX-MTP n_max=2 | 64.41 | 69.74 | 86.33% | 73.67 |
| APEX-MTP n_max=4 | 77.01 | 84.40 | 70.92% | 85.62 |

这张表的重点不是“n 越小越稳”,而是相反:

APEX-MTP n_max=4 是当前单流最优点。它相比同一服务里的 AR baseline,wall TPS 从 60.65 提到 77.01,提升约 27%。相比之前 Q4_K_M-imatrix 的 69.77 mean TPS,也有约 10% 的单流提升。

我又在当前 Spark 生产服务上做了一轮轻量 sanity check,3 次单流请求分别是 72.75 / 76.19 / 80.77 wall TPS,中位 76.19。也就是说,77 TPS 不是离线表格里的偶然数字,当前部署服务就是这条线。

## 但并发策略要分开看

APEX-MTP 现在最适合的是单流和低队列深度。它不是当前 4 路并发的最优解。

同一张表里,4 路并发时 AR baseline 是 124.80 wall TPS,而 APEX-MTP n_max=4 是 85.62 wall TPS。原因也很朴素:speculative decoding 会给每个 slot 增加 draft-context 工作;在当前 llama.cpp 服务循环里,多 slot 并发还没有把这部分开销摊得足够好。

所以正确的服务策略不是“一刀切永远开 MTP”,而是:

```text
单流 / 低队列深度:  APEX-MTP n_max=4
多并发 / 高队列深度: AR 或 request-level n_max=0
```

这也是接下来最值得工程化的方向:动态 MTP admission。用户单独对话时给最快单流体验;请求堆起来时自动回到 AR,保证总吞吐。

## 推荐使用路径

如果你只是想要一个稳、容易复现、质量已经过完整表格验证的 35B-A3B 本地版本:

```text
Qwen3.6-35B-A3B Q4_K_M-imatrix GGUF
llama.cpp
单流约 69.77 tok/s
MMLU 83.00%, GPQA Diamond 50.00%
```

这是当前最适合作为公开默认包的路线。它的优势是文件小、生态通用、Mac/Windows/Linux CUDA 都能接,而且质量数据已经足够清楚。

如果你在 Spark GB10 或类似 NVIDIA CUDA 环境上,更在意单流响应速度:

```bash
llama-server \
  -m Qwen3.6-35B-A3B-APEX-MTP-I-Balanced.gguf \
  --ctx-size 262144 \
  --parallel 4 \
  --n-gpu-layers 999 \
  -fa on \
  --jinja \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --reasoning auto \
  --reasoning-budget -1
```

这条路线目前是 Spark 上最快的单流 35B-A3B 路线,实测约 77 tok/s。它适合本地助手、单用户 agent、长上下文交互和“我希望一个大模型像中小模型一样响应”的场景。

当前 Spark 生产服务已经切到这条路线,启动参数是:

```text
Qwen3.6-35B-A3B-APEX-MTP-I-Balanced.gguf
--spec-type draft-mtp
--spec-draft-n-max 4
```

我又补了一次在线 sanity check,3 次单流请求分别是 72.75 / 76.19 / 80.77 wall TPS,中位数 76.19,和 77 tok/s 的正式 A/B 数据一致。

如果你要做多人并发服务,现在不要静态全局开 MTP。更合理的是做路由:

```text
interactive single user -> MTP n_max=4
queued batch / many users -> AR
```

## HF / ModelScope 上传时最容易踩的坑

如果要把 APEX-MTP 版本传到 Hugging Face 或 ModelScope,不能只传一个普通 Q4_K_M base GGUF,然后写“支持 MTP”。

APEX-MTP 能不能跑,关键要看 GGUF 里有没有 MTP 能力:

```text
qwen35moe.nextn_predict_layers = 1
blk.40.nextn.* tensors present
```

也就是说,可发布的形式有两种:

1. 一个内嵌 MTP tensor 的 GGUF。
2. base GGUF 加一个明确命名的 MTP sidecar,并写清楚 loader 方式。

对普通用户来说,第一种更好。下载一个文件,启动命令加 `--spec-type draft-mtp --spec-draft-n-max 4`,体验最直接。

## 5 月 28 日补充:Q4_K_M + MTP 不是直接替换答案

今天的 77 tok/s 版本是 I-Balanced,不是 Q4_K_M。它已经是 imatrix 量化,但不是最低比特的 Q4_K_M-imatrix。理论上,如果我们重新做一个“带 MTP tensor 的 Q4_K_M-imatrix GGUF”,Spark 这种偏 memory bandwidth bound 的平台还有机会更快。

后续实测把这个判断修正得更清楚:我们确实从带 `mtp.*` 的 BF16/HF 源重新 convert + imatrix + Q4_K_M 量化出了带 MTP tensor 的 GGUF,但它不应该直接替换现有公开默认。

关键数据:

- Q4_K_M-MTP artifact,AR-only 单流:77.73 tok/s server TPS,76.51 tok/s wall TPS。
- 同一个 Q4_K_M-MTP artifact,开启 draft-MTP n=4:64.08 tok/s server TPS,63.12 tok/s wall TPS,accept rate 约 36.34%。
- Q4_K_M-MTP think-off:MMLU 81.40%,GPQA 41.41%。
- APEX-MTP I-Balanced think-off:MMLU 82.20%,GPQA 42.42%。

所以,Q4_K_M+MTP 这条路线的技术结论是:artifact 可以做,MTP tensor 也能保住,但当前 `draft-mtp` runtime 对这组 think-off/短输出场景不是最优。accept rate 只有 36% 左右时,draft 上下文和 verify 调度成本会吃掉 speculative decoding 的收益;质量上,APEX-MTP 系列在 think-off GPQA 上也没有守住旧 Q4_K_M/W4A16 锚点。

这不是放弃 MTP,而是把使用边界写清楚:thinking-on 长推理仍然可以继续用 APEX-MTP I-Balanced;think-off 质量默认仍然应该看普通 Q4_K_M-imatrix 或 W4A16 NVFP4;Q4_K_M-MTP 暂时保留为研究 artifact,不作为默认发布主包。

## 结论

Qwen 3.6 35B-A3B 现在已经不是“能不能在本地跑”的问题,而是“应该用哪条路线跑”的问题。

我的当前判断:

- 稳定公开默认:Q4_K_M-imatrix GGUF。
- Spark 单流最快研究信号:Q4_K_M-MTP artifact 关掉 draft-MTP 后约 77.7 tok/s;生产 fallback 仍保持 APEX-MTP I-Balanced。
- 并发服务:需要动态 admission,单流开 MTP,高并发回 AR。
- 下一代最佳包:先解决 MTP accept rate 和 think-off GPQA,再考虑把 Q4_K_M-MTP 作为公开主包。

这就是我理解的“质量与速度并存”的最优解:不是盲目追更低比特,也不是盲目追更复杂 kernel,而是在质量锚点守住以后,把 speculative decoding 这种可验证的加速路径接到真正能服务用户的 runtime 里。

## 数据来源

- Q4_K_M / BF16 / W4A16 质量与 Spark TPS: `reports/ops/QWEN36_35B_W4A16_OVERNIGHT_STATUS_20260518.md`
- Spark 单流三路线 TPS: `reports/spark/SPARK_QWEN36_SINGLE_STREAM_TPS_BASELINE_20260518.md`
- Q4_K_M thinking-on GPQA50: `reports/qwen36_35b/QWEN36_35B_Q4KM_THINKING32_GPQA50_20260520.md`
- APEX-MTP I-Balanced 32K thinking-on quality: `reports/qwen36_35b/apex_quality32k_20260521/`
- APEX-MTP I-Balanced 单流/并发 A/B: `reports/mtp/LLAMA_CPP_APEX_MTP_SERVICE_AB_20260528.md`
