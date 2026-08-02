# miniSeedance

一个最小但完整的文生视频（text-to-video）/ 音视频联合生成模型。采样方法与训练目标取自 [Self-Flow](../Self-Flow)，Video VAE 与 DiT 结构参考 [daVinci-MagiHuman](../daVinci-MagiHuman)。

# 生成样例
<video controls width="640" height="360">
  <source src="assert/Unknown-2.mp4" type="video/mp4">
  您的浏览器不支持 video 标签。
</video>



流水线全景：

```
合成数据 (python -m src.data.synthesis.toy_video)
    │  .npz (frames + audio) + metadata.jsonl
    ▼
数据处理 (python -m src.data.processing.latent_caching)：Wan / Stable Audio VAE、文本编码器预编码
    │  cache/{latents, audio_latents, texts}.pt
    ▼
训练 (train_video.py / train_ddp.sh)：Self-Flow 或 FM，DDP + wandb + FID + 训练中采样
    │  results/<run>/ckpt_*.pt（含完整配置，可复现）
    ▼
推理 (python -m src.sample)：文本 → mp4/gif/wav       评估 (python -m src.evaluation.{toy_video,av})
```

## 0. 环境准备

环境由 [uv](https://docs.astral.sh/uv/) 管理，依赖锁定在根目录 `requirements.txt`（`uv pip freeze` 生成）。以下命令均在项目根目录执行：

```bash
# 安装 uv（已装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建并激活虚拟环境，安装依赖
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

模型权重放 `checkpoints/`（Wan2.2 VAE 必需，音频/T5Gemma 按需）：

```bash
# Wan2.2 VAE 权重（约 2.8GB，已存在则跳过）
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Wan-AI/Wan2.2-TI2V-5B', 'Wan2.2_VAE.pth', local_dir='./checkpoints')"
```

## 1. 数据合成

`src/data/synthesis/toy_video.py` 生成玩具数据集：单个彩色几何体（6 色 × 3 形状 × 3 背景）以恒定速度朝一个方向（4 方向）平移，caption 为 `"a {color} {shape} moving {direction} on a {bg} background"`。每条数据同时带一条编码运动方向的音轨（44.1 kHz 立体声）：右 = 660 Hz 恒定音、左 = 330 Hz 恒定音、上 = 330→660 Hz 上扫、下 = 660→330 Hz 下扫——音频与视频语义强绑定，专门用来检验音视频联合生成的一致性。

```bash
# 17 帧 128x128，3000 条；--width/--height 可出任意 16 倍数分辨率（如 16:9）
python -m src.data.synthesis.toy_video --output-dir data/toy_video --num-videos 3000
python -m src.data.synthesis.toy_video --output-dir data/toy_169 --num-videos 3000 --width 224 --height 128
```

输出格式（也是自定义数据集的接入格式）：每条视频一个 `.npz`（`frames` uint8 (T,H,W,3)，T = 1+4k；可选 `audio` int16 (2,N)，N 为 2048 的倍数），加一个 `metadata.jsonl`（每行 `{"file_name": ..., "text": ...}`）。放自己的数据只要符合这个格式即可。

## 2. 数据处理（latent 预编码）

`src/data/processing/latent_caching.py` 把三个冻结组件的输出一次性算好，训练时完全不跑大模型：

```bash
python -m src.data.processing.latent_caching --data-dir data/toy_video --config configs/default.json
```

输出 `<data-dir>/cache/`：

| 文件 | 内容 |
|---|---|
| `latents.pt` | Wan2.2 VAE 视频 latent，按分辨率分桶：`{"buckets": [(Nᵢ, 48, T′, H′, W′) fp16], "index": [原始条目下标]}`，16× 空间 / 4× 时间压缩 |
| `audio_latents.pt` | Stable Audio VAE 音频 latent，与视频桶对齐的 `buckets` 列表，(Nᵢ, 64, Lᵢ) fp16，按通道全局归一化（存 mean/std 供解码还原）；仅当数据含 `audio` 键 |
| `texts.pt` | 去重 caption 的文本 token 特征 + 空串特征（CFG 用）+ 所用文本编码器配置 |

**真实数据（AVCaps）**：`src/data/processing/avcaps_semantic_split.py` 把 HuggingFace 上的 `TUT-ARG/AVCaps`（`train_videos.zip` + `test_videos.zip` 解压到 `data/avcaps/`）切成上面这个格式。按源视频宽高比映射到三种分辨率（512x384 / 512x288 / 288x512），横屏在 10s / 20s 之间交替、竖屏固定 10s，共五个桶。

AVCaps 给每条片段约 13 条描述（`audio_visual` / `GPT_AV` / `visual` / `audio` 四组），信息量差别很大。`tools/rebuild_texts.py` 只重算文本侧（`texts.pt` 与 manifest，视频/音频 latent 不动），`--select longest` 换成其中最长的那条（含画面信息的三组里平均 24 token，取第一条只有 14），`--captions all` 则保留全部描述、训练时每次抽到该片段就换一条：

```bash
python tools/rebuild_texts.py --data-dir data/avcaps_full/train \
    --config configs/l.json --select longest --captions one
```

换 `--select` 会换掉评估用的 prompt，指标只在同样构建方式的 run 之间可比。

```bash
# 全量训练集（1661 条）+ AVCaps 官方 test 划分（200 条）
python -m src.data.processing.avcaps_semantic_split --avcaps-dir data/avcaps \
    --out-dir data/avcaps_full --train-count 0 --test-source official --workers 32
python -m src.data.processing.latent_caching --data-dir data/avcaps_full/train --config configs/xxl.json
python -m src.data.processing.latent_caching --data-dir data/avcaps_full/test  --config configs/xxl.json
```

`--test-source semantic` 是另一种划分：test 片段也取自 train 池，但每条 test caption 都与某条训练 caption CLIP 相似而不重复，用来在小数据上观察"近邻泛化"。官方划分的源视频与训练集完全不重叠，是报告指标该用的那个。

**分辨率分桶**：数据集可以混合不同分辨率/时长的视频。缓存阶段按（帧形状, 音频长度）分组，每组编码成一个桶；训练时每个 batch 先按桶大小加权抽一个桶、再在桶内采样，保证 batch 内形状一致。Trainer 为每个桶维护独立的 `TokenLayout` / RoPE 坐标，torch.compile 对每种形状各特化一次静态图（`dynamic=False`，桶数有限所以可控）；MFU 按当前桶的序列长度折算。评估采样与 checkpoint 的默认推理分辨率取最大桶（`latent_shape`），全部桶形状记录在 `latent_shapes`。旧的单张量缓存格式仍可直接加载（视作单桶）。

缓存里记录了实际使用的文本编码器，训练时会写进 checkpoint，保证采样端重建一致的组件。

## 3. 模型结构

单流 DiT 处理一条打包序列 `[视频 token | 音频 token | 文本 token]`（`src/models/dit.py`）：

```
视频 latent (48,T′,H′,W′) ─ patchify ─ video_embedder ─┐
音频 latent (64,L)        ────────── audio_embedder ──┼─ [联合序列] ─ N × DiT Block ─ 视频头/音频头
文本 token 特征            ────────── text_embedder ───┘                  ▲
                                                           per-token 时间步 adaLN-Zero
```

- **DiT Block**（MagiHuman 风格）：RMSNorm 预归一化、q/k RMSNorm、3D RoPE 注意力、SwiGLU7 MLP；全时空联合注意力（无独立 temporal 层），时序信息来自 VAE 时间压缩 + RoPE 的 t 轴。
- **位置编码**：3D RoPE 作用于 (t, h, w) 坐标。视频 token 用 latent 网格坐标；音频 token 的 t 换算到视频 latent 帧单位对齐时间轴（h=w=−1）；文本 token 排在时间轴末尾。`x_ids` 第 4 列标注模态（0=视频 1=音频），各模态有独立 embedder 和输出头，通道零填充到统一宽度打包。
- **时间步条件**：MagiHuman 推理 DiT 是无时间步的蒸馏模型，这里嫁接了 Self-Flow 的 per-token adaLN-Zero（每个 token 可有独立 t，Self-Flow 掩码训练必需）与自蒸馏 projector。`model.adaln` 可选 `per_block`（默认，每个 block 一个 hidden→6·hidden 调制 Linear，DiT 经典结构，XL 下占全模型约 1/3 参数）或 `single`（PixArt-α 方案：全模型共享一个调制头，每个 block 只加一个可学习偏置表；XL 参数 724M→498M，收敛轨迹实测与 per_block 基本重合）。
- **时间维 patchify**：`model.patch_size_t`（默认 1）把相邻 latent 帧合并为一个 token（叠加在 Wan VAE 已有的 4x 时间压缩之上）。T 不整除时序列尾部零填充、解包时裁掉（pad token 参与注意力与损失）。XL 下 patch_size_t=2 让序列 3915→2155，单步 GPU 时间近乎减半；快速运动内容的时序细节可能受损，需按数据验证。与 `adaln` 一样，不同取值的 checkpoint 不互通。
- **mHC 超连接**（Manifold-Constrained Hyper-Connections，DeepSeek，arXiv:2512.24880）：`model.mhc`（默认 0 关闭，设 n>1 启用 n 流版本）。残差状态扩为 n 份 (B, L, n, C)，每个子层连接由输入动态生成三组门控：H_pre（sigmoid，n 流加权读出进子层）、H_post（2·sigmoid，子层输出写回各流）、H_res（Sinkhorn-Knopp 投影到双随机矩阵的 n×n 流混合，凸组合保范数）；初始化使整个连接与标准残差逐位等价（实测差 ~1e-6）。实现做了两处代数化简以贴合 torch.compile：norm 的 affine 吸收进无偏置的 phi；`phi(RMSNorm(x)) ≡ phi(x)·inv_rms(x)`，先 GEMM 再对 (n²+2n) 维输出做标量缩放，避免物化归一化后的 n·C 宽张量（原实现最大的开销）；流混合写成融合乘加而非 M=K=n 的微型批量 GEMM。XL 实测单步 89.6 ms（mhc=0）→ 137 ms（mhc=2）/ 200 ms（mhc=4；`train.compile_mode="max-autotune-no-cudagraphs"` 下 172 ms），剩余开销主体是 n 倍残差流的显存带宽（FLOPs 增量极小，MFU 数值会相应回落；论文靠定制融合 kernel 才压到 ~7%）。2000 步过拟合冒烟：mhc=4 收敛正常（FID 9.3 / FVD 1.2 / FAD 0.11），与基线的质量差异需更长训练验证。checkpoint 与 mhc=0 不互通。
- **冻结组件**（全部可在配置里替换，见第 6 节）：Wan2.2 3D 因果 VAE（视频）、Stable Audio Open 1.0 VAE（音频，Oobleck 1D）、CLIP 或 T5Gemma（文本）。
- 结构由配置声明：`model: {type, hidden_size, depth, num_heads, patch_size, patch_size_t, adaln, attention}`，默认 768×12（≈130M）。`attention` 可选 `flash`（默认，精确 softmax 注意力，eager 路径走 FlashAttention-4（`flash-attn-4` 包，CuTe DSL JIT，Blackwell/Hopper），包不可用或处于 torch.compile 内时自动回退 SDPA，fp32 输入自动以 bf16 计算）、`softmax`（SDPA 自动选后端）或 `linear`（核化线性注意力 φ(x)=elu(x)+1，O(L)，约 8k token 以上序列才比 softmax 快）；三者参数结构相同，flash/softmax 数学等价可互换权重，linear 训练出的权重不可与前两者混用。checkpoint 记录所用类型，推理端自动一致。

forward 保留 Self-Flow 的兼容约定（内部 `timesteps = 1 − timesteps`、输出取负），训练拟合 `MSE(−model(x_t, 1−t, text), x1 − x0)`，其中 `x_t = t·x1 + (1−t)·x0`。

## 4. 模型训练

两种训练目标（`src/training/objectives.py`，`train.objective` 选择）：

- **selfflow**（默认）：flow matching + Dual-Timestep Scheduling（每个样本独立抽 `t,s~p(t)`，`mask_ratio` 比例的 token 使用 `s`、其余使用 `t`）+ EMA teacher 自蒸馏。teacher 用同一噪声在统一的 cleaner timestep 前向，student 浅层特征经 projector 与 teacher 深层原始特征做余弦对齐。代码采用 `t=1` 为 clean 的反向变量，因此论文的 `min(t,s)` 在实现中对应 `max(t,s)`。
- **fm**：纯 conditional flow matching 基线。

缓存含音频时自动变成音视频联合扩散：两个模态在同一序列中一起加噪，损失 `loss = fm_video + audio_weight * fm_audio`（+ Self-Flow 蒸馏项）。

```bash
# 单卡（配置驱动，任何键可用 --set 覆盖；常用项有快捷参数）
python train_video.py --data-dir data/toy_video --results-dir results/video
python train_video.py --data-dir data/toy_video --config configs/v2.json \
    --set train.objective=fm --set model.depth=24

# 多卡 DDP（常用项走环境变量，其余 --set 透传；batch_size 是单卡值）
GPUS=4 STEPS=10000 BATCH_SIZE=32 bash train_ddp.sh
WANDB_MODE=offline GPUS=2 OBJECTIVE=fm bash train_ddp.sh --set train.mask_ratio=0.25
```

训练期间（`src/training/trainer.py`，均可在 `train` 段配置）：

- **wandb**：每步记录 `loss / lr` 及按训练目标命名的分量——self flow 为 `selfflow_video / selfflow_audio / distill`，flow matching 为 `fm_video / fm_audio`；另记 `mfu`（估算 FLOPs ÷ 设备 bf16 峰值，未识别的 GPU 用 `train.peak_tflops` 指定）、`tflops_per_gpu`、累计 `train_time_s`（结束时 summary 写 `total_train_time_s`）；run config 含 `model_params`。未登录自动转 offline（`--no-wandb` 关闭）。
- **日志文件**：所有脚本的 logging 输出（`src/log.py`）除终端外同时写入根目录 `logs/<时间戳>_<入口脚本>[_rankN].log`，DDP 非 0 号 rank 只记 WARNING 以上。
- **优化器**：`train.optimizer` 可选 `adamw`（默认，fused AdamW）或 `muon`（参考 [DiT-Muon](https://github.com/lavinal712/DiT-Muon)：transformer block 内的 2D 权重用 Muon——momentum + Newton-Schulz 正交化，`train.muon_lr` 默认 1e-3——嵌入层/输出头/norm/bias 走内部 aux AdamW（`train.lr`）；同形状矩阵堆叠做批量 NS 迭代，XL 下 step 约 47 ms）。warmup/cosine 调度按比例作用于两组各自的峰值 lr。示例配置 `configs/xl_muon.json`。
- **训练速度**：默认开启 `train.compile`（torch.compile 学生前向反向 + EMA teacher 前向，首步 JIT 约 1 分钟）、fused AdamW、foreach EMA、Self-Flow teacher 在特征层提前退出、DDP bf16 梯度压缩（`train.grad_compress_bf16`）；数据缓存 ≤4GB 时整体驻留 GPU（消掉每步的 pageable H2D 同步）。XL（724M，seq≈3.9k，bs 8）单步 683→245 ms（GB300 实测，compiled 与 eager loss 逐位一致）；再叠加结构项 `adaln=single` + `patch_size_t=2`（`configs/xl_fast.json`，498M）单步 115 ms，同等 loss 的墙钟时间约 1.9x 加速。经实测否定的方向：FP8（torchao，瘦 GEMM 上量化开销反而净慢 21%）、torch.compile max-autotune / cudagraphs（≤2%）、线性注意力（4k 序列无优势）。基准脚本 `scripts/bench_speedups.py`（每进程只测一个变体，同进程多次 compile 会互相污染）、`scripts/profile_step.py`。排查 compile 问题时用 `--set train.compile=false` 关闭。
- **生成质量指标**：每 `train.fid.every` 步采样 `num_videos` 条视频。按 Self-Flow 论文默认使用 EMA 权重、`cfg_scale=1`（无 guidance）；定性 `sample`/`test` 默认使用 EMA + CFG 5，两者配置彼此独立。传了 `--eval-data-dir` 时打分对象是这份留出数据（指标名加 `test_` 前缀），否则退回训练集。评测片段和初始噪声都只抽一次、之后每轮复用——每轮重抽会把片段间的方差混进曲线，指标就会因为与权重无关的原因跳动；生成侧固定 `num_videos` 条（贵的那一侧），真实侧默认取留出集全部片段（`train.fid.num_real`）。留出集的每个桶都在自己的形状下生成，特征跨桶合并，所以分数覆盖整个划分而不只是最大那个桶。记录：
  - `fid`（主指标）：逐视频 FID 的平均——每条生成视频的帧单独构成一个高斯，与真实帧高斯算 FID 后取均值（低秩恒等式实现，每条视频只需 T×T 特征分解）；`fid_pooled` 为全帧合并版作参考。
  - `fvd`：视频级 Fréchet 距离，特征来自 Kinetics-400 预训练的 r3d_18（非官方 I3D 版 FVD，但构造相同）。
  - `clip_score`：均匀抽 8 帧与 caption 的 CLIP（ViT-B/32）余弦相似度均值 ×100，衡量文本一致性。
  - `fad`（有音频时）：Fréchet Audio Distance，特征为 VGGish（AudioSet 预训练，每 0.96s 一个 128 维嵌入，立体声取均值转单声道，内部重采样到 16kHz）。

  另有 `fid_paired`：每条生成视频只跟它 caption 指定的那条真实片段比。两侧内容相同、协方差形状匹配，所以它能趋近 0；`fid` 拿单条视频的帧去比整个真实集合，有一个协方差不匹配的下界，数值上会长期迟钝。

  两侧都过 VAE 解码以排除重建差距；几十条片段的规模下都是趋势指标，报告用的数值走 `tools/eval_test_set.py`（见第 7 节）跑完整划分。
- **训练中采样**：每 `train.sample.every` 步随机取训练 caption 生成视频（含音频，默认原始权重，`train.sample.use_ema` 可切换）存 `results_dir/samples/step*.mp4` 并上传 wandb；有留出数据时另每 `train.test.every` 步随机取一条留出片段，把真实片段和同 prompt 的生成结果成对存到 `results_dir/test_samples/` 并上传 wandb（单条只用来肉眼看，分数由上面的留出集指标给）。
- checkpoint 存模型 / EMA / 优化器 / 完整解析后配置 / latent 形状 / 音频元信息。

## 5. 推理

`src/sample.py` 从 checkpoint 自动重建全部匹配组件（文本编码器、视频 VAE、音频 VAE），无需再传配置：

```bash
python -m src.sample --ckpt results/video/ckpt_last.pt \
    --prompts "a red circle moving right on a white background"

# 分辨率覆盖（16 的倍数；靠 RoPE 外推，训练分辨率之外质量会衰减）
python -m src.sample --ckpt results/video/ckpt_last.pt --width 224 --height 128 \
    --prompts "..."
```

每条 prompt 输出 `.mp4`（H.264，联合 AV 模型自动混入 AAC 音轨）、`.gif`、帧条 `.png`（和 `.wav`）。未显式传参时，CLI 从 checkpoint 的定性采样配置读取 mode、steps、sampleshift、CFG 和权重类型；论文对 WAN2.2 视频采用 50-step ODE、sampleshift 15、EMA 权重和 CFG 5。命令行参数仍可逐项覆盖，`--no-ema` 可强制使用原始权重。

## 6. 配置系统

模型结构、组件和完整训练配方统一在 `configs/*.json`：

```jsonc
{
  "model":        { "type": "video_dit", "hidden_size": 768, "depth": 12, "num_heads": 12, "patch_size": 1, "patch_size_t": 1, "adaln": "per_block", "attention": "flash", "mhc": 0 },  // adaln: per_block | single; attention: flash | softmax | linear; mhc: 0 关闭 | n>1 流数
  "text_encoder": { "type": "clip",  "name_or_path": "openai/clip-vit-base-patch32", "text_len": 16 },   // 或 type=t5gemma
  "video_vae":    { "type": "wan2.2", "path": "checkpoints/Wan2.2_VAE.pth" },
  "audio_vae":    { "type": "stable_audio", "path": "checkpoints/stable-audio-open-1.0" },
  "train":        { "objective": "selfflow", "steps": 6000, "lr": 1e-4, "...": "见 src/config.py TRAIN_DEFAULTS" }
}
```

- **可插拔组件**：每个组件按 `type` 从注册表构建（`src/models/registry.py`）；新增一种 VAE / 文本编码器 / DiT 只需写实现类 + `register_*` 注册工厂，调用方零改动。
- **覆盖机制**：`train` 段缺省项落到 `TRAIN_DEFAULTS`；任何键可用 `--set a.b.c=value` 覆盖（值按 JSON 解析）。
- **可复现**：checkpoint 携带解析后的完整配置，采样/评估端直接重建同样的组件。
- 现成配置：`default.json`（基线）、`v2.json`（改进配方：cosine+warmup+lognorm+蒸馏 warmup）、`t5gemma.json`（换 T5Gemma 文本编码器）、`xl.json`（1280×24）、`xl_fast.json`（XL + adaLN-single + 时间 patchify，速度约 2.1x）、`xl_muon.json`（XL + Muon 优化器）。

## 7. 评估

**留出集完整评测（真实数据）**：`tools/eval_test_set.py` 把一个 checkpoint 在整个缓存划分上打分，构造与训练中那套一致，但跑完每一条片段，并给出逐桶分解——报告数值用它，训练中的指标只看趋势。

```bash
python tools/eval_test_set.py --ckpt results/<run>/ckpt_last.pt \
    --data-dir data/avcaps_full/test --out results/<run>/test_metrics.json
# guidance 扫描：真实侧只解码/提特征一遍，多个 cfg 共用
python tools/eval_test_set.py --ckpt results/<run>/ckpt_last.pt \
    --data-dir data/avcaps_full/test --cfg 1.0 2.0 3.0 5.0 --num-videos 40
# 可选：--weights ema、--num-steps、--mode/--shift（默认取 checkpoint 里的采样配置）
```

`--cfg` 收多个值时按 cfg 逐行打表。训练中的指标固定在 `cfg=1.0` 下测，泛化训练的模型在 cfg=1 出来的接近条件均值、偏糊，FID 会被系统性抬高，所以报告前先扫一遍 guidance。

生成音频 latent 用 checkpoint 记录的训练集归一化还原（模型学的就是那个空间），真实片段用该划分自己的统计量。

**玩具数据集的两个自动评测脚本**，都基于像素/频谱启发式判别器（在真实数据和 VAE 重建上 100% 准确）：

- **`src/evaluation/toy_video.py`** —— 提示词保真度：对全部 216 种组合逐条生成，判别颜色（调色板最近邻）、形状（bbox 四角占有率）、背景、运动方向（首末帧质心位移），报告各属性准确率与全对率。

```bash
python -m src.evaluation.toy_video --ckpts selfflow=results/video/ckpt_last.pt \
    fm=results/video_fm/ckpt_last.pt --seeds 0 1
```

- **`src/evaluation/av.py`** —— 音视频一致性（联合 AV checkpoint）：视频方向照上；音频方向用首末 0.4 s 窗口的主频判断（恒定 660/330 Hz vs 上扫/下扫），报告视频准确率、音频准确率、AV 一致率。

```bash
python -m src.evaluation.av --ckpt results/av_sf/ckpt_last.pt
```

对比 Self-Flow 与 FM 两个训练目标时，可用同一配置各训一个 checkpoint 后传入 `--ckpts label=path` 一起评测；论文设定使用 EMA 权重，必要时可显式切换 raw 权重做诊断。

## 8. 代码结构

```
mini-agnes-video/
├── requirements.txt               # uv pip freeze 生成的锁定依赖
├── configs/                       # default.json / v2.json / t5gemma.json
├── checkpoints/                   # Wan2.2_VAE.pth / stable-audio-open-1.0/ / t5gemma-2b-2b-ul2-it/
├── logs/                          # 各入口脚本的运行日志（自动生成）
├── train_video.py                 # 训练入口 -> src/training/trainer.py（--set 覆盖任意配置）
├── train_ddp.sh                   # torchrun DDP 启动脚本
├── tools/                         # 一次性诊断/评测脚本
│   ├── eval_test_set.py           # 整个划分上的 FID/FVD/CLIP/FAD + 逐桶分解
│   ├── probe_train_test_gap.py    # 训练/留出集损失对照（按模态、按 t，含 blind 参照）
│   ├── probe_sampler_ab.py        # 同一 checkpoint 下的采样器对照
│   ├── probe_text_dependence.py   # 真/换/空 caption 下的留出损失 + CFG 条件差向量
│   ├── probe_conditioning.py      # 数据侧：容量核算、caption 拥挤度、caption→latent R²
│   ├── rebuild_texts.py           # 只重算文本缓存（换 caption 选择策略 / text_len）
│   ├── probe_latent_recon.py      # VAE 往返重建的指标下界
│   └── probe_loss_by_bucket.py    # 损失按桶 / 按 t 的分布
└── src/
    ├── config.py                  # 配置加载 + TRAIN_DEFAULTS + --set 点号覆盖
    ├── log.py                     # logging 统一配置（DDP 下非 rank-0 只输出 WARNING+）
    ├── inference.py               # sample_latents / sample_videos（训练与采样共用）
    ├── sample.py                  # 采样 CLI：python -m src.sample
    ├── media.py                   # gif / 帧条 / wav / mp4 输出
    ├── sampling.py                # denoise_loop（照搬 Self-Flow）
    ├── utils.py                   # TokenLayout：token 序列打包/解包/位置 id
    ├── data/
    │   ├── synthesis/
    │   │   └── toy_video.py       # 合成数据：python -m src.data.synthesis.toy_video
    │   ├── processing/
    │   │   ├── latent_caching.py  # 数据处理：python -m src.data.processing.latent_caching
    │   │   ├── avcaps_subset.py   # AVCaps 子集：解码/裁剪/分辨率分桶的底层函数
    │   │   └── avcaps_semantic_split.py  # AVCaps train/test 划分（全量 + 官方 test）
    │   └── latent_cache.py        # 训练用缓存加载 + 随机 batch
    ├── evaluation/
    │   ├── toy_video.py           # 提示词保真评测：python -m src.evaluation.toy_video
    │   └── av.py                  # 音视频一致性评测：python -m src.evaluation.av
    ├── models/
    │   ├── registry.py            # 组件注册表（type -> 工厂）
    │   ├── dit.py                 # 文本条件 Video DiT（参考 MagiHuman + Self-Flow 条件机制）
    │   ├── video_vae.py           # Wan2.2 3D VAE（照搬 daVinci-MagiHuman）
    │   ├── audio_vae.py           # Stable Audio Open VAE（照搬 daVinci-MagiHuman sa_audio）
    │   └── text_encoders.py       # 冻结文本编码器：CLIP + T5Gemma（照搬 MagiHuman）
    └── training/
        ├── trainer.py             # Trainer：DDP / EMA / LR 调度 / wandb / FID / 采样 / ckpt
        ├── objectives.py          # FlowMatching / SelfFlow 训练目标
        ├── fid.py                 # InceptionV3 帧级 FID
        └── distributed.py         # torchrun 进程组
```
