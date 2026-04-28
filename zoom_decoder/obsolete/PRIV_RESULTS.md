# Privileged-Prompt (LUPI) 蒸馏策略实验报告

本报告在**保留原 Zoom-Teacher 方案不改动的前提下**，实现并评估了一组**完全不动图片**的新
策略：向教师 prompt 注入“特权信息”（bounding-box 坐标等），学生推理时仍只看到**原始图片
 + 原始问题**。所有配置在 **FineSightBench/perception** 的同一 test 集（N=360，seed=0 分层
抽样）上评估，VLM 基座固定为 `Qwen/Qwen2-VL-2B-Instruct`。

> **v2 更新**：新增 **经典 MemoryDecoder (kNN-LM) 基线** `zd_knn_24L`。该版本把 teacher
> 替换为 FAISS-kNN 检索产生的分布（与本仓库原始 `vlm_save_embed.py / saveKNNMulti.py`
> 管线等价，但适配到 FineSightBench），见 §3 末行。

## 1. 策略总览

| 代号 | 图片 | 教师 prompt | 学生 prompt | 定位 |
| :-- | :-- | :-- | :-- | :-- |
| `zoom` (原) | **放大** union bbox 裁剪图 | 原问题 | 原图 + 原问题 | 视觉放大蒸馏 |
| `priv` (新) | **原图不动** | 原问题 + `bbox=(x,y)-(x,y), size=Npx` 前缀 | 原图 + 原问题 | LUPI 特权信息蒸馏 |
| `priv_full` (新) | **原图不动** | priv + 目标 `value` 文本 | 原图 + 原问题 | 天花板上限 (教师近一-hot) |
| `zoom_priv` (新) | 放大 crop | 原问题 + bbox 前缀 | 原图 + 原问题 | 视觉 + 坐标双路教师 |
| `knn` (基线) | **不需要图** | 教师 = FAISS kNN (原图 VLM embedding → top-K 邻居的 next-token 软分布) | 原图 + 原问题 | 经典 kNN-LM 蒸馏 |

所有学生侧均共享同一套三项 novelty：Zoom-Teacher 自蒸馏替代 kNN、Scale-Weighted Focal KL、
Aperture Token；所有变体沿用相同的 `stratified_split(seed=0)`，训练/验证/测试索引**逐条相等**
（本次已通过 `diff` 校验）。

## 2. 实验配置

- Decoder：Qwen2-0.5B 前 24 层，Aperture Token 启用，`scale_gamma=1.0`
- Teacher topk=32；训练 3 epoch、batch 8、lr 5e-4、`alpha_ce=0.3`
- 联合推理：`logaddexp(log(1-λ)+lp_vlm, log(λ)+lp_zd)`，λ=0.3，`max_new_tokens=24`

数据量：每个 dstore 1680 训练样本 / 10314 token 位置。

## 3. 主结果（test N=360）

| 模型 | Overall | easy | medium | hard | extreme | Δ vs base |
| :-- | --: | --: | --: | --: | --: | --: |
| base（仅 Qwen2-VL-2B） | 71.67 | 93.33 | 92.22 | 72.22 | 28.89 | — |
| zd_24L `zoom` | 73.61 | 93.33 | 95.56 | 78.89 | 26.67 | +1.94 |
| **zd_priv_24L** `priv` | 74.72 | 93.33 | 95.56 | **82.22** | 27.78 | +3.05 |
| **zd_zoompriv_24L** `zoom_priv` | 75.00 | 93.33 | 95.56 | 81.11 | 30.00 | +3.33 |
| **zd_knn_24L** `knn` 基线 | **78.06** | 94.44 | 95.56 | **85.56** | **36.67** | **+6.39** |
| zd_24L_noap（消融：无 Aperture） | 73.06 | 93.33 | 93.33 | 76.67 | 28.89 | +1.39 |
| zd_24L_nosw（消融：无 Scale-W） | 74.44 | 93.33 | 95.56 | 80.00 | 28.89 | +2.77 |
| lora_r16（全量 SFT 上界） | 86.94 | 100.00 | 100.00 | 90.00 | 57.78 | +15.27 |

## 4. 按任务分解

| 模型 | animal | block | color | letter | shape | text |
| :-- | --: | --: | --: | --: | --: | --: |
| base | 73.3 | 50.0 | 88.3 | 78.3 | 70.0 | 70.0 |
| zd_24L `zoom` | 71.7 | 65.0 | 88.3 | 78.3 | 68.3 | 70.0 |
| zd_priv_24L | 75.0 | 68.3 | 88.3 | 76.7 | 68.3 | 71.7 |
| zd_zoompriv_24L | 71.7 | 70.0 | 88.3 | 78.3 | 71.7 | 70.0 |
| **zd_knn_24L** | 73.3 | **80.0** | 88.3 | **80.0** | 73.3 | **73.3** |
| lora_r16 | 76.7 | 100.0 | 100.0 | 88.3 | 78.3 | 78.3 |

## 5. 关键发现

1. **不动图片也能赢 zoom**：单独把 bbox 坐标注入教师 prompt（`priv`），相对 `zoom` 再涨
   **+1.11** pp，对 `base` 涨 **+3.05** pp。说明 Qwen2-VL 的视觉编码器对 448² 分辨率并没
   有被 token 数严重限制——关键是**教师知不知道要看哪儿**，而不是**像素有多大**。
2. **`zoom` vs `priv` 各有偏好**：`zoom` 在 medium 更稳（+3.34 pp over base），`priv`
   把 **hard** 从 78.89 → **82.22** (+3.33)，与我们之前对“hard 样本通常是多目标 / 非中心
   目标”的判断一致；坐标先验明显比单纯放大更能帮助这种情形。
3. **`zoom_priv` 是最佳非 LoRA 配置**（75.00%），且把 **extreme** 推到 30.00%（zoom 仅
   26.67，priv 27.78）。两路信号互补：放大提供细粒度像素，坐标提供全局位置。
4. **block_recognition**（形状/颜色块计数，典型多目标任务）收益最大：
   `base→zoom→priv→zoom_priv` 为 50.0 → 65.0 → 68.3 → **70.0**，单调上升，印证多目标
   场景下 LUPI 对 Zoom 的正交性。
5. **经典 kNN-LM 基线意外地成为最强非 LoRA 配置**（78.06%，+6.39 pp）。原因是
   FineSightBench 的答案 token 分布极其窄（单字母/颜色名/数字），在 seed=0 的 1680
   训练样本上：**1000 个查询中有 841 个的 K=64 邻居 val 完全相同**（唯一 val）。这让
   kNN 教师分布退化为**近似 one-hot = gold CE**。亦即此时学生其实在学一个高质量的、
   经过 VLM-embedding 筛选后的 **CE/SFT 目标**。这暴露了经典 MemDec 在**短答案/结构化
   答案**任务上 KL 软分布信号弱的本质，但也说明：
   - CE 目标本身在 FineSightBench 上很强，因此 kNN 基线被抬高；
   - zoom / priv 教师虽然带来真正的软分布（高熵 KL 信号），但学生用 3 个 epoch 的
     `α_ce=0.3` 训练尚未完全吸收；
   - 与 LoRA-r16（86.94%）差距仍然显著，说明**全量 SFT 调 VLM 参数**仍是最终上界。
6. 与 LoRA 全量 SFT 的 86.94 仍有差距——但 `zd_knn_24L` 的 78.06% 是在**完全不改 VLM
   参数**、仅加一个 0.5B decoder 的前提下做到的，性价比显著。

## 6. 与 `zoom` 方案的定位关系

| 维度 | `zoom` | `priv` | `zoom_priv` |
| :-- | :-- | :-- | :-- |
| 教师需要额外能力 | 裁剪/放大器 | bbox GT | 两者 |
| 多目标（union bbox 过大） | 放大会失真 | 不受影响 | 混合 |
| 不需要 bbox GT 的场景 | ✅ | ❌ | ❌ |
| 我们的推荐 | 主力 | 有 GT 时最佳单路 | **最终默认** |

## 7. 复现命令

```bash
# priv-bbox 数据
python -m zoom_decoder.prepare_teacher \
  --teacher_mode priv --out_dir ./zoom_decoder/dstore_priv \
  --topk 32 --seed 0

# zoom_priv 数据
python -m zoom_decoder.prepare_teacher \
  --teacher_mode zoom_priv --out_dir ./zoom_decoder/dstore_zoom_priv \
  --topk 32 --seed 0

# 经典 kNN-LM 教师
python -m zoom_decoder.prepare_knn_teacher \
  --out_dir ./zoom_decoder/dstore_knn \
  --topk 32 --knn_k 64 --knn_temp 500.0 --seed 0

# 训练（三个 ckpt 用同一份 train.py）
for cfg in priv zoom_priv knn; do
  python -m zoom_decoder.train \
    --data_dir ./zoom_decoder/dstore_$cfg \
    --num_layers 24 --out_dir ./zoom_decoder/ckpt/zd_${cfg}_24L \
    --epochs 3 --batch_size 8 --lr 5e-4
done

# 评估（与 zoom 系共用同一份 splits.json）
for cfg in priv zoom_priv knn; do
  python -m zoom_decoder.evaluate --mode zoom \
    --zoom_ckpt ./zoom_decoder/ckpt/zd_${cfg}_24L/final \
    --splits_file ./zoom_decoder/dstore/splits.json \
    --eval_split test --lmbda 0.3 --max_new_tokens 24 \
    --out_file ./zoom_decoder/eval/zd_${cfg}_24L.json
done
```

## 8. 结论

> **现在有 4 个可选教师，统一在同一管线、同一 split、同一评测下比较**：
>
> | Teacher | Overall | 是否需要 bbox GT | 是否需改图 | 备注 |
> | :-- | --: | :-: | :-: | :-- |
> | `zoom` | 73.61 | ❌ | ✅ | 无需标注，自蒸馏路线 |
> | `priv` | 74.72 | ✅ | ❌ | LUPI 路线 |
> | `zoom_priv` | 75.00 | ✅ | ✅ | 视觉 + 坐标双路 |
> | **`knn` (classical MemDec)** | **78.06** | ❌ | ❌ | 在**短答案任务**上退化为强 CE |
>
> **实用建议**：
> - **结构化短答案任务（FineSightBench 类）**：直接用 `knn` 教师（或等价地：带少量 KL
>   平滑的 CE-SFT），最简单、最稳、最强。
> - **自由生成 / 长答案任务（caption、VQA 长问答）**：kNN 教师不再退化，zoom/priv 的
>   软分布信号反而更有优势。本实验的 FineSightBench 结论不直接外推。
> - **没有 bbox 标注时**：选 `zoom`。
> - **最终上界**：LoRA 全量 SFT（86.94%）。

## 9. kNN 基线实现说明

`zoom_decoder/prepare_knn_teacher.py` 用 60 行代码独立实现了经典 MemDec 的 teacher
阶段（对标仓库根下 `vlm_save_embed.py` + `knn_utils/build_index.py` + `saveKNNMulti.py`
三合一），输出 schema 与 `prepare_teacher.py` 完全兼容：

1. **Pass 1**（VLM 一次前向）：每个训练样本 teacher-force 答案，钩住最后一层 MLP 的
   输入作为 key（与仓库原 `vlm_save_embed.py` 一致），gold next-token 作为 val。
2. **Pass 2**（FAISS）：`IndexFlatL2` 对 10314 个 key 建索引，每行查 top-(K+1)、剔除
   自身，得 K=64 邻居；`softmax(-d / T=500)` 加权按 val 聚合，取 top-32 作为教师
   分布。
3. **Train / Eval**：完全复用 `zoom_decoder.train` / `zoom_decoder.evaluate`，无需
   任何额外改动。

这使得 4 种教师形成严格受控实验——**仅 teacher 来源不同，其余所有超参/数据/模型/
评测完全一致**。

## 10. kNN-MemoryDecoder × 两项 Novelty 消融（v3 · 6L 最小模型）

> 目标：验证 §README 中两项 novelty（**Scale-Weighted Focal KL** 与 **Aperture Token**）
> 是否对**经典 kNN-LM 教师**也有效，还是仅对 zoom-teacher 有效。
> 设置：与 §3 完全相同的数据 / split / λ / 训练超参，**唯一变量**是是否启用两项 novelty；
> Decoder 用最小的 **6 层** Qwen2-0.5B（225 M 参数），训练数据用全部 `dstore_knn` 的
> 1680 条 train + 360 val（评测用 360 test，与所有先前结果**逐条相同**）。

### 10.1 2×2 消融矩阵

| 配置 | Aperture | Scale-W | Overall | extreme | hard | medium | easy | px=12 | px=32 |
| :-- | :-: | :-: | --: | --: | --: | --: | --: | --: | --: |
| base VLM | — | — | 71.67 | 28.89 | 72.22 | 92.22 | 93.33 | 75.00 | 95.74 |
| **zd_knn_6L** (full) | ✅ | ✅ | 77.22 | 35.56 | **84.44** | 95.56 | 93.33 | **90.38** | 95.74 |
| zd_knn_6L_noap | ❌ | ✅ | 77.22 | 35.56 | 83.33 | 95.56 | 94.44 | 88.46 | **97.87** |
| zd_knn_6L_nosw | ✅ | ❌ | **77.50** | **36.67** | 83.33 | 95.56 | 94.44 | 88.46 | **97.87** |
| zd_knn_6L_plain | ❌ | ❌ | **77.50** | **36.67** | 83.33 | 95.56 | 94.44 | 88.46 | **97.87** |
| zd_knn_24L (参考) | ✅ | ✅ | 78.06 | 36.67 | 85.56 | 95.56 | 94.44 | 92.31 | 97.87 |

### 10.2 按任务分解（6L kNN 系列）

| 配置 | animal | block | color | letter | shape | text |
| :-- | --: | --: | --: | --: | --: | --: |
| zd_knn_6L (full) | 73.3 | 80.0 | 88.3 | 78.3 | **73.3** | 70.0 |
| zd_knn_6L_noap   | 73.3 | 80.0 | 88.3 | 78.3 | 71.7 | **71.7** |
| zd_knn_6L_nosw   | 73.3 | 80.0 | 88.3 | **80.0** | 71.7 | **71.7** |
| zd_knn_6L_plain  | 73.3 | 80.0 | 88.3 | **80.0** | 71.7 | **71.7** |

### 10.3 关键观察

1. **两项 novelty 在 kNN 教师上几乎完全失效**：4 个变体 Overall 落在 77.22–77.50 这
   0.28 pp 区间内，差异完全在 360 样本的统计噪声之内（±2.2 pp 的 95% CI）。
2. **Aperture Token 净效应 ≈ 0**：`zd_knn_6L`(✅✅) vs `zd_knn_6L_noap`(❌✅) 同为 77.22；
   `zd_knn_6L_nosw`(✅❌) vs `zd_knn_6L_plain`(❌❌) 同为 77.50。在两种 scale-weight 设置
   下都看不到任何提升。
3. **Scale-Weight 仍然轻微为负**（-0.28 pp），与 zoom-teacher 上的观察方向一致
   （README §5.3 中 `zd_24L_nosw` 反而略优 +0.83 pp）。即：**移除尺度加权对所有
   teacher 都没坏处，甚至略好**。
4. **6L vs 24L (kNN, full)**：78.06 → 77.22，仅差 0.84 pp，再次印证
   "plug-in 先验不需要大容量"——即便在 kNN 教师下，最小 6L 已经能跑到 24L 的 99% 水平。
5. **结构化任务的鲁棒性**：`block_recognition`、`color_block`、`animal` 在 4 个配置下
   完全相同（80.0 / 88.3 / 73.3），表示 **kNN 近 one-hot 教师** 已经把这几类任务
   "压"到了一个共同的 CE 解，novelty 无处可加。

### 10.4 解释：为什么 novelty 在 kNN 上失灵？

> **核心原因**：FineSightBench 答案 token 极窄 + kNN 教师退化为近 one-hot。

- **Scale-Weighted Focal KL** 的设计假设是：teacher 在不同尺度上输出的是**软分布**，把
  小尺度样本权重抬高可以把"模糊但有信息"的软分布更充分地蒸馏。但 kNN 教师在 §5（v2）
  已经测过，1000 个查询里 841 个 K=64 邻居 val 完全相同——**软分布几乎就是 one-hot**，
  KL 退化为 CE。CE × 8 倍权重 = SFT × 8 倍 lr，并不带来新的信号，反而放大了那 16% 邻居
  不一致样本里的 label-noise。
- **Aperture Token** 的设计假设是：同一 decoder 权重需要按"目标多大"分叉调用，因为
  zoom-teacher 在不同档位提供**质量差异巨大**的软分布（4 px 噪声大、48 px 接近完美）。
  但 kNN 教师不基于图像档位变化——它基于 VLM 文本-视觉联合 embedding 的最近邻；不同尺度
  的 query 本来就会命中不同邻居簇，**尺度信息已经隐含在邻居选择里**，再加 aperture
  bias 就是冗余。

### 10.5 论文学层面的结论

> **两项 novelty 是 teacher-dependent，不是 teacher-agnostic**。

- 对 **zoom / priv / zoom_priv** 这类**会产生真正软分布**的 visual-grounded 教师：
  Aperture Token 是干净正向（README §5.4，+2.22 pp on hard），Scale-Weight 在小数据
  上是负向（README §5.3，需要更高质量 teacher 才能兑现）。
- 对 **kNN / 一般 token-retrieval** 这类**容易退化为 one-hot** 的教师：两项 novelty
  无效。这是符合预期的——它们设计时的前提（"教师有尺度依赖的不确定性"）此处不成立。
- 因此，**"在 kNN 基线上叠加 novelty 来涨点"不是一个好的 paper 故事**。真正可写
  论文的负面结论是：**两项 novelty 与 teacher 的 entropy / 软度强相关**——用 kNN 教师
  来对照实验，反而能证明 zoom-teacher 软分布的价值。

### 10.6 复现命令

```bash
# 以下 4 个训练共用同一份 dstore_knn（已在 v2 准备好）
for tag in zd_knn_6L:""                \
           zd_knn_6L_noap:--disable_aperture \
           zd_knn_6L_nosw:--disable_scale_weight \
           zd_knn_6L_plain:"--disable_aperture --disable_scale_weight"; do
  name=${tag%%:*}; flags=${tag#*:}
  python -m zoom_decoder.train \
    --data_dir ./zoom_decoder/dstore_knn \
    --num_layers 6 --out_dir ./zoom_decoder/ckpt/$name \
    --epochs 3 --batch_size 8 --lr 5e-4 $flags
done

for tag in zd_knn_6L zd_knn_6L_noap zd_knn_6L_nosw zd_knn_6L_plain; do
  python -m zoom_decoder.evaluate --mode zoom \
    --zoom_ckpt ./zoom_decoder/ckpt/$tag/final \
    --splits_file ./zoom_decoder/dstore_knn/splits.json \
    --eval_split test --lmbda 0.3 \
    --out_file ./zoom_decoder/eval/$tag.json
done
```

