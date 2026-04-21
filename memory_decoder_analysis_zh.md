# MemoryDecoder 仓库分析与 Decoder 实现原理解读

## 1. 这是什么仓库

这个仓库实现的是论文 **Memory Decoder: A Pretrained, Plug-and-Play Memory for Large Language Models** 的一个工程版本。它的核心目标不是再训练一个更大的基础模型，也不是在推理时在线检索文档，而是：

- 先用外部 kNN 检索构造出“领域记忆分布”作为监督信号；
- 再训练一个较小的自回归语言模型去模仿这种检索分布；
- 推理时让这个小模型和原始 base LM 并行前向，然后把两者的 next-token 概率融合。

因此，这里的 “Memory Decoder” 本质上不是一个挂在 base model 里的额外 cross-attention memory 模块，也不是经典意义上的 encoder-decoder 里的 decoder，而是：

> 一个单独训练好的、可插拔的、以参数形式存储领域记忆的因果语言模型。

这也是论文标题里 “pretrained” 和 “plug-and-play memory” 的真实含义。

---

## 2. 仓库整体结构

这个仓库可以分成 4 个部分看：

### 2.1 数据预处理

- `utils/preprocess_dataset.py`

作用：

- 对原始文本做 tokenizer；
- 用 sliding window 切成定长块；
- 生成 `input_ids`、`labels`、`attention_mask`；
- 额外生成一个很关键的字段 `dstore_range`，它把“这个样本里有效预测 token 对应到外部 kNN 监督文件中的哪一段”记录下来。

这个 `dstore_range` 是后面把检索得到的稀疏分布重新对齐回训练样本的桥梁。

### 2.2 外部 kNN 监督信号构建

- `train_base.py`
- `knn_utils/saveEmbedMulti.py`
- `knn_utils/build_index.py`
- `knn_utils/saveKNNMulti.py`
- `scripts/save_pipeline.sh`

作用：

- 先从一个已有 LM 中抽取每个 token 位置的隐藏表示，形成 datastore；
- 用 FAISS 给这些向量建索引；
- 对每个位置做近邻搜索，得到一个 token-level 的 kNN 概率分布；
- 把这个分布保存成 Arrow 文件，供 Memory Decoder 训练使用。

### 2.3 Memory Decoder 训练

- `train_memdec.py`
- `utils/cal_loss.py`
- `scripts/train_memdec.sh`

作用：

- 读取语言模型训练样本；
- 读取离线保存好的 kNN 分布；
- 用 KL loss + 标准 LM loss 训练一个新的 causal LM；
- 这个模型最终就成为 “memory decoder”。

### 2.4 推理与评估

- `evaluate_joint.py`
- `demo/memDec.py`
- `demo/generation_example.py`

作用：

- 加载 base LM；
- 加载训练好的 memory decoder；
- 对同一段前缀分别前向；
- 将两边 logits 先转成 log-prob，再做加权融合；
- 输出联合分布对应的 token。

---

## 3. 先说结论：这个 decoder 到底是怎么工作的

如果只保留一句最核心的话，可以这样理解：

> Memory Decoder 先在训练期学习“如果我此时做一次 kNN 检索，那么检索会偏向哪些 token”，再在推理期直接用参数化模型近似这件事，从而避免在线检索的延迟。

它的工作逻辑可以写成两阶段：

### 3.1 训练阶段

训练目标不是单纯拟合真实下一个 token，而是拟合“检索器给出的 token 分布”。

也就是说，teacher 不再只是 one-hot label，而是一个更丰富的分布 $p_{kNN}(y_t \mid x_{\le t})$。

同时，为了不让模型完全偏离正常语言模型能力，训练时还保留标准 next-token cross entropy。

于是训练损失大致是：

$$
\mathcal{L} = \alpha \cdot \mathrm{KL}(p_{kNN} \parallel p_{mem}) + (1-\alpha) \cdot \mathrm{CE}(y, p_{mem})
$$

这里：

- $p_{kNN}$ 是离线检索产生的 token 分布；
- $p_{mem}$ 是 memory decoder 输出的分布；
- $y$ 是真实下一个 token；
- $\alpha$ 控制“模仿检索器”和“保持语言建模能力”的权重。

### 3.2 推理阶段

推理时不再做 kNN 搜索，而是让两个模型并行输出：

- base LM 输出 $p_{base}$
- memory decoder 输出 $p_{mem}$

最终联合分布是对数空间下的加权混合：

$$
\log p_{joint} = \log \left((1-\lambda)p_{base} + \lambda p_{mem}\right)
$$

代码里用的是数值稳定的写法：

$$
\log p_{joint} = \mathrm{logaddexp}(\log p_{base} + \log(1-\lambda),\; \log p_{mem} + \log \lambda)
$$

直觉上：

- base LM 提供通用语言能力；
- memory decoder 提供领域偏置和长尾知识；
- $\lambda$ 越大，越信任 memory decoder。

---

## 4. 它和 RAG、kNN-LM、DAPT 分别有什么区别

### 4.1 和 RAG 的区别

RAG 是推理时实时检索，把检索文本塞进上下文，再让模型继续生成。

Memory Decoder 不是这样。它把“检索行为”蒸馏到了一个小模型里，推理时不检索，只做两次前向。

所以它的优点是：

- 没有在线检索延迟；
- 不需要额外扩展上下文长度；
- 不需要改 base model 参数。

### 4.2 和 kNN-LM 的区别

kNN-LM 在推理时对每个 token 都去查最近邻，然后把检索分布和 LM 分布插值。

Memory Decoder 的核心思想是：

> 用一个小型 causal LM 去近似这个 “kNN 分布生成器”。

所以它像是把 kNN-LM 的非参数记忆，蒸馏成了一个参数化记忆。

### 4.3 和 DAPT 的区别

DAPT 会继续预训练 base model，本质上是在修改原模型参数。

Memory Decoder 不改原模型，只训练一个外挂模型，因此：

- 更容易复用；
- 不容易破坏 base LM 原有能力；
- 同 tokenizer 的多个 base model 可以共享同一个 memory decoder。

---

## 5. 这个仓库的完整训练链路

下面按代码执行链路，把整个方法拆开。

## 5.1 第一步：预处理数据并建立对齐索引

文件：`utils/preprocess_dataset.py`

### 关键点 1：滑动窗口切块

代码把长文本切成长度为 `block_size` 的窗口，但窗口之间使用 `stride` 滑动，而不是完全不重叠切块。

这样做的目的，是让每个预测 token 都尽量拥有足够上下文，同时通过把重复区域的 label 设成 `-100`，避免重复计算 loss。

这段逻辑对应：

- `cur_input_ids = concatenated_examples['input_ids'][begin_loc:end_loc]`
- `cur_labels[:-trg_len] = [padding_index] * (len(cur_labels) - trg_len)`

本质上是：

- 输入窗口可以有上下文重叠；
- 但真正参与训练监督的，只有这一步新引入的尾部 token。

### 关键点 2：构建 `dstore_range`

这是这个仓库里最容易被忽略、但非常关键的设计。

每个样本会统计自身有多少个有效预测 token：

- 忽略第一个位置；
- 忽略 label 为 `-100` 的位置；

然后把每个样本在整个 datastore 监督文件中的起止位置记成：

$$
dstore\_range = (start, end)
$$

后面训练时就能知道：

> 这个 batch 中第 i 个样本对应的 kNN 分布，应该去外部 Arrow 文件的哪一段切出来。

这一步让“语言模型样本”与“离线检索监督”严格按 token 对齐。

---

## 5.2 第二步：保存 datastore 向量

文件：

- `train_base.py`
- `knn_utils/saveEmbedMulti.py`

这部分的目标是从一个已有 LM 中抽取每个 token 的查询向量和目标 token。

### 关键点 1：抽取哪一层的向量

`saveEmbedMulti.py` 中有一个 `KEY_TYPE`，默认是：

- `last_ffn_input`

对 GPT-2，它对应：

- 最后一层 block 的 MLP 输入

也就是：

```python
'gpt2': {
    KEY_TYPE.last_ffn_input: (lambda model: model.base_model.h[-1].mlp, True),
    KEY_TYPE.last_ffn_output: (lambda model: model.base_model.h[-1], False),
}
```

直觉上，这个位置的隐藏状态已经融合了足够多的上下文信息，适合作为 “我现在要预测下一个 token 时的语义查询向量”。

### 关键点 2：用 hook 截获激活

`KNNSaverMulti` / `KNNWrapperMulti` 并没有改模型结构，而是通过 forward hook：

- 截获隐藏表示作为 key；
- 截获对应 token label 作为 value；
- 最后把 `(key, value)` 逐步写入 Arrow datastore。

所以 datastore 里实际上保存的是：

- `keys`: 每个有效 token 位置的隐藏表示；
- `vals`: 对应的目标 token id。

这和经典 kNN-LM 的做法一致。

---

## 5.3 第三步：建立 FAISS 索引并离线生成 kNN 分布

文件：

- `knn_utils/build_index.py`
- `knn_utils/saveKNNMulti.py`

这一步的逻辑是：

1. 对每个查询向量去 datastore 中找最近邻；
2. 根据距离做 softmax，得到每个邻居的权重；
3. 把这些权重 scatter 到对应 token id 上；
4. 最终得到一个词表大小的概率分布。

公式上可以写成：

$$
w_j = \mathrm{softmax}(-d_j / T)
$$

$$
p_{kNN}(v) = \sum_{j: y_j = v} w_j
$$

其中：

- $d_j$ 是查询到第 $j$ 个近邻的距离；
- $T$ 是 `knn_temp`；
- $y_j$ 是该近邻对应的 token id。

### 关键代码逻辑

在 `saveKNNMulti.py` 中：

- `get_knns()` 返回距离和近邻 id；
- `knns_to_probs()` 用 `scatter_add` 把近邻概率聚合成 vocab 分布；
- `sparsify_distribution()` 只保存非零概率 token，降低存储成本。

因此输出 Arrow 文件并不是一个完整的稠密词表分布，而是稀疏格式：

- `id_cnt`: 当前 token 位置有多少个非零 token；
- `token_id`: 非零概率对应的 token id 列表；
- `prob`: 对应概率；
- `label`: 真实标签 token。

这一步非常重要，因为如果直接保存完整 vocab 分布，存储会非常大。

---

## 5.4 第四步：训练 Memory Decoder

文件：

- `train_memdec.py`
- `utils/cal_loss.py`

这是整个仓库最核心的部分。

### 关键点 1：训练对象是谁

`train_memdec.py` 里只加载了一个 `AutoModelForCausalLM`，这个模型本身就是 memory decoder。

也就是说，训练时并没有把 base LM 和 memory decoder 一起联合训练。base LM 在这个阶段只起到“前面构造监督信号时的教师来源”作用。

更准确地说：

- Memory Decoder 的训练输入是普通的 token 序列；
- 训练标签不是只有 one-hot 真值；
- 还有一个来自离线 kNN 的软分布监督。

### 关键点 2：如何把稀疏 kNN 分布喂给模型

`train_memdec.py` 中自定义了 `knn_collate_fn()`。

它会：

1. 读取 batch 样本自带的 `dstore_range`；
2. 去 knn Arrow 文件里切出对应 token 范围；
3. 把稀疏的 `token_id + prob` 重新还原成稠密的 vocab 分布；
4. 拼接成 `batch["knn_probs"]` 和 `batch["knn_label"]`。

所以训练时每个有效 token 都会拿到一个 teacher distribution。

### 关键点 3：损失函数设计

在 `utils/cal_loss.py` 里，`kl_loss_token()` 的实现是：

$$
\mathcal{L}_{KL} = \mathrm{KL}(\log p_{mem}, p_{kNN})
$$

$$
\mathcal{L}_{LM} = \mathrm{CE}(p_{mem}, y)
$$

$$
\mathcal{L}_{total} = \alpha \mathcal{L}_{KL} + (1 - \alpha) \mathcal{L}_{LM}
$$

代码里的含义很直白：

- KL loss 让 decoder 学会“像检索器一样分配概率”；
- CE loss 保证 decoder 仍然是个正常的因果语言模型，不会只会模仿一份过于尖锐或噪声化的教师分布。

### 关键点 4：这个模型把“记忆”存在哪里

答案是：存进参数里。

这就是为什么论文把它称为 **pretrained memory**。

训练完成后，decoder 的参数已经学会：

- 在某些领域前缀下；
- 哪些 token 更像检索器会偏好的答案；
- 从而在推理时直接输出带有领域偏置的分布。

所以它本质上是一种 “retrieval behavior distillation”。

---

## 5.5 第五步：推理时如何与 base model 融合

文件：

- `evaluate_joint.py`
- `demo/memDec.py`

### 关键点 1：两个模型并行接收同一前缀

推理时：

- base LM 前向一次，输出 `logits_base`；
- memory decoder 前向一次，输出 `logits_knn`；

这里的 `knn` 变量名有点历史包袱。严格来说，这时已经不是检索器，而是“近似检索器行为的 memory decoder”。

### 关键点 2：先转 log-prob 再融合

在 `demo/memDec.py` 的 `forward()` 里，关键逻辑是：

```python
logp_base = F.log_softmax(logits_base, dim=-1)
logp_knn = F.log_softmax(logits_knn, dim=-1)

logp_joint = torch.logaddexp(
    logp_base + torch.log(torch.tensor(1.0 - self.lmbda, device=logp_base.device)),
    logp_knn + torch.log(torch.tensor(self.lmbda, device=logp_base.device)),
)
```

这一步是标准的 mixture of experts 风格概率融合，只不过 expert 只有两个：

- 一个是原始 LM；
- 一个是 memory decoder。

### 关键点 3：生成时维护两套 KV cache

`demo/memDec.py` 最工程化、也最值得注意的地方在 `generate()`。

因为有两个独立模型，所以它分别维护：

- `past_key_values` 给 base LM；
- `knn_past_key_values` 给 memory decoder。

每生成一个 token：

1. 用两个 cache 分别前向；
2. 融合最后一个位置的联合 log-prob；
3. greedy 选下一个 token；
4. 更新两套 cache；
5. 继续循环。

这意味着它在推理成本上大致接近：

> 一次 base LM 前向 + 一次较小 memory decoder 前向

而不是一次在线检索 + 一次长上下文重编码。

---

## 6. 为什么这个方法能工作

从方法论上看，这个仓库利用了一个非常关键的观察：

> kNN 检索的强项并不是“把文档拼进去”，而是它能在某个隐藏状态附近，给出一个带有领域偏好的 next-token 分布。

如果一个小型 decoder 能学会从当前前缀直接预测出这种分布，那它就等价于把外部非参数记忆压缩成了参数记忆。

换句话说，Memory Decoder 学的不是原始知识库本身，而是：

- 当前上下文在领域空间中会落到哪里；
- 检索器在这个位置通常会偏向哪些 token。

这是一种比直接模仿 one-hot label 更强的监督，因为：

- one-hot 只告诉你“标准答案是谁”；
- kNN 分布还告诉你“哪些备选 token 也合理，以及它们有多合理”。

这会让模型更容易学到长尾领域词汇和局部事实模式。

---

## 7. 这个 repo 里 decoder 的“实现本质”

很多人看到名字会误以为这里的 decoder 是一种新层或者新模块。实际上，从代码看，它更接近下面这个定义：

### 定义

Memory Decoder = 一个单独训练好的 causal LM，用于近似外部 kNN 检索器产生的 next-token 分布。

所以它的实现本质不是“结构创新”，而是“训练目标创新 + 推理融合策略创新”。

### 它不是什么

它不是：

- RAG 的检索模块；
- transformer 里的 cross-attention memory bank；
- prefix tuning / adapter / LoRA；
- 需要修改 base model 权重的域适配模块。

### 它是什么

它是：

- 一个共享 tokenizer 的外挂 causal LM；
- 通过离线 kNN 教师分布训练出来；
- 推理时与 base model 概率插值；
- 用参数来承载领域记忆。

---

## 8. 从代码实现角度总结 decoder 的输入、输出和训练目标

### 输入

- 与普通 causal LM 相同的 `input_ids` 和 `attention_mask`

### 输出

- 一个词表大小的 next-token logits 分布

### 训练监督

- 真实标签 `y`
- 离线 kNN 分布 `p_{kNN}`

### 推理时的联合输出

$$
p_{joint} = (1-\lambda)p_{base} + \lambda p_{mem}
$$

### 其中几个超参数的作用

- `alpha`: 训练时 KL loss 和 LM loss 的混合权重
- `lmbda`: 推理时 base LM 与 memory decoder 的融合权重
- `knn_temp`: 构建 kNN 教师分布时的温度；在 demo 包装器中也可用于缩放 memory decoder logits

---

## 9. 这个实现里几个容易忽略的工程细节

### 9.1 训练时保存的是稀疏分布，不是完整词表分布

这显著降低了存储量，但训练 batch 时需要重新 densify。

### 9.2 `dstore_range` 是整个管线能对齐起来的关键

没有这个范围映射，训练集样本和离线 kNN token 分布无法一一对应。

### 9.3 `demo/memDec.py` 目前只支持 greedy decoding

代码里明确限制了：

- `do_sample=False`

所以这个 demo 包装器现在不是一个完整替代 Hugging Face generate 的通用实现，而是一个清晰、最小化的演示版本。

### 9.4 `evaluate_joint.py` 中的 `knn_temp` 没有真正参与联合评估

从代码看，`evaluate_joint.py` 解析了 `knn_temp` 参数，但评估时直接对 `knn_outputs.logits` 做 `log_softmax`，并没有像 `demo/memDec.py` 那样先除以温度。

这说明：

- 演示生成封装和评估脚本之间存在轻微实现差异；
- 但主线思想不变，都是把 memory decoder 当作第二个分布来源做融合。

---

## 10. 如果把这个仓库浓缩成一张脑图

可以把整个方法理解成下面这条链：

1. 用基础 LM 抽隐藏状态，建立 token 级 datastore。
2. 用 FAISS 做离线 kNN 搜索，得到每个 token 的教师分布。
3. 用这个教师分布训练一个小型 causal LM。
4. 这个小模型学到“检索器会怎样分配下一个 token 概率”。
5. 推理时不再检索，只让 base LM 和 memory decoder 并行前向。
6. 把两者分布插值得到最终输出。

这就是它相比 kNN-LM 最关键的转变：

> 把测试时的非参数检索，前移并压缩到了训练期的参数学习中。

---

## 11. 一句话评价这个实现

这个仓库里的 Memory Decoder，本质上是在做：

> 用一个小型自回归语言模型去蒸馏外部 kNN 记忆的 token 分布，再把它作为可插拔的第二专家，与 base LM 在推理时做概率级融合。

如果你从工程实现角度理解它，那么最核心的不是 transformer 层怎么改，而是下面三件事：

- 如何离线构造可靠的 kNN 教师分布；
- 如何把这个分布和训练样本逐 token 对齐；
- 如何在推理时以低成本把 memory decoder 和 base LM 融合起来。

这三件事分别对应这个仓库中的：

- `knn_utils/`
- `utils/preprocess_dataset.py` + `train_memdec.py`
- `demo/memDec.py` + `evaluate_joint.py`

---

## 12. 对你读这个仓库的建议顺序

如果你准备继续深入源码，建议按下面顺序读：

1. 先读 `demo/memDec.py`
   先理解推理时到底怎么把两个模型拼起来。

2. 再读 `train_memdec.py` 和 `utils/cal_loss.py`
   看 memory decoder 是如何通过 kNN 分布训练出来的。

3. 最后读 `utils/preprocess_dataset.py`、`saveEmbedMulti.py`、`saveKNNMulti.py`
   补齐教师分布是怎么离线构造出来的。

这样阅读成本最低，也最容易把论文里的概念和代码里的具体实现一一对应起来。