# Transformer Notebook

## Truncated Normalization

在初始化一个 model 的 weight 矩阵的时候没使用随机值可能会导致出现特别大或者特别小的噪点，从而导致在提督计算的时候会出现不稳定的值或者是 NaN, 所以需要规划到 Bell Shape 中来防止这个现象; 同时 Bell Shape 并没有禁止极端值的出现，所以往往规划到 $\pm 3\sigma$ 的 99.7% 区间内来确保初始计算梯度的时候是稳定的

### std 计算公式 (Xavier/Glorot 初始化)

std = sqrt(2/(d_in + d_out)) 表示的是利用输入输出维度的平均值倒数来表示std, 让计算传播过程中方差变化不明

## 逐元素操作 vs 降维操作 (element-wise vs reduction)

PyTorch 的张量操作大致分两类，分清楚才知道哪些要写 `dim`:

- **逐元素 (element-wise)**: `pow` / `**` / `+` / `-` / `*` / `exp` / `sqrt` / `abs` ...
  - 对每个元素独立处理，**形状完全不变**，和维度无关
  - `x.pow(2)` 与 `x ** 2` 等价，就是把每个数平方，`(B,S,D)` 进 `(B,S,D)` 出
  - 这类操作**没有 `dim` 参数**，因为它压根不碰维度结构
- **降维 (reduction)**: `mean` / `sum` / `max` / `min` / `std` / `var` ...
  - 会**压缩(吃掉)某些维度**，把多个数聚合成更少的数
  - 这类操作**才需要 `dim`** 来指定压哪一维

> ⚠️ 易错点: `x.pow(2)` 不是"沿某个维度平方"，它就是逐元素平方。真正"沿最后一维"的是外层的 `mean(..., dim=-1)`。

## tensor 的维度 (dim) 是什么

维度 = "需要几个下标才能定位到一个具体的数"。shape 从外往里编号:

- `shape = (B, S, D)` → dim 0 = B, dim 1 = S, dim -1 = D (最里层)
- `dim 0` 是最外层, `dim -1` 是倒数第一维 = 最里层

在 Transformer 里激活张量典型是 `(batch, seq_len, d_model)`:

- dim 0 (batch): 第几句话
- dim 1 (seq_len): 这句话里第几个 token
- dim -1 (d_model): **这个 token 的特征向量** ← 最常操作的维度

所以"沿 dim=-1 操作" = "对每个 token 各自处理"。

## mean 带不带 dim 的区别

设 `x = [[1,2,3],[4,5,6]]` (shape `(2,3)`):

- `mean(x)` (无 dim): 所有数倒进一个桶 → (1+...+6)/6 = 3.5，**形状塌成标量 `()`**，维度结构全丢
- `mean(x, dim=-1)`: 只沿最后一维，**每行各算各的** → [2, 5]，shape `(2,)`
- `mean(x, dim=0)`: 沿第 0 维(列方向) → [2.5, 3.5, 4.5]，shape `(3,)`

## keepdim=True 的作用

reduction 默认会把被压的维度**删掉**; `keepdim=True` 让它**保留成长度 1**，方便后续广播:

- `mean((B,S,D), dim=-1)` → `(B,S)` (D 被删)
- `mean((B,S,D), dim=-1, keepdim=True)` → `(B,S,1)` (D 保留为 1)

为什么需要 keepdim: 后面要用这个均值去除原张量 `x` `(B,S,D)`:

- `(B,S,1)` 能和 `(B,S,D)` 广播 → 每个 token 的 D 个特征都除以它自己那一个 rms ✓
- `(B,S)` 和 `(B,S,D)` 末维 `D ≠ S` 对不上 → 广播失败 `RuntimeError` ✗

## 串起来: RMSNorm 为什么这样写

```python
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
out = (x / rms) * weight
```

- `x.pow(2)`: 逐元素平方, `(B,S,D)` 形状不变
- `.mean(dim=-1, keepdim=True)`: 只对每个 token 的 D 维向量求均值 → `(B,S,1)`
- `x / rms`: 广播, 每个 token 用**自己**的 rms 归一化
- 若误写成 `mean(x.pow(2))` (无 dim): 把所有 token、所有特征搅成一个全局均值 → 错

## 激活函数的演进: ReLU → GELU → SiLU → SwiGLU

激活函数的作用是**引入非线性**: 没有它，再多层 Linear 叠起来在数学上还是等价于一层 Linear，学不到复杂的东西。

### ReLU

$$\text{ReLU}(x) = \max(0, x)$$

- 最经典、最简单、算得快
- **问题**: ① "dead neuron(死亡神经元)"——负半轴输出恒为 0、梯度也为 0，神经元一旦掉进负区可能永远学不回来; ② 在 0 点不光滑、不可导

### GELU (Gaussian Error Linear Unit)

$$\text{GELU}(x) = x \cdot \Phi(x)$$

其中 $\Phi(x)$ 是标准正态分布的 CDF(累积分布函数，值域 0~1)。

- 直觉: 不像 ReLU 那样"硬切到 0"，而是按"这个值有多大概率该被保留"来**平滑加权**; 输入越大越接近原值，越小越趋近 0，负半轴还留一点点"泄漏"
- 平滑、处处可导; **GPT / BERT 用的就是它**

### SiLU (Sigmoid Linear Unit，又叫 Swish)

$$\text{SiLU}(x) = x \cdot \sigma(x), \quad \sigma(x) = \frac{1}{1 + e^{-x}}$$

- 思路和 GELU 几乎一样(都是 `x` 乘一个 0~1 的平滑门)，只是把门换成更简单的 sigmoid
- 平滑、可导、**非单调**(负的小区间会先降再回升); 比 ReLU 更平滑，实践常更好
- 实现就一行: `x * torch.sigmoid(x)`——逐元素操作，形状不变，没有 `dim`

### SwiGLU —— 注意它不是"激活函数"，而是"带门控的 FFN"

> ⚠️ 关键区分: ReLU / GELU / SiLU 都是**逐元素的激活函数**(一个标量进、一个标量出); 而 **SwiGLU 是一整个前馈网络结构**(GLU 家族)，内部有**可学习的投影矩阵 + 门控机制**。不是同一类东西。

$$\text{SwiGLU}(x) = W_2\big(\,\underbrace{\text{SiLU}(W_1 x)}_{\text{门 gate}} \odot \underbrace{W_3 x}_{\text{值 value}}\,\big)$$

- **GLU (Gated Linear Unit) 思想**: 用一条支路当"门"去逐元素调制另一条支路，让网络**自己学**"哪些信息该通过、通过多少"
  - `SiLU(W1 x)`: 门(平滑的权重)
  - `W3 x`: 值
  - `⊙`: 逐元素相乘做门控
  - `W2`: 投回 `d_model`
- **代价**: 比传统 `Linear → 激活 → Linear`(2 个矩阵)多了一个矩阵 `W3`; 为了参数量持平，通常把 `d_ff` 缩到约 $\tfrac{2}{3}$(经验上 $d_{ff} \approx \tfrac{8}{3}d_{model}$)
- **LLaMA / PaLM 等现代大模型用的就是 SwiGLU**

### 一句话对比

| 名字 | 类别 | 公式 | 备注 |
|---|---|---|---|
| ReLU | 激活函数 | $\max(0,x)$ | 简单快; 死亡神经元、不光滑 |
| GELU | 激活函数 | $x\,\Phi(x)$ | 平滑; GPT/BERT |
| SiLU | 激活函数 | $x\,\sigma(x)$ | 平滑、非单调; GELU 的简化版 |
| SwiGLU | **门控 FFN** | $W_2(\text{SiLU}(W_1x)\odot W_3x)$ | 3 个矩阵 + 门控; LLaMA |