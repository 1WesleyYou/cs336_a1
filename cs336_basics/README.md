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

## nn.Parameter vs register_buffer vs 普通张量

挂在 `self.` 上的张量分三类, `nn.Module` 对它们处理不同:

| 存法 | 会训练(梯度更新)? | 跟 `.to(device)` 搬? | 进 checkpoint? |
|---|---|---|---|
| `nn.Parameter` | ✅ | ✅ | ✅ |
| `register_buffer` | ❌ | ✅ | ✅(persistent=True 时) |
| 普通 `self.x = tensor` | ❌ | ❌ | ❌ |

- **`nn.Parameter`**: "这是要训练的权重"。只有它进 `model.parameters()`、被优化器更新。忘了包会导致权重永远学不动(不报错、极难查)。
- **`register_buffer`**: "固定状态,不训练,但要跟模型搬设备、可存档"。RoPE 的 cos/sin 表就是它——常量、不学、但 forward 要和输入同设备。用 `self.register_buffer("cos", t, persistent=False)` 注册,用 `self.cos` 访问。
- **`persistent=False`**: 不存进 checkpoint(能随时重算,省空间)。

### device 与 .to()

张量数据物理上住在 `cpu`(主内存 RAM)或 `cuda`(GPU 显存);运算时所有张量必须同设备。`.to(dtype)` 转类型、`.to('cuda')` 搬设备,都**返回新张量**要赋值接住。

## 调用 nn.Module,不是矩阵乘

`nn.Module`(Linear/RMSNorm/整个模型)要像函数一样**调用**来跑 forward:

```python
self.w1(x)      # ✅ 调用 → 触发 forward → 内部才做 x @ weight.T
self.w1 @ x     # ❌ w1 是模块对象不是矩阵,报 TypeError
```

矩阵乘是层"内部"的事;从外面只管"调用"这个层。

## @ (矩阵乘) vs * (逐元素);// 整除

- **`@`** = `torch.matmul`: `(...,m,k) @ (...,k,n) → (...,m,n)`,**左末维 == 右倒数第二维**,内维收缩;batch 维广播。例: `x @ W.T`、`Q @ K.transpose(-2,-1)`、`attn @ V`。
- **`*`** = 逐元素乘: 形状不变(或广播),对应位置相乘。例: RMSNorm `* weight`、门控 `⊙`。
- ⚠️ 两个一维向量 `(k,) @ (k,)` → 标量(点积)——曾在 RMSNorm 误用 `@` 把 d_model 点积掉。
- **`//`** = 整除(向下取整,返回 int): `d_model // num_heads` 得整数 `d_k`(形状必须 int,`/` 给 float 会报错)。

## squeeze / unsqueeze

- **`unsqueeze(dim)`**: 在 dim 位置**插入**一个 size-1 维,`(3,) → (3,1)`。等价 `x[..., None]`。
- **`squeeze(dim)`**: **删掉** dim 位置的 size-1 维(不是 1 就不动),`(3,1) → (3,)`。
- 用途: 对齐形状/广播。如 `keepdim=True` 留的 size-1 维用 `squeeze` 去掉。

## softmax(数值稳定)

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

坑: `x` 有大数时 `exp` 溢出。技巧 **先减最大值**(结果不变,分子分母同乘常数):
```python
m = x.max(dim=dim, keepdim=True).values
e = torch.exp(x - m)
return e / e.sum(dim=dim, keepdim=True)
```
减完最大指数变 `e^0=1`,绝不溢出。`max`/`sum` 都 `keepdim=True`。

## Scaled Dot-Product Attention(注意力核心)

$$\text{Attention}(Q,K,V) = \text{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V$$

**直觉(信息检索)**: 每个 token 产出 Query("我找什么")、Key("我有什么")、Value("我的信息")。对每个 query 和所有 key 算点积相似度 → softmax 成权重 → 加权平均所有 value。**本质: 每个 token 按相关性从其他 token 收集信息**。

- **为什么 ÷√d_k**: 点积是 d_k 个数相加,维度越高方差越大,分数太大 softmax 饱和、梯度趋 0。除 √d_k 拉回。
- **softmax 沿 dim=-1(keys)**: 每个 query 对所有 key 权重和为 1。
- **因果 mask**: 语言模型不能看未来。`torch.tril(torch.ones(seq,seq,bool))` 下三角 True(query i 只看 key j≤i);`masked_fill(~mask, -inf)` → softmax 后未来权重≈0。

## RoPE(旋转位置编码)

注意力本身无序,需注入位置。RoPE 不"加"位置向量,而是**按位置把 Q、K 旋转一个角度**。

- **成对旋转**: d_k 维向量相邻两两分组 `(x[2i], x[2i+1])`,每对当二维平面点转(旋转本质是二维操作,要两个轴)。
- **二维旋转**(角 θ): `x0' = x0·cosθ − x1·sinθ`,`x1' = x0·sinθ + x1·cosθ`。**长度不变,只转方向**(是旋转不是位移)。
- **角度 = 位置 × 该对频率**: `θ = p·θ_i`,`θ_i = 1/base^(2i/d_k)`(base 常 10000)。频率从 1 等比衰减到 ≈1/base——高频对区分局部、低频对表达长程(像钟表秒针/时针)。频率数组全局共享、固定不学。
- **为什么能表达相对位置**: 两向量各自旋转后点积只依赖旋转角之差 `(m−n)θ` → 只看相对距离。
- **只转 Q、K 不转 V**: 位置影响"谁关注谁"(QKᵀ 分数),不改"被传递的内容"(V)。
- **实现**: `__init__` 预计算所有位置的 cos/sin 表(register_buffer 缓存,避免每次 forward 重算);`forward` 按 token_positions 查表 → 拆偶奇对 → 套旋转公式 → `stack(...).flatten(-2)` 交错拼回。

## einops rearrange(命名轴变形)

用字符串描述形状变换,比 reshape+transpose 可读且不易错:

- **括号 `(a b)` = 一个轴 = a×b**(拆分/合并);空格 `a b` = 两个独立轴。
- **拆头**: `rearrange(Q, "... seq (h d) -> ... h seq d", h=num_heads)` — d_model 拆成 h×d、h 提前 → `(...,num_heads,seq,d_k)`。
- **合头**: `rearrange(x, "... h seq d -> ... seq (h d)")` — 逆操作。
- `...` = 任意前置维。名字随便起(标签),但**结构(轴数/顺序/括号)是真执行的指令**,不是注释。

## Multi-Head Attention(多头注意力)

不做一个 d_model 大注意力,而拆成 `num_heads` 个小注意力(各 `d_k = d_model/num_heads`)独立做再拼回。多个头学不同类型的关系。

流程:
1. `q/k/v_proj(x)` → Q,K,V 各 `(...,seq,d_model)`(一个大矩阵算出所有头的投影)
2. rearrange 拆成 `(...,num_heads,seq,d_k)`
3. RoPE(Q), RoPE(K)
4. 因果 mask
5. SDPA(num_heads 当 batch,所有头并行)
6. rearrange 合头 → `(...,seq,d_model)`
7. `output_proj`

关键: **`d_model = num_heads × d_k`(必须整除)**;投影输出的 d_model 被解释成"num_heads 个头各 d_k",前 d_k 个是头0、接着头1……

## Transformer Block(pre-norm)

一个 block 两个子层(注意力 + 前馈),各配一个 norm(共 2 个,**不是 double norm**——那种每子层前后各一个共 4 个):

```python
y   = x + attn(ln1(x))     # 注意力子层,norm 在输入端
out = y + ffn(ln2(y))      # 前馈子层
```

**pre-norm**: norm 放子层输入,残差通路保持干净 → 梯度顺畅、训练稳(GPT-2 之后主流)。代价: 残差流幅度随层数累加,所以最后要补一个 `ln_final`。

## Transformer LM(完整模型)

```
token IDs ─[Embedding]→ 向量 ─[Block×N]→ ─[ln_final]→ ─[lm_head]→ logits
```

- `token_embeddings`(Embedding): 整数 ID → 向量(查表)
- `layers`: **必须用 `nn.ModuleList`**(普通 list 里的 block 不注册成子模块,参数不进 model.parameters())
- `ln_final`(RMSNorm): pre-norm 收尾
- `lm_head`(Linear d_model→vocab): 输出每个词的分数(logits)
- 所有层**共享一个 RoPE**

## logits: 模型输出的是"每个词的分数",不是索引

forward 每个位置输出 `vocab_size` 个数(logits),**不是一个 token 索引**:

```
logits(vocab个分数) → softmax → 概率 → argmax/采样 → 一个索引
                                          ↑ 生成时你做的选择,不是模型输出
```

`lm_head = Linear(d_model, vocab_size)` 就是为每个词出一个分数。索引是**顺着**分数挑的,不是反推。