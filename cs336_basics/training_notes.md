# Training Notebook

CS336 训练部分学习笔记: 损失、优化器、学习率调度、梯度裁剪、数据采样、checkpoint。
(模型部分的笔记见 [README.md](./README.md) 的 Transformer Notebook。)

## 参数 vs 超参数

| | 参数 (parameters) | 超参数 (hyperparameters) |
|---|---|---|
| 谁决定 | **训练学出来**(梯度更新) | **你手动设定** |
| 训练中 | 每步都变 | 固定不变 |
| 数量 | 极多(百万~百亿) | 很少(几个) |
| 例子 | 所有 `nn.Parameter`(权重矩阵、embedding…) | lr、betas、eps、weight_decay、d_model、num_layers… |

> 优化器做的事 = "用**超参数**(旋钮)去更新**参数**(模型的知识)"。参数是被改的对象,超参数是改的规则。

## Cross-Entropy(交叉熵损失)

**概念**: 衡量"模型对正确答案有多惊讶"。模型给正确词的概率 `p`,损失 = `−log(p)`:

- `p=1`(自信且对) → 损失 0;`p→0`(押错) → 损失 →∞。
- 训练 = 最小化惊讶 = 让模型给真实的下一个词更高概率。

**为什么是分类损失**: 完整交叉熵 `H(p,q) = −Σ p·log q`,真实答案是 one-hot,求和塌成一项 `−log q(正确词)`。

**数值稳定公式**(别 softmax 再 log):

$$\text{CE} = \text{logsumexp}(\text{logits}) - \text{logit}[\text{target}]$$

推导: $-\log\frac{e^{z_c}}{\sum_j e^{z_j}} = \log\sum_j e^{z_j} - z_c$。

> **为什么 target_logit 不用 exp**: `log(e^{z_c}) = z_c`——分子是单项,exp/log 抵消,只剩原始 logit;分母是"求和",`log(Σ)` 拆不开,保留成 logsumexp。两项都在 log 空间,对齐。

```python
m = inputs.max(dim=-1, keepdim=True).values
logsumexp = m.squeeze(-1) + torch.log(torch.exp(inputs - m).sum(dim=-1))
target_logit = torch.gather(inputs, -1, targets.unsqueeze(-1)).squeeze(-1)
return (logsumexp - target_logit).mean()
```

### logsumexp

`logsumexp(x) = log(Σ eˣ)` = softmax 分母的 log。直接算会溢出(`exp(大数)=inf`),用稳定形式 `max(x) + log(Σ exp(x−max(x)))`(减最大值后最大项=1,绝不溢出)。它是"软化的 max"。

### gather

`torch.gather(input, dim, index)` 按索引挑元素。`gather(inputs, -1, targets.unsqueeze(-1))` 取每个样本正确词那**一个** logit。index 维数要和 input 一致(所以先 `unsqueeze`,取完 `squeeze`)。

### logits

模型 forward 每位置输出 `vocab_size` 个分数(logits),**不是索引**。softmax→概率→采样才得索引。

## AdamW(优化器)

**作用**: 反向传播算出**梯度**(loss 对每个参数的偏导),优化器拿梯度更新参数降 loss。

**演进**:

- **SGD**: `θ = θ − lr·g`。所有参数一个 lr,梯度噪声大易震荡。
- **Adam**: 每个参数存两个"记忆":
  - `m`(一阶矩): 梯度滑动平均 → **方向稳**(动量)
  - `v`(二阶矩): 梯度**平方**滑动平均 → 每参数梯度**幅度**
  - `θ = θ − lr·m/(√v+ε)`,**除 √v = 每参数自适应步长**(梯度大走小步)。
- **AdamW**: Adam + **解耦权重衰减** `θ = θ − lr·λ·θ`(正则化,和梯度更新分开做)。

**一步更新(每参数)**:

```
1. m = β1·m + (1−β1)·g              一阶矩(方向)
2. v = β2·v + (1−β2)·g²             二阶矩(幅度)
3. α_t = lr·√(1−β2^t)/(1−β1^t)      偏差修正后的步长
4. θ = θ − α_t·m/(√v+ε)             Adam 更新
5. θ = θ − lr·λ·θ                   解耦权重衰减 (= θ·(1−lr·λ))
```

**超参数**:

- `lr`: 步子大小(大→快但易冲过头)
- `betas`(β1,β2): 两个滑动平均的**衰减率/记忆长短**(β 越大记越久;β1=0.9≈记10步、β2=0.999≈记1000步)
- `eps`: 分母 `√v+eps` **防除零**(不是 decay rate!)
- `weight_decay`(λ): 往 0 拉的强度(正则化,防过拟合)

**state(优化器的每参数记忆)**: `self.state[p]` 存 `m`、`v`、`t`,**跨 step 保留**。第一次见到参数时初始化为 0。

- `t`: 步数计数器,用于**偏差修正**——m/v 从 0 开始早期偏小,`t` 决定补偿多少(早期补得多,`β^t→0` 后修正因子→1 不再补)。

**为什么第 4-5 步两次更新 p.data**: 第4步是梯度下降(**学习**),第5步是权重衰减(**缩向0**),两件不同的事;分开做正是 AdamW 的"解耦"。

**param_groups**: 参数分组,**每组一份超参数**,允许不同参数不同设置(如权重矩阵做 weight decay、bias/norm 不做)。默认 `AdamW(model.parameters())` 只有 **1 组**(全统一);多组是可选口子。step 里 `for group in param_groups` → 内层 `for p in group["params"]`。

**p.data vs p.grad**: `p.data` = 参数**值**(要更新的);`p.grad` = **梯度**(同形状,读来定方向)。`.data` 取原始张量、绕过 autograd 追踪(免得"更新参数"被记进计算图)。

## Cosine 学习率调度

lr 不该恒定。三段(`frac = (it−T_w)/(T_c−T_w)`):

```
it < T_w (warmup):  lr = it/T_w · max_lr                                     线性升
T_w ≤ it ≤ T_c:     lr = min_lr + 0.5(1+cos(π·frac))·(max_lr−min_lr)          余弦衰减
it > T_c:           lr = min_lr                                              恒定
```

- **warmup**: 随机初始权重上大 lr 易震荡/发散,先从 0 线性升到 max 热身。
- **cosine 衰减**: 前期大步探索,后期小步精细收敛。`cos(0)=1`→起点 max,`cos(π)=−1`→终点 min。
- lr 由**外部调度器**改(每步写进 `group["lr"]`),AdamW 自己不改 lr。

## 梯度裁剪(gradient clipping)

防**梯度爆炸**(坏 batch → 巨大梯度 → 参数炸/loss NaN)。把所有梯度合成一个大向量,若 L2 范数超阈值,按比例整体缩小(**保方向、压幅度**):

```python
grads = [p.grad for p in parameters if p.grad is not None]
tot_norm = torch.sqrt(torch.stack([(g**2).sum() for g in grads]).sum())
if tot_norm > max_l2_norm:
    scale = max_l2_norm / (tot_norm + eps)
    for g in grads:
        g.mul_(scale)   # 原地(下划线),改的就是 p.grad
```

> `g.mul_(scale)` 的下划线 = **原地修改**,直接改 `p.grad`;用 `g = g*scale`(新建张量)改不到原 grad。

## get_batch(数据采样)

训练数据是**一整条超长 token ID 序列**。每步随机切 `batch_size` 段、每段长 `context_length`:

```python
n = len(dataset)
starts  = np.random.randint(0, n - context_length, size=batch_size)   # 随机起点
inputs  = np.stack([dataset[i : i+context_length]     for i in starts])
targets = np.stack([dataset[i+1 : i+1+context_length] for i in starts])  # 右移一位
inputs  = torch.tensor(inputs,  dtype=torch.long, device=device)   # long!
targets = torch.tensor(targets, dtype=torch.long, device=device)
```

- **context_length**: 每段长度 = 模型一次看多少 token(上下文窗口)
- **starts**: 随机起点(整条语料太长喂不下,每步抓几段);范围 `[0, n−context_length)` 保证右移的 target 不越界
- **target = input 右移一位**: 语言模型在位置 t 预测 token t+1
- **dtype=long**: token ID 是整数索引,Embedding 查表 / gather 硬性要求 int64(float/int32 会报错)
- **为什么位置0不是瞎猜**: 学的是"给定这个词,下一个词的**概率分布**"(统计规律);且一个序列里每个位置上下文长度不同(位置0只1词,后面越来越长),模型**同时在所有上下文长度上训练**;随机切块让每个词都见过各种上下文。

## Checkpoint(存/取)

训练很久,**定期存档**以便崩溃后续训。存三样:

```python
torch.save({
    "model": model.state_dict(),          # 学到的权重
    "optimizer": optimizer.state_dict(),  # AdamW 的 m/v/t —— 也要存!
    "iteration": iteration,               # 第几步
}, out)
```

读回:

```python
ckpt = torch.load(src)
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
return ckpt["iteration"]
```

> ⚠️ **优化器状态(m/v/t)必须存**,否则续训时动量从 0 重来会抖。`state_dict()` 把状态导出成"名字→张量"字典,`load_state_dict()` 灌回去。
