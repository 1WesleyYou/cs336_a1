# Tokenizer 设计思路

## Unicode, UTF-8

encoding 的过程可以理解为是人类语言到计算机语言的一个转换过程

- Unicode 可以理解为是人类角度的一个编码方案
- UTF-8 可以理解为是计算机角度的一个编码方案
  - UTF-8 表示最小表示单位是 8b 但支持多字节, 是动态长度编码
  - UTF-8 兼容了 ASCII 编码且适配大多数 英语的字符, 所以是目前最流行的编码方案

## Subword Tokenization

标准的 tokenization 就是尝试用数学表达一个人类可以理解的单词的语义, 也就是将自然语义变成 vector;
传统方法中存在的是 word-level language model 和 character-level language model, 但是这两种方法都存在一些问题:

- word level 的粒度比较大, 但是会存在 OOV(out-of-vocabulary) 的问题, 因为词表是有限的, 所以当遇到一些没有见过的单词时, 就无法处理
  - 比如说我们现在词表理解 "I love cats" 的每一个 word, 但是当我们输入 "I love capybaras" 的时候, 由于不认识 `capybaras` 所以会导致输出 `<UNK>`, 结果就是信息大量丢失;
  - 如果这里采用 subword 分解的话至少可以继续拆分 `capybaras` 到 `capy` `baras`
    这样的结果, 可能可以通过识别 `capy` 来推断出 `capybara` 的意思, 这样就不会丢失太多信息了
- character-level 的粒度太小了, 会导致拆分之后的 attention 节点太多, 计算量太过庞大, 有点没有必要了

因此找一个折中的方案: __subword tokenization__ 也就是不用完整的单词那么大粒度, 也不用单一的单词那么小的粒, 而采用介于之间的粒度;

### BPE Tokenization

BPE (Byte Pair Encoding) 是一种基于统计的 subword tokenization 方法, 其核心思想是通过统计 __语料中出现频率最高的__ byte pair 来进行合并 (merge), 从而得到一个新的 subword token;

这样做的好消息是: 可以将热点单词压缩成一个 token, 从而避免 character-level
的计算复杂问题, 并且也不会再出现 word level 的 OOV 问题了因为我们可以通过合并的方式不断扩展 vocabulary 的覆盖范围;

也就是我们在做 tokenization 的时候要统计临近字母组合出现的频率, 但是这个步骤会相当地消耗算力因为任意两个字母临近都要计算的组合太多了, 并且我们这里合并之后还会递归地继续进行合并

### Pre-Tokenization

基于上面的讨论, Pre-Tokenization 的目的就是两个:

- 减少计算成本
- 避免学到无效, 过于碎片的语义 token

#### 节省计算成本

按照 naive 实现, 计算一起出现的次数的时候每次都要扫描一整个 corpus (语料库),
也就是要看所有的 character 集合, 太琐碎, 所以要先粗糙地切割一次, 比如按照
空格, 标点等切分成单词, 这样就可以统计单词之间的组合了, 不用单词内部乱搞了

#### 避免学到无效 token

比如有单词 `dog` 和 `dog!` 两个应该就是完全一个意思, 但是如果直接切割可能会认为这是两个意思, 但是实际上我们要合并起来认为它们是一个产物, 才能避免语义学习不对

