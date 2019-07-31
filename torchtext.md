# Torchtext 工作流

前提：数据集，最好已经划分 train/dev/test

torchtext 最核心的类为 **Field**， 字段。那么这个字段是指什么呢？以最常见的 NLP 数据集为例，每个样例的字段为：(text, label)，而我处理的多模态数据集字段为：(text, visual, acoustic), label, utterance_id 。如果以每行一个样例的形式构建数据集，字段可以理解为数据集的列名。由于每列对应的是不同的数据，所以我们可以为每列创建对应的 Field，每个 Field 只处理对应列的数据。比如多模态，我只需要前面四个字段的数据，所以创建四个 Field 就好。

```python
# TEXT 负责处理文本数据
TEXT = Field(lower=True, include_lengths=True, batch_first=True)

# VISUAL 负责处理视觉数据
VISUAL = Field(sequential=False, use_vocab=False, postprocessing=pp, dtype=torch.float, batch_first=True)

# ACOUSTIC 负责处理听觉数据
ACOUSTIC = Field(sequential=False, use_vocab=False, postprocessing=pp, dtype=torch.float, batch_first=True)

# LABEL 负责处理标签数据
LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float, batch_first=True)
```

1. 初始化 Field ：sequential=True 则有 pad_token='<pad>'，否则 pad_token=None。
2. 构建 Dataset
   - 本地数据集给 path；否则不给 path，但要给 root
   - 将数据集中的样例用 Field 进行预处理，然后添加到 Example 的属性。
   - 将得到的所有 Examples 和 Fields 添加到 Dataset 对象属性。
3. 为 text 构造词库
4. 为每个数据集构造 Iterator 

关键参数：

预处理过程 (preprocess)：构造 Example 的过程中，需要先对原始数据进行预处理。

- sequential=True and x is string $\rightarrow$ tokenize
- lower=True $\rightarrow$ lower(x)
- sequential and use_vocab and stop_words not None $\rightarrow$ 删掉 x 中的 stop_words
- **preprocessing** not None $\rightarrow$ do preprocessing

构造词库过程 (build_vocab) ：

- equential=False $\rightarrow$ x = [x]

构造 batch 过程 (process)：

- (pad) equential=False $\rightarrow$ 不填充，直接返回原数据
- (numericalize) se_vocab and sequential $\rightarrow$ arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
- (numericalize) use_vocab and not sequential $\rightarrow$ arr = [self.vocab.stoi[x] for x in arr]
- (numericalize) **postprocessing** not None $\rightarrow$ do postprocessing

注意：

1. 不是 sequential 也可以选择 use_vocab
2. 如果 use_vocab，一定要构建词库，否则会报错。构建词库并不受 use_vocab 约束，只是添加一个新属性。你可以建了不用，但是不能说要用，但是不建。