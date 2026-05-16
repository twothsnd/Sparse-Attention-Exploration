# NLP Project Proposal

**标题：**
面向高效长上下文建模的稀疏注意力机制研究

**摘要：**
标准 Transformer 的 full attention 在处理长序列时具有较高的计算和显存开销，限制了模型在长上下文任务中的应用。本项目计划探索稀疏注意力相关算法，包括 *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention* 及其他相关工作。我们将选择若干具有代表性的稀疏注意力机制，在小参数量语言模型上进行实现和训练，并与 full attention 模型进行系统对比。项目目标是分析不同稀疏注意力设计在模型性能、训练效率、推理效率和长上下文建模能力之间的权衡。

**研究问题：**
本项目关注的问题是：在较小模型规模和有限计算资源下，稀疏注意力机制能否有效替代 full attention，并在长上下文场景中取得更好的效率-性能平衡。具体而言，我们希望研究稀疏注意力在小模型中是否仍然有效，以及不同稀疏化策略对模型建模能力、计算成本和长距离信息利用能力的影响。

**使用数据：**
训练数据方面，我们计划使用公开文本语料进行小规模语言模型训练或继续训练，例如常见的开放文本数据集。评估方面，我们将使用语言建模验证集指标衡量模型的基础建模能力，并选择若干长上下文相关 benchmark 评估模型对长距离信息的利用能力。此外，我们会记录不同输入长度下的显存占用、训练速度和推理延迟，从而比较不同注意力机制的实际效率。

**基本方法：**
我们将首先搭建一个统一的小参数量 Transformer 训练框架，并实现 full attention baseline。随后，我们会调研并选择若干代表性的稀疏注意力算法，在尽可能相同的模型规模、训练数据和训练预算下进行实现与训练。为了保证实验对比公平，我们会尽量保持模型主体结构一致，仅改变注意力机制本身。

实验阶段，我们将从三个方面比较不同方法：第一，评估模型在语言建模或下游任务上的性能；第二，测试模型在长上下文 benchmark 中的表现；第三，统计训练和推理过程中的显存、速度与延迟。最终，我们希望通过系统实验总结不同稀疏注意力机制在小规模模型中的优势、局限和适用场景，并分析其相对于 full attention 的实际收益。

## 参考文献

[1] Jingyang Yuan et al. *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*. arXiv:2502.11089, 2025.

[2] Iz Beltagy, Matthew E. Peters, Arman Cohan. *Longformer: The Long-Document Transformer*. arXiv:2004.05150, 2020.

[3] Manzil Zaheer et al. *Big Bird: Transformers for Longer Sequences*. arXiv:2007.14062, 2020.

[4] Aurko Roy, Mohammad Saffar, Ashish Vaswani, David Grangier. *Efficient Content-Based Sparse Attention with Routing Transformers*. arXiv:2003.05997, 2020.

## Group

**ID:** 39

**Members:** 谢唯 23307130044、罗俊昊 23307130030、兰景茜 23301030082
