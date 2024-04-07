## 一、背景介绍
Word2Vec是词嵌入的经典模型，它通过词之间的上下文信息来建模词的相似度。TransE是知识表示学习领域的经典模型，它借鉴了Word2Vec的思路，用“头实体+关系=尾实体”这一简单的训练目标取得了惊人的效果。本次任务要求在给定的框架中分别基于Text8和Wikidata数据集实现Word2Vec和TransE，并用具体实例体会词向量和实体/关系向量的含义。

## 二、实验目的
### 2.1 Word2Vec
* 探究Word2Vec中各个参数对模型的影响，例如词向量维度、窗口大小、最小出现次数。
* Word2Vec模型进行改进，改进的方法可以参考“A uniﬁed model for word sense representation and disambiguation. in Proceedings of EMNLP, 2014”，包括加入词义信息、字向量和词汇知识等方法。详细叙述采用的改进方法和实验结果分析。
### 2.2 TransE
* 在文件`TransE.py`中，需要补全`TransE`类中的缺失项，完成TransE模型的训练。需要补全的部分为：
* `_calc()`：计算给定三元组的得分函数(score function)
* `loss()`：计算模型的损失函数(loss function)
* 完成TransE的训练，得到实体和关系的向量表示，存储在`entity2vec.txt`和`relation2vec.txt`中。
* 给定头实体Q30，关系P36，最接近的尾实体是哪些？
* 给定头实体Q30，尾实体Q49，最接近的关系是哪些？
* 在 https://www.wikidata.org/wiki/Q30 和 https://www.wikidata.
org/wiki/Property:P36 中查找上述实体和关系的真实含义，看结果是否合理并分析原因。
* 改变参数`p_norm`和`margin`，重新训练模型，分析模型的变化。
## 三、实验环境
|Model|Version|
| ----------- | ----------- |
 | CPU | 			Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz|
 | GPU | 			NVIDIA GeForce RTX 3070|
 | NVIDIA-Driver: 515.86.01   |
 | NVIDIA-CUDA: 11.7 |	   
 | OS | 			Linux Mint 21.1 Cinnamon|
 | python: 		3.10.9	|
 | pytorch | 	1.13.1|
 | pytorch-cuda | 11.7|
 | pytorch-mutex:1.0|
 | torchaudio | 0.13.1|
 | torchvision | 0.14.1|
 | matplotlib | 	3.7.0 |
 | numpy | 	1.23.5|
 | pandas | 	1.5.3|
 | scipy | 		1.10.0|
 | jupyter_client | 7.4.9 |
 | gensim | 	4.3.0|
## 四、Word2Vec实现
#### 所有Word2Vec实验的训练过程及模型参数的可视化对比，详见 Word2Vec/word2ver.ipynb
#### 所有Word2Vec实验的结果，用Pandas记录后，以CSV的格式保存，详见 Word2Vec/exec_result.csv
## 五、TransE实现、训练及结果
#### ransE对应的代码、训练过程和结果记录文件，详见 TransE/TransE.ipynb	