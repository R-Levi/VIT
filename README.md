# PyTorch-ViT-Vision-Transformer
## Introduction
PyTorch implementation of the Vision Transformer architecture using this paper: https://arxiv.org/pdf/2010.11929.pdf. 

![](.\images\VIT.jpg)

## Methodology
(1) **patch embedding**：例如输入图片大小为224x224，将图片分为固定大小的patch，patch大小为16x16，则每张图像会生成224x224/16x16=196个patch，即输入序列长度为196，每个patch维度16x16x3=768，线性投射层的维度为768xN (N=768)，因此输入通过线性投射层之后的维度依然为196x768，即一共有196个token，每个token的维度是768。这里还需要加上一个特殊字符cls，因此最终的维度是197x768。特殊字符cls对应的输出作为encoder的最终输出 ，代表最终的image presentation

(2) **positional encoding**：ViT同样需要加入位置编码，位置编码可以理解为一张表，表一共有N行，N的大小和输入序列长度相同，每一行代表一个向量，向量的维度和输入序列embedding的维度相同（768）。注意位置编码的操作是sum，而不是concat。加入位置编码信息之后，维度依然是197x768

(3) **LN/multi-head self attention/LN**：LN输出维度依然是197x768。多头自注意力时，先将输入映射到q，k，v，如果只有一个头，qkv的维度都是197x768，如果有12个头（768/12=64），则qkv的维度是197x64，一共有12组qkv，最后再将12组qkv的输出拼接起来，输出维度是197x768，然后在过一层LN，维度依然是197x768

(4) **MLP**：将维度放大再缩小回去，197x768放大为197x3072，再缩小变为197x768



## Results 
**test on MNIST**

‘.’是top1精度，‘-’是top5精度。

<img src="images\MNIST\my_accuracy.png" style="zoom:40%;" />

![](C:\Users\LEVI\Desktop\VIT\images\MNIST\my_cm_train.png)
