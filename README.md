# 实现以及部署过程（方案）

## 一、前景主体分割要求

- 模型大小：不超过100M
- 算法性能指标：在1080 TI GPU上处理一张图片的时间不超过5秒
- 效果指标：主体边缘细节清晰，MIOU越高越好
- 网络模型工程化：离线可进行验证的包含全部依赖的C++例程

## 二、实现方案

使用显著性目标检测(SOD)模型[U2Net](https://github.com/xuebinqin/U-2-Net)，U2Net提供了源码以及两个模型，一个167.3M，一个4.7M。

至此，将问题转化为将**优化U2Net模型大小**，以下为实现方案：

- 降低模型精度，使用float16进行计算（失败，预测不正确，得到全黑的图像）
- 将普通的卷积层改为分组卷积，分为2组
- 将普通的卷积层改为深度可分离卷积

## 三、训练

### 3.1 数据集

将[DUTS-TR](http://saliencydetection.net/duts)（5019张图像）、[DUTS-TE](http://saliencydetection.net/duts)（10553张图片）作为训练集（总共15572张图像），在其他数据集图像（共25615张）随机取1000张作为验证集。（包括[DUT-OMRON](http://saliencydetection.net/dut-omron/#org96c3bab)（5168张）、[ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)（1000张）、[HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html)（4447张）、[MSRA10K](https://mmcheng.net/msra10k/)（10000张）、[MSRA-B](https://mmcheng.net/msra10k/)（5000张））。

### 3.2 训练细节
在两张1080 TI上训练，批次大小(batch size)设置为16，每迭代2000次，保存一次模型，
迭代200000次，结束训练（即保存有100个模型），其他参数使用原U2Net的参数。
<!-- 当训练整体损失（由side1-6以及side fuse分类的损失相加）小于0.5或者side fuse分类的损失小于0.05时停止训练。 -->

训练时使用以下两种参数初始化方式：
- 加载修改后U2Net的模型（167.3M）做为初始化参数、
    1. 分组卷积（将卷积层权重相邻的通道，两两切分作为一组，取平均值，得到一个通道，处理权重的所有通道，并将处理后的参数级联到一起。）【模型大小86M】
    2. 深度可分离卷积（将depthwise conv的权重设置为原来普通卷积层(out_channels, in_channels, kernel_h, kernel_width)的第一维取平均得到，再第一维与第二位做转置，得到(in_channels, 1, kernel_h, kernel_width)的张量作为初始化参数；depthwise conv的bias，设置为size为(in_channels)的全零张量，pointwise conv的权重，将原来普通卷积层的张量的第三第四维求平均得到(out_channels, in_channels, 1, 1)大小的张量，pointwise conv的bias，使用原来普通卷积层的bias。）【模型大小21M】
- 直接默认的初始化方式，进行训练。（尝试之后，决定放弃这种训练方式）

加载修改后的模型初始化参数，可以加快损失下降速度（而且不只是快一点点。）如下图为分组卷积在两种不同训练方式下的损失变化。

默认初始化：训练十几轮（每轮2000次迭代）从0.48降到0.4左右
![](./figures/train_groupconv_nopretrain.png)

修改后的模型初始化参数：训练三十几轮（每轮2000次迭代）从0.44左右降到0.06左右，十几轮就已经降到了0.1左右
![](./figures/train_groupconv_pretrain.png)

## 四、部署方案

将pytorch下训练得到的pth模型，转为onnx模型，使用OpenCV中的dnn模块加载进行推理（其中对OpenCV进行重新编译，加入CUDA加速处理相关依赖，时间从1800ms左右，降到了120ms左右，在i5-7400 CPU @ 3.00GHz + 1050Ti的主机上）。