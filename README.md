# 实现以及部署方案

## 一、前景主体分割要求

- 模型大小：不超过100M
- 算法性能指标：在1080 TI GPU上处理一张图片的时间不超过5秒
- 效果指标：主体边缘细节清晰，MIOU越高越好
- 网络模型工程化：离线可进行验证的包含全部依赖的C++例程

## 二、实现方案

使用显著性目标检测(SOD)模型[U2Net](https://github.com/xuebinqin/U-2-Net)，U2Net提供了源码以及两个模型，一个167.3M，一个4.7M。

至此，将问题转化为将**优化U2Net模型大小**，以下为实现方案：

- 降低模型精度，使用float16进行计算（失败）
- 将普通的卷积层改为分组卷积，分为2组、4组
- 将普通的卷积层改为深度分离卷积

比较结果，最后使用分为2组的分组卷积。

## 三、训练

### 3.1 数据集

将[DUTS-TR](http://saliencydetection.net/duts)（5019张图像）、[DUTS-TE](http://saliencydetection.net/duts)（10553张图片）作为训练集（总共15572张图像），在其他数据集图像（共25615张）随机取1000张作为验证集。（包括[DUT-OMRON](http://saliencydetection.net/dut-omron/#org96c3bab)（5168张）、[ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)（1000张）、[HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html)（4447张）、[MSRA10K](https://mmcheng.net/msra10k/)（10000张）、[MSRA-B](https://mmcheng.net/msra10k/)（5000张））。

### 3.2 训练细节
在两张1080 TI上训练，批次大小(batch size)设置为16，每迭代2000次，保存一次模型，当训练整体损失（由side1-6以及side fuse分类的损失相加）小于0.5或者side fuse分类的损失小于0.05时停止训练。

对比以下两种训练方式：
- 加载修改后U2Net的模型（167.3M）做为初始化参数（修改卷积层的权重，将权重相邻的通道，两两切分作为一组，取平均值，得到一个通道，处理权重的所有通道，并将处理后的参数级联到一起。）
- 直接默认的初始化方式，进行训练。

如论文中所说，不需要使用图像分类中的任何预训练的模型，从头开始训练也可以达到好的效果。

## 四、部署方案

将pytorch下训练得到的pth模型，转为onnx模型，使用OpenCV中的dnn模块加载进行推理（其中对OpenCV进行重新编译，加入CUDA加速处理相关依赖，时间从1800ms左右，降到了120ms左右，在i5-7400 CPU @ 3.00GHz + 1050Ti的主机上）。