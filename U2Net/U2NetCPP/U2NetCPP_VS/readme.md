# U2NetCPP_VS 环境以及相关说明

## 一、 开发环境及运行环境

### 1.1 开发环境
- Windows (20H2)： 10.0.19042.928
- Microsoft Visual Studio Community 2017： 15.9.34
- Microsoft .NET Framework： 4.8.04084
- CUDA：11.2.142
- cuDNN：8.0.5
- OpenCV：4.5.1
- CMake：3.20.0

### 1.2 运行环境

- 使用CUDA加速需要GPU算力大于3.5
- 如果有遇到“MSVSP140.dll丢失相关的错误”，请到[The latest supported Visual C++ downloads](https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0)下载 [vc_redist.x64.exe](https://aka.ms/vs/16/release/vc_redist.x64.exe)

## 二、U2NetCPP_VS 相关说明

## 2.1 配置文件

使用不同的U2Net ONNX模型，只需要修改配置文件config.ini即可。

```ini
[Path]
ONNXModel = ./onnx_models/u2netp.onnx
TestImages = ./test_images
ImageExt = jpg
[ModelSetting]
InputHeight = 400
InputWidth = 400
UseGPU = 1
```

section [Path]：
- ONNXModel onnx：模型文件路径
- TestImages： 测试图片文件夹（不支持图片路径，可以将图片放在文件夹下，并将文件夹路径配置在此处）
- ImageExt：测试图片文件后缀

section [ModelSetting]：
- InputHeight：ONNX模型输入图像的高
- InputWidth：ONNX模型输入图像的宽
- UseGPU：是否使用GPU (CUDA)加速，设置为大于0的值为使用CUDA加速

## 2.2 程序处理过程

### 2.1 预处理

- 图像归一化处理：均值 (0.485, 0.456, 0.406), 标准差 (0.229, 0.224, 0.225)

### 2.2 后处理

- 直接输出预测前景概率图（*.png）
- 对概率图使用OTSU算法二值化得到mask，输出背景为白色的分割结果图（*-result.png）