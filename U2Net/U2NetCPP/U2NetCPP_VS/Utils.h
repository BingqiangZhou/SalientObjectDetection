#pragma once

# include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Utils
{
public:
	
	// 概率图通过OTSU阈值化得到mask，最后得到image图像前景主体图像（背景为白色）
	static Mat GetImageByPredictResult(Mat image, Mat predict);
	
	// 将string类型的整形转为int类型
	static int StringToInt(string stringVal);

	// 修改fullPath文件路径后缀为newSuffixNetIncludeDot，当extStr不为""时将extStr添加到新的后缀之前，如*-result.png
	static string ChangeFilePathSuffix(string fullPath, string newSuffixNetIncludeDot, string extStr = "");

	// 获取imageDir文件夹下的imageType格式的图片路径
	static vector<string> GetImageList(string imageDir, string imageType = "jpg");
};
