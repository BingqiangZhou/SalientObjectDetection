
#include "Utils.h"

// 概率图通过OTSU阈值化得到mask，最后得到image图像前景主体图像（背景为白色）
Mat Utils::GetImageByPredictResult(Mat image, Mat predict) {
	Mat outMask, outImage;
	Mat whiteImage(image.size(), image.type(), Scalar(255, 255, 255));
	threshold(predict, outMask, 0, 255, THRESH_OTSU);
	bitwise_and(image, image, outImage, outMask);
	bitwise_not(outMask, outMask);
	bitwise_or(outImage, whiteImage, outImage, outMask);
	return outImage;
}

// 将string类型的整形转为int类型
int Utils::StringToInt(string stringVal) {
	return atoi(stringVal.c_str());
}

// 修改fullPath文件路径后缀为newSuffixNetIncludeDot，当extStr不为""时将extStr添加到新的后缀之前，如*-result.png
string Utils::ChangeFilePathSuffix(string fullPath, string newSuffixNetIncludeDot, string extStr) {
	int index = fullPath.find_last_of('.');
	string newPath;
	if (extStr != "") {
		newPath = fullPath.substr(0, index) + extStr + '.' + newSuffixNetIncludeDot;
	}
	else
	{
		newPath = fullPath.substr(0, index + 1) + newSuffixNetIncludeDot;
	}
	return newPath;
}

// 获取imageDir文件夹下的imageType格式的图片路径
vector<string> Utils::GetImageList(string imageDir, string imageType) {
	cout << "get image list..." << endl;
	vector<string> imagePaths;
	glob(imageDir + "\\*." + imageType, imagePaths);
	return imagePaths;
}