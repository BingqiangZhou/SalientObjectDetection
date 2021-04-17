
#include "Utils.h"

// ����ͼͨ��OTSU��ֵ���õ�mask�����õ�imageͼ��ǰ������ͼ�񣨱���Ϊ��ɫ��
Mat Utils::GetImageByPredictResult(Mat image, Mat predict) {
	Mat outMask, outImage;
	Mat whiteImage(image.size(), image.type(), Scalar(255, 255, 255));
	threshold(predict, outMask, 0, 255, THRESH_OTSU);
	bitwise_and(image, image, outImage, outMask);
	bitwise_not(outMask, outMask);
	bitwise_or(outImage, whiteImage, outImage, outMask);
	return outImage;
}

// ��string���͵�����תΪint����
int Utils::StringToInt(string stringVal) {
	return atoi(stringVal.c_str());
}

// �޸�fullPath�ļ�·����׺ΪnewSuffixNetIncludeDot����extStr��Ϊ""ʱ��extStr��ӵ��µĺ�׺֮ǰ����*-result.png
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

// ��ȡimageDir�ļ����µ�imageType��ʽ��ͼƬ·��
vector<string> Utils::GetImageList(string imageDir, string imageType) {
	cout << "get image list..." << endl;
	vector<string> imagePaths;
	glob(imageDir + "\\*." + imageType, imagePaths);
	return imagePaths;
}