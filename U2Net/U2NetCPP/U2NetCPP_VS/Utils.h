#pragma once

# include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Utils
{
public:
	
	// ����ͼͨ��OTSU��ֵ���õ�mask�����õ�imageͼ��ǰ������ͼ�񣨱���Ϊ��ɫ��
	static Mat GetImageByPredictResult(Mat image, Mat predict);
	
	// ��string���͵�����תΪint����
	static int StringToInt(string stringVal);

	// �޸�fullPath�ļ�·����׺ΪnewSuffixNetIncludeDot����extStr��Ϊ""ʱ��extStr��ӵ��µĺ�׺֮ǰ����*-result.png
	static string ChangeFilePathSuffix(string fullPath, string newSuffixNetIncludeDot, string extStr = "");

	// ��ȡimageDir�ļ����µ�imageType��ʽ��ͼƬ·��
	static vector<string> GetImageList(string imageDir, string imageType = "jpg");
};
