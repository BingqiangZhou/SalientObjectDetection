#include <fstream>

#include "U2Net.hpp"
#include "Utils.h"
#include "inipp.h" // 来自Nuget下载inipp v1.0.12，与 https://github.com/mcmtroffaes/inipp 不同

int main(int argc, char* argv[]) {
	
	// 读取并解析配置文件参数
	inipp::Ini<char> ini;
	ifstream iniFile("config.ini");
	ini.parse(iniFile);

	// 路径参数
	string modelPath = ini.sections["Path"]["ONNXModel"]; // onnx 模型路径
	string imageDir = ini.sections["Path"]["TestImageDir"]; // 测试图片文件夹
	string imageExt = ini.sections["Path"]["ImageExt"]; // 测试图片后缀

	// 模型参数
	int height = Utils::StringToInt(ini.sections["ModelSetting"]["InputHeight"]); // onnx模型输入图像的高
	int width = Utils::StringToInt(ini.sections["ModelSetting"]["InputHeight"]); // onnx模型输入图像的宽
	Size inputSize = Size(width, height);
	bool useGPU = Utils::StringToInt(ini.sections["ModelSetting"]["UseGPU"].c_str()); //  是否使用GPU加速

	// 初始化 U2Net
	U2Net u2Net = U2Net(modelPath, inputSize, useGPU, "onnx"); 
	
	// 获取图片路径列表，并循环处理图像
	vector<string> imageList = Utils::GetImageList(imageDir, imageExt);
	if (imageList.size() <= 0) {
		cout << "not image, exiting..." << endl;
	}
	else {
		for (string imagePath : imageList) {
			cout << "begin process:" << imagePath << endl;

			// 读取图片
			Mat img = imread(imagePath);

			// 预测
			Mat out = u2Net.PredictByMat(img);

			// 保存预测结果
			string newImagePath = Utils::ChangeFilePathSuffix(imagePath, "png");
			imwrite(newImagePath, out);

			// 保存图像 
			newImagePath = Utils::ChangeFilePathSuffix(imagePath, "png", "-result");
			out = Utils::GetImageByPredictResult(img, out);
			imwrite(newImagePath, out);

			cout << "process finished." << endl;
			cout << endl;
		}
		cout << "all image has been process finished." << endl;
	}
	system("pause");
	return 0;
}

