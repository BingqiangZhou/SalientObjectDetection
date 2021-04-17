#pragma once

# include <opencv2/opencv.hpp>
# include <time.h>
# include <iostream>

using namespace cv;
using namespace std;
using namespace dnn;
using namespace cuda;

class U2Net
{
private:
	bool useGPU;
	int curImageHeight = 0, curImageWidth = 0;
	Size netInputSize;
	// Net LoadModelByONNX(string onnx_path); // ¼ÓÔØONNXÄ£ÐÍ
	Net LoadModel(string modelFilePath, string modelType = "onnx");
	Mat PreProcessImage(Mat image);
	Mat PostProcessImage(Mat predict);
	Net net;
public:
	U2Net(string modelPath, Size inputSize, bool useGPU = true, string modelFileType = "onnx");
	// ~U2Netp();
	//Mat PredictByPath(string imagePath);
	Mat PredictByMat(Mat image);
};