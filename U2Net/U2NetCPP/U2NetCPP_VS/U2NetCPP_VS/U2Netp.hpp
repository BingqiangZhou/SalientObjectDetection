#pragma once

# include <opencv2/opencv.hpp>
# include <time.h>
# include <iostream>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace cuda;

class U2Netp
{
private:
	bool useGPU;
	int curImageHeight = 0, curImageWidth = 0;
	Size netInputSize;
	// Net LoadModelByONNX(string onnx_path); // ¼ÓÔØONNXÄ£ÐÍ
	Net LoadModel(string modelFilePath, string modelType = "onnx");
	Mat PreProcessImage(Mat image);
	Mat PostProcessImage(Mat image);
	Net net;
public:
	U2Netp(string modelPath, Size inputSize, bool useGPU = true, string modelFileType = "onnx");
	// ~U2Netp();

	Mat PredictByPath(string imagePath);
	Mat PredictByMat(Mat image);
};