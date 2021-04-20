
#include "U2Net.hpp"

Net U2Net::LoadModel(string modelFilePath, string modelType) {
	try
	{
		cout << "begin loading model..." << endl;
		clock_t start, end;
		start = clock();
		if (modelType == "onnx") {
			this->net = readNetFromONNX(modelFilePath);
		}
		else {
			this->net = readNet(modelFilePath);
		}
		end = clock();
		cout << "model loaded, spend " << end - start << "ms" << endl;
		return this->net;
	}
	catch (const std::exception& e)
	{
		// cout << "current can't support this" << model_file_type << "type model file" << endl;
		std::cerr << e.what() << '\n';
		// this->~U2Netp();
		return Net();
	}
}

Mat U2Net::PreProcessImage(Mat image) {
	this->curImageHeight = image.rows;
	this->curImageWidth = image.cols;

	Scalar mean = Scalar(0.485 * 255.0, 0.456 * 255.0, 0.406* 255.0); // 123.675 116.28 103.53
	double scale = 1 / 255.0;
	Scalar std = Scalar(0.229, 0.224, 0.225);
	Mat imageBlob = blobFromImage(image, scale, this->netInputSize, mean);
	divide(imageBlob, std, imageBlob);
	return imageBlob;
}

Mat U2Net::PostProcessImage(Mat predict) {
	/*const float* data = reinterpret_cast<const float*>(predict.data);
	Mat outImageFloat(this->netInputSize.width, this->netInputSize.height, CV_32FC1);
	for (int col = 0; col < this->netInputSize.width; col++) {
		for (int row = 0; row < this->netInputSize.height; row++) {
			outImageFloat.at<float>(col, row) = *data;
			data++;
		}
	}*/
	/*vector<int> outSize{ this->netInputSize.width, this->netInputSize.height };
	Mat outImage = predict.reshape(1, outSize);*/
	Mat outImage = predict.reshape(1, this->netInputSize.height);
	convertScaleAbs(outImage * 255, outImage);
	resize(outImage, outImage, Size(this->curImageWidth, this->curImageHeight));
	return outImage;
}

//Mat U2Net::PredictByPath(string imagePath) {
//	clock_t start, end;
//
//	Mat img = imread(imagePath);
//	Mat imgNCHW = this->PreProcessImage(img);
//
//	start = clock();
//	Mat out = this->PredictByMat(imgNCHW);
//	end = clock();
//
//	
//
//	cout << imagePath << ", speed " << end - start << "ms" << endl;
//	return out;
//}

Mat U2Net::PredictByMat(Mat image) {

	clock_t start, end;
	start = clock();
	Mat imgNCHW = this->PreProcessImage(image);
	this->net.setInput(imgNCHW);
	Mat result = this->net.forward();
	end = clock();
	cout << "inference speed " << end - start << "ms" << endl;
	result = this->PostProcessImage(result);
	return result;
}

U2Net::U2Net(string modelPath, Size inputSize, bool useGPU, string modelFileType)
{
	this->netInputSize = inputSize;
	this->net = LoadModel(modelPath, modelFileType);
	if (getCudaEnabledDeviceCount() > 0 && useGPU) {
		cout << "GPU avaliable." << endl;
		this->useGPU = true;
		this->net.setPreferableBackend(DNN_BACKEND_CUDA);
		this->net.setPreferableTarget(DNN_TARGET_CUDA);
	}
	else {
		this->useGPU = false;
		cout << "GPU not avaliable, use CPU." << endl;
	}
}