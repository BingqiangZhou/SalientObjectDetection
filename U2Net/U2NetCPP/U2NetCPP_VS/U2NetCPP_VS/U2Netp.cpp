
#include "U2Netp.hpp"

Net U2Netp::LoadModel(string modelFilePath, string modelType) {
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

Mat U2Netp::PreProcessImage(Mat image) {
	this->curImageHeight = image.rows;
	this->curImageWidth = image.cols;

	Scalar mean = Scalar(0.485 * 255.0, 0.456 * 255.0, 0.406* 255.0); // 123.675 116.28 103.53
	double scale = 1 / 255.0;
	Scalar std = Scalar(0.229, 0.224, 0.225);
	Mat imageBlob = blobFromImage(image, scale, this->netInputSize, mean);
	divide(imageBlob, std, imageBlob);
	return imageBlob;
}

Mat U2Netp::PostProcessImage(Mat image) {
	const float* data = reinterpret_cast<const float*>(image.data);
	Mat outImageFloat(this->netInputSize.width, this->netInputSize.height, CV_32FC1);
	for (int col = 0; col < this->netInputSize.width; col++) {
		for (int row = 0; row < this->netInputSize.height; row++) {
			outImageFloat.at<float>(col, row) = *data;
			data++;
		}
	}
	Mat outImage;
	convertScaleAbs(outImageFloat * 255, outImage);
	threshold(outImage, outImage, 0, 255, THRESH_OTSU);
	resize(outImage, outImage, Size(this->curImageWidth, this->curImageHeight));
	return outImage;
}

Mat U2Netp::PredictByPath(string imagePath) {
	clock_t start, end;

	Mat img = imread(imagePath);
	Mat imgNCHW = this->PreProcessImage(img);

	start = clock();
	Mat out = this->PredictByMat(imgNCHW);
	end = clock();

	out = PostProcessImage(out);

	cout << imagePath << ", speed " << end - start << "ms" << endl;
	return out;
}

Mat U2Netp::PredictByMat(Mat image) {

	this->net.setInput(image);
	Mat result = this->net.forward();
	return result;
}

U2Netp::U2Netp(string modelPath, Size inputSize, bool useGPU, string modelFileType)
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