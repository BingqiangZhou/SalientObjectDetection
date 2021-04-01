#include "U2Netp.hpp"

vector<string> GetImageList(string imageDir, string imageType="jpg"){
    cout << "get image list..." << endl;
    vector<string> imagePaths;
    glob(imageDir+"\\*."+imageType, imagePaths);
    return imagePaths;
}

string ChangeFilePathSuffix(string fullPath, string newSuffixNetIncludeDot){
    int index = fullPath.find_last_of('.');
    string newPath = fullPath.substr(0, index+1) + newSuffixNetIncludeDot;
    return newPath;
}

int main(int argc, char* argv[]){
    bool useGPU = true;
    if (argc > 1){
        string argString(argv[1]);
        transform(argString.begin(), argString.end(), argString.begin(), ::tolower);
        if(argString == "cpu"){
            useGPU = false;
        }
    }
    
    string modelPath = "E:\\Workspaces\\zbq\\projects\\U-2-Net-master\\saved_models\\u2netp.onnx";
    string imageDir = "E:\\Workspaces\\zbq\\projects\\U-2-Net-master\\test_image";
    Size inputSize = Size(400, 400);
    U2Netp u2Netp = U2Netp(modelPath, inputSize, useGPU, "onnx");
    vector<string> imageList = GetImageList(imageDir);
    for(string imagePath : imageList){
        Mat out = u2Netp.PredictByPath(imagePath);
        string newImagePath = ChangeFilePathSuffix(imagePath, "png");
        imwrite(newImagePath, out);
        waitKey(0);
    }
    return 0;
}