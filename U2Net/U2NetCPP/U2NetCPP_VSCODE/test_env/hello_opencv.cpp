#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
    Mat img = imread("E:\\Workspaces\\zbq\\projects\\U-2-Net-master\\test_image\\01.jpg");
    imshow("image", img);
    waitKey(0);
    return 0;
}