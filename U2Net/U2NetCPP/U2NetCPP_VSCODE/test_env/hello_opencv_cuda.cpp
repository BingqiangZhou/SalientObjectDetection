#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace cuda;

int main(){

    cout << getCudaEnabledDeviceCount() << endl;
    return 0;
}
