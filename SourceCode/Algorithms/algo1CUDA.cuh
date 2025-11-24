#include <cuda_runtime.h>
#include<opencv2/opencv.hpp>
using namespace cv;

__global__ void EdgeDetectKernel(const unsigned char* input, unsigned char* output, int width, int height, int step);
Mat RunKernel(Mat img, Mat edgeImage, int timingBlocks, int timingsPerBlock);