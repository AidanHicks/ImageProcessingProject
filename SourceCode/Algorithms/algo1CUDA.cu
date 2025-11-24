#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "helpers.h"
#include <vector>
#include <chrono>
using namespace std;
using namespace std::chrono;
using namespace cv;

__global__ void EdgeDetectKernel(const unsigned char* input, unsigned char* output, int width, int height, int step)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 

    if (x >= width - 1 || y >= height - 1) return;

    const unsigned char* rowPtr = input + y * step;
    const unsigned char* nextRowPtr = input + (y + 1) * step;
    unsigned char* outPtr = output + y * step;

    int I00 = rowPtr[x];
    int I01 = rowPtr[x + 1];
    int I10 = nextRowPtr[x];
    int I11 = nextRowPtr[x + 1];

    int Gx = (I01 + I11) - (I00 + I10);
    int Gy = (I10 + I11) - (I00 + I01);
    int G = (int)sqrtf((float)(Gx * Gx + Gy * Gy));

    if (G > 255) G = 255;
    if (G < 0) G = 0;

    outPtr[x] = static_cast<uchar>(G);
}

Mat RunKernel(Mat img, Mat edgeImage, int timingBlocks, int timingsPerBlock) {
    cout << "CUDA algorithm begin...\n";

    int width = img.cols;
    int height = img.rows;
    int step = img.step;

    unsigned char* device_inputImage, * device_outputImage;
    cudaMalloc(&device_inputImage, height * step);
    cudaMalloc(&device_outputImage, height * step);

    cudaMemcpy(device_inputImage, img.data, height * step, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16); 
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
        (height + blockSize.y - 1) / blockSize.y);

    vector<long> timings;

    for (int i = 0; i < timingBlocks; i++) {
        long blockTotalTime = 0;

        for (int j = 0; j < timingsPerBlock; j++) {

            auto start = high_resolution_clock::now();
            EdgeDetectKernel <<<gridSize, blockSize >>> (device_inputImage, device_outputImage, width, height, step);
            cudaDeviceSynchronize();
            auto end = high_resolution_clock::now();
            auto algoDuration = duration_cast<microseconds>(end - start).count();
            blockTotalTime += algoDuration;
        }
        long blockAvg = blockTotalTime / timingsPerBlock;
        timings.push_back(blockAvg);
        PrintCompletedBlock(i, timingsPerBlock, blockTotalTime);
    }
    SaveCSV(timings, "CUDATimings");
    PrintAverageAcrossBlocks(timings, timingBlocks);
    cudaMemcpy(edgeImage.data, device_outputImage, height * step, cudaMemcpyDeviceToHost);
    return edgeImage;
}

