#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;
using namespace std::chrono;

//KERNEL
//Each thread executes this function (each thread computes unique x,y pixel) (e.g. block idx 0, block dim 256, thread idx 0 = (0*256)+0 = pixel 0)
__global__ void EdgeDetectKernel(const unsigned char* input, unsigned char* output, int width, int height, int step)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global x pixel index 
	int y = blockIdx.y * blockDim.y + threadIdx.y; // Calculate global y pixel index 

    // Exit before leaving bounds
    if (x >= width - 1 || y >= height - 1) return;

    // Calculate pointers to current and next row
    const unsigned char* rowPtr = input + y * step; 
    const unsigned char* nextRowPtr = input + (y + 1) * step;
    unsigned char* outPtr = output + y * step; 

    // Read 2x2 neighborhood pixels
    int I00 = rowPtr[x];
    int I01 = rowPtr[x + 1];
    int I10 = nextRowPtr[x];
    int I11 = nextRowPtr[x + 1];

    // Robert Cross operator
    int Gx = (I01 + I11) - (I00 + I10);
    int Gy = (I10 + I11) - (I00 + I01);
    int G = (int)sqrtf((float)(Gx * Gx + Gy * Gy));

    // Apply thresholding
    if (G > 255) G = 255;
    if (G < 0) G = 0;

    outPtr[x] = static_cast<uchar>(G);
}

//MAIN FUNCTION
int main()
{
    Mat img = imread("C:\\Users\\MESH USER\\Downloads\\edgeflower.jpg", IMREAD_GRAYSCALE);
    if (img.empty())
    {
        cout << "Error: Could not load image" << endl;
        return -1;
    }

    int width = img.cols;
    int height = img.rows;
    int step = img.step; // bytes per row

    Mat edgeImage = Mat::zeros(img.size(), CV_8UC1); //allocate output image on CPU

    // Allocate GPU memory
    unsigned char* device_inputImage, * device_outputImage;
    cudaMalloc(&device_inputImage, height * step);
    cudaMalloc(&device_outputImage, height * step);

    // Copy image data from host to GPU
    cudaMemcpy(device_inputImage, img.data, height * step, cudaMemcpyHostToDevice);

    // Set kernel launch parameters
    dim3 blockSize(16, 16); // 256 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, //enough blocks to cover the image and handles non-divisible dimensions
		(height + blockSize.y - 1) / blockSize.y); 

    auto algoStart = high_resolution_clock::now();

    // Launch kernel on GPU
    EdgeDetectKernel << <gridSize, blockSize >> > (device_inputImage, device_outputImage, width, height, step); 

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    auto algoEnd = high_resolution_clock::now();
    const auto algoDuration = duration_cast<microseconds>(algoEnd - algoStart).count();

    // Copy output image back to CPU
    cudaMemcpy(edgeImage.data, device_outputImage, height * step, cudaMemcpyDeviceToHost);

    cout << "CUDA Algorithm Process Time: " << algoDuration << " microseconds" << endl;

    // Show results
    imshow("Original Image", img);
    imshow("Edge Detected Image - CUDA", edgeImage);
    waitKey(0);

    // Free GPU memory
    cudaFree(device_inputImage);
    cudaFree(device_outputImage);

    return 0;
}
