#include<opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <iostream>
#include <chrono>
#include <omp.h>
using namespace std;
using namespace cv;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::duration_cast;

Mat BaseAlgorithm(Mat& img, Mat& edgeImage)
{
	auto algoStart = high_resolution_clock::now();

	//Loop through each pixel (execute on 1 thread)
	for (int y = 0; y < img.rows - 1; y++) {
		for (int x = 0; x < img.cols - 1; x++) {
			
			//Extract intensity values of the 2x2 pixel block 
			int I00 = img.at<uchar>(y, x);
			int I01 = img.at<uchar>(y, x + 1);
			int I10 = img.at<uchar>(y + 1, x);
			int I11 = img.at<uchar>(y + 1, x + 1);
			
			//Apply Robert Cross operator
			int Gx = (I01 + I11) - (I00 + I10);
			int Gy = (I10 + I11) - (I00 + I01);
			
			//Calculate gradient magnitude
			int G = (int)sqrt((Gx * Gx) + (Gy * Gy));
			
			//Apply thresholding
			if (G > 255) G = 255;
			if (G < 0) G = 0;
			
			//Assign result to output image
			edgeImage.at<uchar>(y, x) = G;
		}
	}
	auto algoEnd = high_resolution_clock::now();
	const auto algoDuration = duration_cast<microseconds>(algoEnd - algoStart).count();
	cout << "Base Algorithm Process Time: " << algoDuration << " microseconds" << std::endl;
	return edgeImage;
}

Mat OpenMPAlgorithm(Mat& img, Mat& edgeImage)
{
	auto algoStart = high_resolution_clock::now();

	//Run the loop across multiple threads, automatically dividing rows among CPU cores
	#pragma omp parallel for schedule(static)
	for (int y = 0; y < img.rows - 1; y++) {
		for (int x = 0; x < img.cols - 1; x++) {
			
			//Pointers used for direct indexing, avoids repeated checking if pixel is within bounds

			//Start of row Y
			const uchar* rowPtr = img.ptr<uchar>(y);
			//Pointer to the start of the next row in the input image (needed for 2x2)
			const uchar* nextRowPtr = img.ptr<uchar>(y + 1);
			//Pointer to the start of the current row in the output image
			uchar* outPtr = edgeImage.ptr<uchar>(y);
			
			//Direct index via pointers
			int I00 = rowPtr[x]; //(Int I00 = y[x])
			int I01 = rowPtr[x + 1];
			int I10 = nextRowPtr[x];
			int I11 = nextRowPtr[x + 1];
			
			int Gx = (I01 + I11) - (I00 + I10);
			int Gy = (I10 + I11) - (I00 + I01);
			
			int G = (int)sqrt((Gx * Gx) + (Gy * Gy));
			
			if (G > 255) G = 255;
			if (G < 0) G = 0;
			outPtr[x] = static_cast<uchar>(G);
		}
	}
	auto algoEnd = high_resolution_clock::now();
	const auto algoDuration = duration_cast<microseconds>(algoEnd - algoStart).count();
	cout << "OpenMP Algorithm Process Time: " << algoDuration << " microseconds" << std::endl;
	cout << "Threads available: " << omp_get_max_threads() << endl;
	return edgeImage;
}

int main()
{
	Mat img = imread("C:\\Users\\MESH USER\\Downloads\\edgeflower.jpg", IMREAD_GRAYSCALE);
	if (img.empty())
	{
		cout << "Error: Could not load image" << std::endl;
		return -1;
	}
	Mat edgeImage = Mat::zeros(img.size(), CV_8UC1);

	imshow("Original Image", img);
	edgeImage = BaseAlgorithm(img, edgeImage);
	imshow("Edge Detected Image - Base", edgeImage);
	edgeImage = Mat::zeros(img.size(), CV_8UC1);
	edgeImage = OpenMPAlgorithm(img, edgeImage);
	imshow("Edge Detected Image - OpenMP", edgeImage);
	waitKey(0);
}
