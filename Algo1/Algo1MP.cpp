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

int main()
{
	
	//Load the image in grayscale
	Mat img = imread("C:\\Users\\MESH USER\\Downloads\\edgeflower.jpg", IMREAD_GRAYSCALE);
	if (img.empty())
	{
		cout << "Error: Could not load image" << std::endl;
		return -1;
	}


	Mat edgeImage = Mat::zeros(img.size(), CV_8UC1);
	auto algoStart = high_resolution_clock::now();

	//Run the loop across multiple threads, automatically dividing rows among CPU cores)
	#pragma omp parallel for schedule(static)
	for (int y = 0; y < img.rows - 1; y++) {
		for (int x = 0; x < img.cols - 1; x++) {
			
			//Pointers used for direct indexing, avoids repeated checking if pixel is within bounds
			const uchar* rowPtr = img.ptr<uchar>(y);
			const uchar* nextRowPtr = img.ptr<uchar>(y + 1);
			uchar* outPtr = edgeImage.ptr<uchar>(y);

			int I00 = rowPtr[x];
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
	cout << "Threads available: " << omp_get_max_threads() << endl;
	cout << "Algorithm Process Time: " << algoDuration << " microseconds" << std::endl;
	imshow("Original Image", img);
	imshow("Edge Detected Image", edgeImage);
	waitKey(0);
	return 0;
}