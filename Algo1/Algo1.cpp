#include<opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::duration_cast;

int main()
{
	auto algoStart = high_resolution_clock::now();
	//Load the image in grayscale
	Mat img = imread("C:\\Users\\MESH USER\\Downloads\\edgeflower.jpg", IMREAD_GRAYSCALE);
	if (img.empty())
	{
		cout << "Error: Could not load image" << std::endl;
		return -1;
	}

	//Create an output image to store the edges
	Mat edgeImage = Mat::zeros(img.size(), CV_8UC1);
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
	//Display Results
	const auto algoDuration = duration_cast<microseconds>(algoEnd - algoStart).count();
	cout << "Algorithm Process Time: " << algoDuration << " microseconds" << std::endl;
	imshow("Original Image", img);
	imshow("Edge Detected Image", edgeImage);
	waitKey(0);
	return 0;
}