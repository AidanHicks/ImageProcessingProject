#include<opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <fstream>
using namespace std;
using namespace cv;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::duration_cast;
int numTimings = 10000;

void saveCSV(const std::vector<long long>& durations, string fileName) {
	std::ofstream file(fileName);
	if (!file.is_open()) return;

	for (size_t i = 0; i < durations.size(); i++) {
		file << i << "," << durations[i] << "\n";
	}

	file.close();
}

Mat BaseAlgorithm(Mat& img, Mat& edgeImage)
{
	vector<long long> durations(numTimings);

	for (int i = 0; i < numTimings; i++) {
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
		durations[i] = algoDuration;
	}
	//cout << "Base Algorithm Process Time: " << algoDuration << " microseconds" << std::endl;
	saveCSV(durations, "C:\\Users\\MESH USER\\Desktop\\MainProj\\Algo1\\Timings\\BaseTimings.csv");
	return edgeImage;
}

Mat OpenMPAlgorithm(Mat& img, Mat& edgeImage)
{
	vector<long long> durations(numTimings);

	for (int i = 0; i < numTimings; i++) {
		auto algoStart = high_resolution_clock::now();

		//Run the loop across multiple threads, automatically dividing rows among CPU cores
		#pragma omp parallel for schedule(static)
		for (int y = 0; y < img.rows - 1; y++) {

			//Pointers used for direct indexing, avoids repeated checking if pixel is within bounds
			//Changed pointer location, no need for it in x loop (faster timings)
			const uchar* rowPtr = img.ptr<uchar>(y);
			const uchar* nextRowPtr = img.ptr<uchar>(y + 1);
			uchar* outPtr = edgeImage.ptr<uchar>(y);

			for (int x = 0; x < img.cols - 1; x++) {

				int I00 = rowPtr[x];
				int I01 = rowPtr[x + 1];
				int I10 = nextRowPtr[x];
				int I11 = nextRowPtr[x + 1];

				int Gx = (I01 + I11) - (I00 + I10);
				int Gy = (I10 + I11) - (I00 + I01);

				int G = sqrt(Gx * Gx + Gy * Gy);

				if (G > 255) G = 255;
				if (G < 0) G = 0;

				outPtr[x] = (uchar)G;
			}
		}
		auto algoEnd = high_resolution_clock::now();
		const auto algoDuration = duration_cast<microseconds>(algoEnd - algoStart).count();
		durations[i] = algoDuration;
	}
	saveCSV(durations, "C:\\Users\\MESH USER\\Desktop\\MainProj\\Algo1\\Timings\\OpenMPTimings.csv");
	//cout << "OpenMP Algorithm Process Time: " << algoDuration << " microseconds" << std::endl;
	//cout << "Threads available: " << omp_get_max_threads() << endl;
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
	cout << "Algorithms Finished";
	waitKey(0);
}
