#include "helpers.h"
#include "algo1.h"
#include <chrono>
#include <omp.h>
using namespace std;
using namespace cv;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::duration_cast;

Mat BaseAlgorithm(Mat& img, Mat& edgeImage, int timingBlocks, int timingsPerBlock) {
	cout << "Base algorithm begin...\n";
	vector<long> timings;

	for (int i = 0; i < timingBlocks; i++) {
		long blockTotalTime = 0;
		
		for (int j = 0; j < timingsPerBlock; j++) {
			auto algoStart = high_resolution_clock::now();
			
			for (int y = 0; y < img.rows - 1; y++) {
				for (int x = 0; x < img.cols - 1; x++) {

					int I00 = img.at<uchar>(y, x);
					int I01 = img.at<uchar>(y, x + 1);
					int I10 = img.at<uchar>(y + 1, x);
					int I11 = img.at<uchar>(y + 1, x + 1);

					int Gx = (I01 + I11) - (I00 + I10);
					int Gy = (I10 + I11) - (I00 + I01);

					int G = (int)sqrt((Gx * Gx) + (Gy * Gy));

					if (G > 255) G = 255;
					if (G < 0) G = 0;

					volatile int temp = G;

					edgeImage.at<uchar>(y, x) = G;
				}
			}
			auto algoEnd = high_resolution_clock::now();
			auto algoDuration = duration_cast<microseconds>(algoEnd - algoStart).count();
			blockTotalTime += algoDuration;
		}
		double blockAvg = blockTotalTime / timingsPerBlock;
		timings.push_back(blockAvg);
		PrintCompletedBlock(i, timingsPerBlock, blockTotalTime);
	}
	SaveCSV(timings, "BaseTimingsOptiOFF");
	PrintAverageAcrossBlocks(timings, timingBlocks);
	return edgeImage;
}

Mat OpenMPAlgorithm(Mat& img, Mat& edgeImage, int timingBlocks, int timingsPerBlock) {
	cout << "OpenMP algorithm begin...\n";
	vector<long> timings;
	
	for (int i = 0; i < timingBlocks; i++) {
		long blockTotalTime = 0;

		for (int j = 0; j < timingsPerBlock; j++) {
			auto algoStart = high_resolution_clock::now();

			#pragma omp parallel for schedule(static)
			for (int y = 0; y < img.rows - 1; y++) {
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

					volatile int temp = G;

					outPtr[x] = (uchar)G;
				}
			}
			auto algoEnd = high_resolution_clock::now();
			auto algoDuration = duration_cast<microseconds>(algoEnd - algoStart).count();
			blockTotalTime += algoDuration;
		}
		double blockAvg = blockTotalTime / timingsPerBlock;
		timings.push_back(blockAvg);
		PrintCompletedBlock(i, timingsPerBlock, blockTotalTime);
	}
	SaveCSV(timings, "OpenMPTimingsOptiOFF");
	PrintAverageAcrossBlocks(timings, timingBlocks);
	return edgeImage;
}