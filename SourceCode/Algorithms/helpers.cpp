#include "helpers.h"
#include<opencv2/opencv.hpp>
#include <fstream>
#include <chrono>
using namespace cv;
using namespace std;

void SaveCSV(const vector<long>& durations, string fileName) {
	ofstream file("C:\\Users\\MESH USER\\Desktop\\CS\\ImageProcessing\\TimingProcessing\\RawTimings\\" + fileName + ".csv");
	if (!file.is_open()) return;

	for (size_t i = 0; i < durations.size(); i++) {
		file << i << "," << durations[i] << "\n";
	}

	file.close();
}

Mat LoadImageGrayscale(string filePath) {
	Mat img = imread(filePath, IMREAD_GRAYSCALE);
	return img;
}

Mat LoadImageColor(string filePath) {
	Mat img = imread(filePath);
	return img;
}

Mat LoadEdgeImage(Mat baseImage) {
	Mat edgeImage = Mat::zeros(baseImage.size(), CV_8UC1);
	return edgeImage;
}

Mat LoadSegImage(Mat baseImage) {
	Mat edgeImage = Mat::zeros(baseImage.size(), CV_8UC3);
	return edgeImage;
}

void DisplayImage(string imgName, Mat img) {
	imshow(imgName, img);
}

void PrintAverageAcrossBlocks(vector<long> timings, int numBlocks) {
    long total = 0;
	for (int t : timings) {
		total += t;
	}
	long average = total / numBlocks;
	
	cout << numBlocks << " blocks executed. Average execution time accross all blocks: " << average << " microseconds.\n";
}

void PrintCompletedBlock(int blockNum, int timingsPerBlock, int blockTime) {
	cout << "Completed block " << blockNum << ". Total time for " << timingsPerBlock << " executions: " << blockTime << " microseconds.\n";
}

