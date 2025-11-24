#pragma once
#include <vector>
#include <string>
#include<opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void SaveCSV(const vector<long>& durations, string fileName);
Mat LoadImageGrayscale(string filePath);
Mat LoadEdgeImage(Mat baseImage);
void DisplayImage(string imgName, Mat img);
void PrintAverageAcrossBlocks(vector<long> timings, int numBlocks);
void PrintCompletedBlock(int blockNum, int timingsPerBlock, int blockTime);