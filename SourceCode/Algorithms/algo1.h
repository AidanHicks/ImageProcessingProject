#pragma once
#include<opencv2/opencv.hpp>
using namespace cv;

Mat BaseAlgorithm(Mat& img, Mat& edgeImage, int timingBlocks, int timingsPerBlock);
Mat OpenMPAlgorithm(Mat& img, Mat& edgeImage, int timingBlocks, int timingsPerBlock);
Mat BaseMultiClassColorSegmentation(Mat& img, Mat& segImage, int timingBlocks, int timingsPerBlock, int classes);