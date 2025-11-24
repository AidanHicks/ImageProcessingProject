#include "helpers.h"
#include "algo1.h"
#include "algo1CUDA.cuh"

int main() {

	Mat baseImage = LoadImageGrayscale("C:\\Users\\MESH USER\\Desktop\\CS\\SampleImages\\edgeflower.jpg");
    
	Mat baseEdgeImage = LoadEdgeImage(baseImage);
	baseEdgeImage = BaseAlgorithm(baseImage, baseEdgeImage, 2, 10000);

	Mat openmpEdgeImage = LoadEdgeImage(baseImage);
	openmpEdgeImage = OpenMPAlgorithm(baseImage, openmpEdgeImage, 2, 10000);

	Mat cudaEdgeImage = LoadEdgeImage(baseImage);
	cudaEdgeImage = RunKernel(baseEdgeImage, cudaEdgeImage, 2, 10000);
	
	
	DisplayImage("Base Image", baseImage);
	DisplayImage("Base Algorithm", baseEdgeImage);
	DisplayImage("OpenMP Algorithm", openmpEdgeImage);
	DisplayImage("CUDA Algoritm", cudaEdgeImage);
	waitKey(0);
}