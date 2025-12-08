#include "helpers.h"
#include "algo1.h"
#include "algo1CUDA.cuh"

Mat CowEdgeSemanticSegmentation(const Mat& edgeInput)
{
    Mat edges;

    if (edgeInput.channels() == 3)
        cvtColor(edgeInput, edges, COLOR_BGR2GRAY);
    else
        edges = edgeInput.clone();

    int rows = edges.rows;
    int cols = edges.cols;

    threshold(edges, edges, 15, 255, THRESH_BINARY);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(edges, edges, MORPH_OPEN, kernel);

    Mat labels, stats, centroids;
    int n = connectedComponentsWithStats(edges, labels, stats, centroids, 8);

    cout << "Connected components: " << n << endl;

    Mat output(rows, cols, CV_8UC3, Scalar(0, 255, 0));

    for (int i = 1; i < n; i++)
    {
        int area = stats.at<int>(i, CC_STAT_AREA);

        if (area < 80)
            continue;

        for (int y = 0; y < rows; y++)
        {
            int* lbl = labels.ptr<int>(y);
            Vec3b* out = output.ptr<Vec3b>(y);

            for (int x = 0; x < cols; x++)
            {
                if (lbl[x] == i)
                    out[x] = Vec3b(0, 0, 255);
            }
        }
    }

    return output;
}

Mat FlowerSegmentationFromEdges(const Mat& edgeInput)
{
    Mat edges;

    // Ensure grayscale
    if (edgeInput.channels() == 3)
        cvtColor(edgeInput, edges, COLOR_BGR2GRAY);
    else
        edges = edgeInput.clone();

    int rows = edges.rows;
    int cols = edges.cols;

    //Keep only strong edges
    threshold(edges, edges, 40, 255, THRESH_BINARY);

    //Thicken edges to fully close petals
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
    dilate(edges, edges, kernel, Point(-1, -1), 2);

    //Create barrier image
    Mat barrier;
    bitwise_not(edges, barrier);

    //4. Flood - fill from border(background)
    Mat flood = barrier.clone();
    floodFill(flood, Point(0, 0), Scalar(0));

    //Enclosed regions = flower
    Mat flowerMask;
    bitwise_not(flood, flowerMask);

    //Clean small holes
    morphologyEx(flowerMask, flowerMask, MORPH_CLOSE,
        getStructuringElement(MORPH_ELLIPSE, Size(9, 9)));

    //Semantic Output
    Mat output(rows, cols, CV_8UC3, Scalar(0, 255, 0)); // background = green

    for (int y = 0; y < rows; y++)
    {
        const uchar* m = flowerMask.ptr<uchar>(y);
        Vec3b* out = output.ptr<Vec3b>(y);

        for (int x = 0; x < cols; x++)
        {
            if (m[x] > 0)
                out[x] = Vec3b(0, 0, 255); // flower = red
        }
    }

    return output;
}

int main() {

    //Cow
    Mat baseCowImage = LoadImageGrayscale("C:\\Users\\MESH USER\\Desktop\\CS\\SampleImages\\animals.jpg");
    
    Mat cowEdgeDetected = Mat::zeros(baseCowImage.size(), CV_8UC1);
    cowEdgeDetected = BaseAlgorithm(baseCowImage, cowEdgeDetected, 1, 1);

    Mat segmentedCow = CowEdgeSemanticSegmentation(cowEdgeDetected);

    //Flower
    Mat baseFlowerImage = LoadImageGrayscale("C:\\Users\\MESH USER\\Desktop\\CS\\SampleImages\\edgeflower.jpg");
    
    Mat flowerEdgeDetected = Mat::zeros(baseFlowerImage.size(), CV_8UC1);
    flowerEdgeDetected = BaseAlgorithm(baseFlowerImage, flowerEdgeDetected, 1, 1);

    Mat segmentedFlower = FlowerSegmentationFromEdges(flowerEdgeDetected);

    //Display
    imshow("Base Cow", baseCowImage);
    imshow("Edge-Detected Cow", cowEdgeDetected);
    imshow("Segmented Cow", segmentedCow);
    imshow("Base Flower", baseFlowerImage);
    imshow("Edge-Detected Flower", flowerEdgeDetected);
    imshow("Segmented Flower", segmentedFlower);
    waitKey(0);
}	


