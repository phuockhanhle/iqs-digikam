#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/dnn.hpp"
#include <filesystem>

using namespace std;
using namespace cv;
using namespace dnn;


const string IMAGE_PATH = "C:\\ML\\iqs-digikam\\inference\\inference_iqs\\candles.jpg";
const string MODEL_PATH = "C:\\ML\\iqs-digikam\\frozen_models\\simple_frozen_graph.pb";

int main()
{
	Mat img = imread(IMAGE_PATH);
	Net net = readNetFromTensorflow(MODEL_PATH);
	Mat blob = blobFromImage(img, 1, Size(224, 224), Scalar(0, 0, 0), false, false);
	net.setInput(blob);
	cv::Mat out = net.forward();
	std::cout << out;
}
