#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>

int main()
{
    std::cout << "Hello World!" << std::endl;
    cv::Mat img;
    img = cv::imread("/home/davinci/Pictures/Screenshot.png", CV_LOAD_IMAGE_COLOR);
    cv::imshow("window", img);
    cv::waitKey();
    return 0;
}

