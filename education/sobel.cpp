#include <iostream>
#include "opencv2/core/core.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
    cv::Mat img = cv::imread("/home/apletea/Downloads/maxresdefault.jpg"), src_gray;
    int scale = 1;
    int delta = 0;
    int deptth = CV_16S;


    cv::GaussianBlur(img,img,cv::Size(3,3), 0 ,0, cv::BORDER_DEFAULT);

    cv::cvtColor(img, src_gray,CV_BGR2GRAY);


    cv::Mat x,y,res;

    cv::Sobel(src_gray,x,deptth,1,0,3,scale,delta,cv::BORDER_DEFAULT);
    cv::Sobel(src_gray,y,deptth,0,1,3,scale,delta,cv::BORDER_DEFAULT);

    cv::convertScaleAbs(x,x);
    cv::convertScaleAbs(y,y);

    cv::addWeighted(x,0.5 ,y,0.5,0, res);



    cv::waitKey(1);
    cv::imshow("res",res);
    cv::waitKey();
    return 0;
}
