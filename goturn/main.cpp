#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/caffe.hpp>
#include <iostream>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <string>

#include"boundinbox.h"
#include"regressor.h"
#define INPUT_SIZE 227
#define NUM_CHANNELS 3


int drag = 0, select_flag = 0;

cv::Point point1, point2;
bool callback = false;

cv::Mat frame;
cv::Rect init;

void mouseHandler(int event, int x, int y, int flags, void* param)
{
    if (event == CV_EVENT_LBUTTONDOWN && !drag && !select_flag)
    {
        /* left button clicked. ROI selection begins */
        point1 = cv::Point(x, y);
        drag = 1;
    }

    if (event == CV_EVENT_MOUSEMOVE && drag && !select_flag)
    {
        /* mouse dragged. ROI being selected */
        cv::Mat img1 = frame.clone();
        point2 = cv::Point(x, y);
        cv::rectangle(img1, point1, point2, CV_RGB(255, 0, 0), 3, 8, 0);
        cv::imshow("or", img1);
    }

    if (event == CV_EVENT_LBUTTONUP && drag && !select_flag)
    {
        cv::Mat img2 = frame.clone();
        point2 = cv::Point(x, y);
        drag = 0;
        select_flag = 1;
        cv::imshow("or", img2);
        cv::Rect a(point1,point2);
        init = a;
        callback = true;
    }
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cout << "./run.bin model_file model_file video_file " << std::endl;
        return 0;
    }
    const std::string& model_file   = argv[1];
    const std::string& trained_file = argv[2];
    int gpu_id = 0;
    const bool do_train = false;


    cv::VideoCapture vc(argv[3]);
    vc >> frame;

    cv::imshow("or",frame);
    cv::setMouseCallback("or",mouseHandler,0);

    while(true)
    {
        if(callback)
        {
            vc >> frame;
            if( frame.empty() )
                    break;
            cv::imshow("or",frame);
            break;
        }

        cv::waitKey(5);
    }
    cv::Mat start = frame(init);
    cv::imshow("or",start);
    cv::waitKey();
    Regressor regressor(model_file, trained_file, gpu_id, do_train);
    while(true)
    {
        vc >> frame;
        cv::imshow("or",frame);
        cv::waitKey(5);

    }
    std::cout <<"Heelo World" << std::endl;
}
