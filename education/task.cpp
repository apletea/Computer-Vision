#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <bits/stdc++.h>
std::set<std::pair<int,int>> checked;

void procces(const cv::Mat & in, cv::Mat & mask, std::vector<std::pair<int,int>> pts)
{
    std::vector<std::pair<int,int>> newPts;
    std::vector<int> tmp = {1,-1, 2, -2};
    for(auto point : pts)
    {
//        if (mask.at<bool>())
        for (int i  = 0; i < tmp.size(); ++i)
        {
            double delta =( int(in.at<uchar>(point.first,point.second) - in.at<uchar>(point.first + tmp[i]/2,point.second + tmp[i]%2) ));
            if (delta < 5 && checked.find({point.first + tmp[i]/2,point.second + tmp[i]%2})==checked.end())
            {
                std::cout << "delta =" << delta << std::endl;
                std::cout << point.first << " " << point.second << " " << point.first + tmp[i]/2 << " " << point.second + tmp[i]%2  << std::endl;
                mask.at<uchar>({point.first + tmp[i]/2,point.second + tmp[i]%2}) = 255;
                newPts.push_back({point.first + tmp[i]/2,point.second + tmp[i]%2});
                checked.insert({point.first + tmp[i]/2,point.second + tmp[i]%2});

            }
        }
    }
    cv::imshow("tmp",mask);
    cv::waitKey(10);
    std::cout << "newPts size=" <<newPts.size() << std::endl;
    if (!newPts.empty())
        procces(in,mask,newPts);

}
int main(int argc, char *argv[])
{
    std::cout << "Hello" << std::endl;
    cv::Mat img = cv::imread("/home/please/work/unet_segment/train/1.png",0);
    cv::Mat mask;
    cv::threshold(img,mask,135,255,cv::THRESH_BINARY);
    std::vector<std::pair<int,int>> pts;
    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            if (mask.at<double>(i,j))
            {
                pts.push_back({i,j});
                checked.insert({i,j});
            }
        }
    }
    procces(img,mask,pts);
    cv::imshow("mask",mask);
    cv::imshow("im",img);

    cv::waitKey();
    return 0;
}
