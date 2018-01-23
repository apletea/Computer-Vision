#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

void procces(const cv::Mat & in, cv::Mat & mask, std::vector<std::pair<int,int>> pts)
{
    std::vector<std::pair<int,int>> newPts;
    std::vector<int> tmp = {1,-1, 2, -2};
    for(auto point : pts)
    {
        if (mask.at<bool>(point.first,point.second))
        {
//            std::cout << "YYYYYYYYYYYYYYYYYy" << std::endl;
            continue;
        }
        for (int i  = 0; i < tmp.size(); ++i)
        {
            if (point.first + tmp[i]/2 >= in.cols+1 || point.second + tmp[i]%2 >= in.rows+1 || point.first + tmp[i]/2 <= 0 || point.second + tmp[i]%2 <= 0)
                continue;
            int delta = std::abs(( int(in.at<uchar>(point.first,point.second)) - int(in.at<uchar>(point.first + tmp[i]/2,point.second + tmp[i]%2) )));
            mask.at<uchar>({point.second,point.first}) = 255;
            if (delta < 10 &&  !mask.at<bool>(point.first + tmp[i]/2,point.second + tmp[i]%2))
            {
                std::cout << "delta =" << delta << std::endl;
//                std::cout << point.first << " " << point.second << " " << point.first + tmp[i]/2 << " " << point.second + tmp[i]%2  << std::endl;
                newPts.push_back({point.first + tmp[i]/2,point.second + tmp[i]%2});
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
    cv::Mat img = cv::imread("/home/davinci/work/unet_segment/train/1.png",0);
    cv::Mat mask;
    cv::threshold(img,mask,140,255,cv::THRESH_BINARY);
    cv::imshow("img",img);
    cv::imshow("mask",mask);
    std::vector<std::pair<int,int>> pts;
    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            if (mask.at<bool>(i,j))
                pts.push_back({i,j});
        }
    }
    cv::Mat newMask(cv::Size(img.cols,img.rows),mask.type());
    procces(img,newMask,pts);
    cv::imshow("newmask",newMask);
    cv::waitKey();
    return 0;
}
