#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>


struct Pt
{
    int x;
    int y;

    Pt(int x, int y)
    {
        this->x = x;
        this->y = y;
    }
};

void proc2(const cv::Mat & in, cv::Mat & mask, std::vector<Pt>& pts)
{
    std::vector<Pt> newPts;
    std::vector<Pt> offsets{ Pt(0,-1), Pt(-1,0), Pt(1,0), Pt(0, 1)};

    auto thr = 3;
    for (auto& point : pts)
    {
        for (auto& offset : offsets)
        {
            auto col = point.x + offset.x;
            auto row = point.y + offset.y;
            if (col < 0 || col >= in.cols || row < 0 || row >= in.rows)
                continue;
            auto m = mask.data + mask.cols * row + col;
            if (*m != 0)
                continue;

            auto val0 = in.datastart + in.cols * point.y + point.x;
            auto val1 = in.datastart + in.cols * row + col;
            auto delta = std::abs(((int)(*val1) - (int)(*val0)));
            if (delta > thr)
                continue;
            *m = 255;
            newPts.emplace_back(Pt(col, row));
        }
    }
    cv::imshow("mask",mask);
    cv::waitKey(1);
    if (!newPts.empty())
        proc2(in, mask, newPts);
}

void procces(const cv::Mat & in, cv::Mat & mask, std::vector<std::pair<int,int>> pts)
{
    std::vector<std::pair<int,int>> newPts;
    std::vector<int> tmp = {1,-1, 2, -2};

    for(auto& point : pts)
    {

        for (auto& i : tmp)
        {
            if (point.first + i/2 >= in.cols+1 || point.second +i%2 >= in.rows+1 || point.first + i/2 <= 0 || point.second + i%2 <= 0)
                continue;
            int delta = std::abs(( int(in.at<uchar>(point.first,point.second)) - int(in.at<uchar>(point.first + tmp[i]/2,point.second + tmp[i]%2) )));
            mask.at<uchar>({point.second,point.first}) = 255;
            if (delta < 10 &&  !mask.at<uchar>(point.first + tmp[i]/2,point.second + tmp[i]%2))
                newPts.push_back({point.first + tmp[i]/2,point.second + tmp[i]%2});
        }
    }
    if (!newPts.empty())
        procces(in,mask,newPts);
    return;

}

cv::Mat forOpening = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3,3));
cv::Mat forClosening = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3,3));


#define CLOSENING 3
#define OPENING 2

void makeAndSaveMask(const cv::Mat & img, const  std::string & folder,const  std::string & name)
{
    cv::Mat mask;
    cv::threshold(img,mask,125,255,cv::THRESH_BINARY);
    std::vector<std::pair<int,int>> pts;
    std::vector<Pt> pts2;


    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            if (mask.at<uchar>(i,j))
            {

                pts.push_back({i,j});
                pts2.emplace_back(Pt(j,i));
            }
        }
    }
    proc2(img,mask,pts2);
    cv::Mat res, resMask;
    cv::bitwise_or(img,mask,resMask);
//    cv::morphologyEx(mask,resMask,OPENING,forOpening);
    cv::morphologyEx(mask,mask,CLOSENING,forClosening);
    //morphClosing(mask,mask, 3);
    cv::bitwise_or(img,mask,res);
    cv::imshow("bitwiseBefore",resMask);
    cv::imshow("bitwise",res);
    cv::imshow("img",img);
//    cv::waitKey();
    cv::imwrite(folder+name,mask);
    return;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cout << "./bin fileWithNames.txt pathToFolder pathToSaveFolder" << std::endl;
        return 0;
    }
    std::ifstream in(argv[1]);
    std::string baseFolder(argv[2]);
    std::string pathTosave(argv[3]);
    std::string fileName;
    while (!in.eof())
    {
//        cv::waitKey(150);
        in >> fileName;
        cv::Mat img = cv::imread(baseFolder + fileName,0);
        makeAndSaveMask(img,pathTosave,fileName);
    }
    return 0;
//    cv::Mat img = cv::imread("/home/please/work/unet_segment/train/2.png",0);
//    makeAndSaveMask(img,"/home/please/work/unet_segment/train_mask/","2.png");
}

