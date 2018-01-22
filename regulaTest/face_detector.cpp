#include "face_detector.h"
#include "ThreadPool.h"

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


void proccesImg(const std::string & imgName, const std::string & folder)
{
    std::string imgPath = folder + imgName;
    cv::Mat img = cv::imread(imgPath);

    ThreadPool pool(4);
    auto result = pool.enqueue(detectFaces,imgPath);
    auto boxes = result.get();


    for (int i = 0; i < boxes.size(); i+=3)
    {
        std::string faceImgName = imgName;
        faceImgName.insert(imgName.size()-4,"_"+std::to_string(i));
        faceImgName.replace(faceImgName.end()-4,faceImgName.end(),".jpg");
        cv::Mat faceROI = img(boxes[i][0]);
        cv::Mat flipedFace;
        cv::flip(faceROI,flipedFace,1);
        cv::imwrite(folder + faceImgName,flipedFace);
    }

}

std::vector<std::vector<cv::Rect>> detectFaces(const std::string & imgPath)
{

    cv::Mat gray = cv::imread(imgPath,0);
    std::vector<cv::Rect> faces;
    std::vector<std::vector<cv::Rect>> ans;

    faceDetect.detectMultiScale(gray,faces,1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

    findEyesAndMouth(gray,faces,ans);
    clearAns(ans);

    if (faces.empty())
    {
        std::cout << "Theare no detections" << std::endl;
    }
    else
    {
        std::cout << "Found " << ans.size()/3 << " faces in file: " << imgPath << std::endl;
    }
    ans[1][0].width
    return ans;
}

void findEyesAndMouth(const cv::Mat & grayImg, const  std::vector<cv::Rect> & faces, std::vector<std::vector<cv::Rect>> & ans)
{
    for (int i = 0; i < faces.size(); ++i)
    {
        std::vector<cv::Rect> eyes;
        std::vector<cv::Rect> mouth;
        ans.push_back({faces[i]});

        cv::Mat faceROI = grayImg(faces[i]);

        eyeDetect.detectMultiScale(faceROI,eyes,1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
        mouthDetect.detectMultiScale(faceROI,mouth,1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

        ans.push_back(eyes);
        ans.push_back(mouth);
    }
}

void clearAns(std::vector<std::vector<cv::Rect>> & ans)
{
    int ansSize = 0;
    for (int i = 0; i < ans.size(); i+=3)
    {
        if (ans[i+1].size() == 2 && ans[i+2].size() ==1 )
            ansSize++;
        else
        {
            ans[i] = ans[i+3];
            ans[i+1] = ans[i+4];
            ans[i+2] = ans[i+5];
            i-=3;
        }
    }
    ans.resize(ansSize*3);
}

void saveToJson(std::vector<std::string>, std::string folderPath)
{

}



