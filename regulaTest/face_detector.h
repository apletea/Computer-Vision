#ifndef TEST_TASK_LIBRARY_H
#define TEST_TASK_LIBRARY_H

#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

void proccesImg(const std::string & imgName, const std::string & folder);
void findEyesAndMouth(const cv::Mat & grayImg, const std::vector<cv::Rect> & faces,std::vector<std::vector<cv::Rect>> & ans);
void clearAns(std::vector<std::vector<cv::Rect>> & ans);
void saveToJson(std::vector<std::string>, std::string folderPath);
std::vector<std::vector<cv::Rect>> detectFaces(cv::Mat img);

#if defined(linux) || defined(__linux) || defined(__linux__)
cv::CascadeClassifier faceDetect("./data/haarcascade_frontalface_default.xml");
cv::CascadeClassifier eyeDetect("./data/haarcascade_eye_tree_eyeglasses.xml");
cv::CascadeClassifier mouthDetect("./data/Mouth.xml");
#else
cv::CascadeClassifier faceDetect("data\haarcascade_frontalface_default.xml");
cv::CascadeClassifier eyeDetect("data\haarcascade_eye_tree_eyeglasses.xml");
cv::CascadeClassifier mouthDetect("data\Mouth.xml");
#endif
#endif