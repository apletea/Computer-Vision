#include <iostream>              //std::cout
#include <opencv2/highgui.hpp>   //cv::namedWindow, cv::imshow

#include <vector>                //std::vector
#include <algorithm>             //std::nth_element, std::sort
#include <cmath>                 //std::abs, std::round

#include <opencv2/core/mat.hpp>  //cv::Mat
#include <opencv2/imgcodecs.hpp> //cv::imread
#include <opencv2/imgproc.hpp>   //cv::findContours, cv::contourArea, cv::convexHull

class Nude
{
public:
  bool
    is_nude(cv::Mat mat,cv::Mat matGS)
    {
      const size_t kTOTAL_PIXEL_COUNT = mat.total();
            size_t   SKIN_PIXEL_COUNT = 0;

      for(size_t i = 0 ; i < mat.rows ; ++i)
        for(size_t j = 0 ; j < mat.cols ; ++j)
          if(is_skin(mat.at<cv::Vec3b>(i, j)))
          {
            ++SKIN_PIXEL_COUNT;
            matGS.at<uchar>(i, j) = 255;
          }

      const double kSKIN_REL_PIXEL_COUNT = SKIN_PIXEL_COUNT/((double) kTOTAL_PIXEL_COUNT);

      if(kSKIN_REL_PIXEL_COUNT < 0.15) return false;

      std::vector<std::vector<cv::Point> > contour;
      std::vector<cv::Vec4i>               hierarchy;
      cv::findContours(matGS, contour, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

      n_elements(contour.begin(), contour.end(), 3,
        [](const std::vector<cv::Point>& i, const std::vector<cv::Point>& j)
          { return cv::contourArea(i) > cv::contourArea(j); });

      const size_t kROI_1_PIXEL_COUNT = cv::contourArea(contour[0]);
      const size_t kROI_2_PIXEL_COUNT = cv::contourArea(contour[1]);
      const size_t kROI_3_PIXEL_COUNT = cv::contourArea(contour[2]);

      std::vector<std::vector<cv::Point> > cvx_hull(3);
      for(int i = 0 ; i < cvx_hull.size() ; ++i)
        cv::convexHull(cv::Mat(contour[i]), cvx_hull[i]);

      const size_t kCVX_HULL_1_PIXEL_COUNT = cv::contourArea(cvx_hull[0]);
      const size_t kCVX_HULL_2_PIXEL_COUNT = cv::contourArea(cvx_hull[1]);
      const size_t kCVX_HULL_3_PIXEL_COUNT = cv::contourArea(cvx_hull[2]);

      if((kROI_1_PIXEL_COUNT/((double) SKIN_PIXEL_COUNT)        < 0.35   &&
          kROI_2_PIXEL_COUNT/((double) SKIN_PIXEL_COUNT)        < 0.30   &&
          kROI_3_PIXEL_COUNT/((double) SKIN_PIXEL_COUNT)        < 0.30)  ||
          kROI_1_PIXEL_COUNT/((double) SKIN_PIXEL_COUNT)        < 0.45   ||
         (kSKIN_REL_PIXEL_COUNT                                 < 0.30   &&
          kROI_1_PIXEL_COUNT/((double) kCVX_HULL_1_PIXEL_COUNT) < 0.55   &&
          kROI_2_PIXEL_COUNT/((double) kCVX_HULL_2_PIXEL_COUNT) < 0.55   &&
          kROI_3_PIXEL_COUNT/((double) kCVX_HULL_3_PIXEL_COUNT) < 0.55))
        return false;

      return true;
    }

private:
  bool
    is_skin(cv::Vec3b BGR)
    {
      const unsigned char kB = BGR[0];
      const unsigned char kG = BGR[1];
      const unsigned char kR = BGR[2];

      return (kR > 95 &&
              kG > 40 &&
              kB > 20 &&
              kR > kG &&
              kR > kB &&
              max(kR, kG, kB) - min(kR, kG, kB) > 15 &&
              std::abs(kR - kG) > 15);
    }

  template<typename T>
    T
      min(const T& a, const T& b, const T& c)
      { return a < b ? a < c ? a : c : b < c ? b : c; }

  template<typename T>
    T
      max(const T& a, const T& b, const T& c)
      { return a > b ? a > c ? a : c : b > c ? b : c; }

  template<typename RandomAccessIterator,
           typename BinaryPredicate>
    void
      n_elements(RandomAccessIterator begin,
                 RandomAccessIterator end,
                 size_t               n,
                 BinaryPredicate      comp)
      {
        std::nth_element(begin, begin + n, end, comp);
        std::sort(begin, begin + n, comp);
      }
};


bool isNude(cv::Mat mat, cv::Mat gray )
{
    Nude o;
    return o.is_nude(mat,gray);
}
