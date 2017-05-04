#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d.hpp>
#include<fstream>


using namespace cv;
using namespace cv::xfeatures2d;

using namespace std;

int nfeatures=500;
float scaleFactor=1.2f;
int nlevels=8;
int edgeThreshold=15; // Changed default (31);
int firstLevel=0;
int WTA_K=2;
int scoreType=ORB::HARRIS_SCORE;
int patchSize=31;
int fastThreshold=20;


class vek{
public:
    double x;
    double y;

    vek(){
        x = 0;
        y = 0;
    }

    void addTuVek(KeyPoint & p1, KeyPoint & p2){
        double dx = p2.pt.x - p1.pt.y;
        double dy = p2.pt.y - p1.pt.y;
        x = x + dx;
        y = y + dy;
    }
};

void add_to_distance(KeyPoint & p1, KeyPoint & p2, double & distance){
    double dr = sqrt((p2.pt.x - p1.pt.x) * (p2.pt.x - p1.pt.x) + (p2.pt.y - p1.pt.y) * (p2.pt.y - p1.pt.y));
    distance += dr;
}

void find_cos(vek & v1, vek & v2, double & ans){
    double scalar_product = v1.x * v2.x + v1.y * v2.y;
    double module_1 = sqrt(v1.x * v1.x + v1.y * v1.y);
    double module_2 = sqrt(v2.x * v2.x + v1.y * v1.y);
    ans = scalar_product / (module_1 * module_2);
}

void fiendCoeficients(vector<KeyPoint> & keypoints_1, vector<KeyPoint> & keypoints_2, vector<DMatch> & good_matches){
    double distance_1 = 0;
    double distance_2 = 0;
    vek* vek1 = new vek();
    vek* vek2 = new vek();

    if (good_matches.size() < 2)
        cout << "here is no mathc";
    for (int i = 1 ; i < good_matches.size(); ++i){
        vek1->addTuVek(keypoints_1[good_matches[0].queryIdx], keypoints_1[good_matches[i].queryIdx]);
        vek2->addTuVek(keypoints_2[good_matches[0].trainIdx], keypoints_2[good_matches[i].trainIdx]);
        add_to_distance(keypoints_1[good_matches[0].queryIdx], keypoints_1[good_matches[i].queryIdx], distance_1);
        add_to_distance(keypoints_2[good_matches[0].trainIdx], keypoints_2[good_matches[i].trainIdx], distance_2);
    }
    double dsize = distance_1/distance_2;
    cout << "distance changed in:" << dsize << endl;
    double cos = 0;
    vek2->x *= dsize;
    vek2->y *= dsize;
    find_cos(*vek1, *vek2, cos);
    cout << "cos =" << cos << endl;
    cout << "angle changes on :" << acos(cos)*180.0 / 3.14 << endl;
}


void procesShit(Mat & img_1, Mat & img_2, std::vector<KeyPoint> & keypoints_1, Mat & descriptors_1){

    std::vector<KeyPoint>  keypoints_2;
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
    detector->detect(img_2,keypoints_2);
    Mat descriptors_2;
    detector->detectAndCompute(img_2,Mat(),keypoints_2,descriptors_2);

    FlannBasedMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    double max_dist = 0; double min_dist = 100;

    for( int i = 0; i < descriptors_1.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    std::vector< DMatch > good_matches;
    for( int i = 0; i < descriptors_1.rows; i++ )
    { if( matches[i].distance <= max(2*min_dist, 0.02) )
        { good_matches.push_back( matches[i]); }
    }


    Mat img_matches;
    drawMatches( img_1, keypoints_1, img_2, keypoints_2,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    fiendCoeficients(keypoints_1, keypoints_2, good_matches);
    imshow( "Good Matches", img_matches );


}

void matchingKeyPoints( vector<KeyPoint> & ans, Mat & descriptor1, Mat & descriptor2,vector<KeyPoint> & current){
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptor1,descriptor2,matches);
    double max_dist = 0; double min_dist = 100;

    for( int i = 0; i < descriptor1.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    std::vector< DMatch > good_matches;
    for( int i = 0; i < descriptor1.rows; i++ )
    { if( matches[i].distance <= max(2*min_dist, 0.02) )
        { good_matches.push_back( matches[i]); }
    }
    for (int i = 0; i < good_matches.size(); ++i){
        ans.push_back(current[good_matches[i].queryIdx]);
    }

}

int main(){
    try {
        cv::VideoCapture capWebcam(0);

        cv::Mat imgOr;
        cv::Mat currentImage;
        cv::Mat imgWithPoints2;
        for (int i = 0 ; i < 10; i ++){
            capWebcam.read(imgOr);
        }
        cv::cvtColor(imgOr, imgWithPoints2, CV_BGR2GRAY);

        char charCheckForEscKey = 0;
        std::vector<KeyPoint>  keypoints1;


        cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
        int i =100;


        detector->detect(imgWithPoints2,keypoints1);
        Mat descriptor;
        detector->detectAndCompute(imgWithPoints2,Mat(),keypoints1,descriptor);

        for (int i = 0; i  < 15; ++i){
            Mat tmp,imgToProcces;
            capWebcam.read(tmp);
            cv::cvtColor(tmp, imgToProcces, CV_BGR2GRAY);
            Mat descriptorTMP;
            vector<KeyPoint> KPtoProcces;
            detector->detect(imgToProcces, KPtoProcces);
            detector->detectAndCompute(imgToProcces, Mat(),KPtoProcces,descriptorTMP);
            vector<KeyPoint> newKp;
            matchingKeyPoints(newKp, descriptor,descriptorTMP,keypoints1);
            keypoints1 = newKp;
            Mat tmpDsc;
            detector->detectAndCompute(imgToProcces,Mat(),keypoints1,tmpDsc);
            descriptor = tmpDsc;
        }

        while (true){
            capWebcam.read(currentImage);
            Mat imtGray;
            cv::cvtColor(currentImage,imtGray,CV_BGR2GRAY);

            procesShit(imgWithPoints2, imtGray,keypoints1, descriptor);

            charCheckForEscKey = cv::waitKey(1);
        }

    }
    catch (std::exception e ){
        std::cout<<"exception throw: "<<e.what() <<std::endl;
    }

}