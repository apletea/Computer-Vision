
int main(int argc, char ** argv)
{
    cv::Mat img = cv::imread("/home/apletea/work/JPEGImages/000001.jpg");
    cv::waitKey(1);
    cv::Mat warp_rotate_dst;
    cv::Point center = cv::Point(img.cols/2,img.rows/2);
    char k;
    double scale = 2;
    while(true)
    {
        k = cv::waitKey(1);
        std::cout << k << std::endl;
        if (k=='+')
        {
            k = 'a';
            scale+=0.5;
        }
        else if (k == '-')
        {
            k = 'a';
            scale-=0.5;
        }
        cv::Mat rot_mat = cv::getRotationMatrix2D(center,0,scale);
        cv::warpAffine(img,warp_rotate_dst,rot_mat,(cv::Size(300,300)));
        cv::waitKey(2);
        warp_rotate_dst.copyTo(img(cv::Rect(img.cols/2,img.rows/2,warp_rotate_dst.cols, warp_rotate_dst.rows)));
        cv::imshow("test",img);
    }
}
