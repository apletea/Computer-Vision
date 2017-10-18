

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>
#include <cv.h>
#include <highgui.h>
#include "opencv2/legacy/legacy.hpp"
#include <math.h>



/*********************************************************************************************
 * compile with:
 * g++ -O3 -o opticalFlow opticalFlow.cpp -std=c++11 `pkg-config --cflags --libs opencv`
*********************************************************************************************/


/*
 *                                SPECIFICATION
 * This is a program for OpenCV mini project assignment, designed and implemented by
 * Yonghui Rao and Yujia Liu.
 * The programe aims to compare different opical flow algorithms with the sample video
 * "vtest.avi" from OpenCV.
 * Run it with ./opticalFlow, no parameter needs to be specified, the program will try
 * to load "vtest.avi" under the same directory, and user could specify an algorithm.
 *  
 *
 *
 *                                   NOTE
 *  #1 This program is based on OpenCV 2.x, some APIs are deprecated on OpenCV3.x. So it may 
 *     fail if you compile it with OpenCV 3.x.
 *  #2 You must make sure the "vtest.avi" exists under the same directory with the executable.
 *  #3 If you choose Lucas-Kanade algorithm, you can press "r" to generate tracking points
 *     automatically.
 *  #4 The Simple Flow algorithm is very slow, that may because it is designed for parallelled
 *     GPUs but this program will run it on CPU. 
 */


/*
 * 				        RESULTS
 *
 *	Method					TimeCost(seconds)		Accuracy
 * calcOpticalFlowPyrLK(Lucas-Kanade)		17.319	Relatively 		Low
 * calcOpticalFlowFarneback(Gunnar Farneback)	155.909	Relatively 		High
 * CalcOpticalFlowHS(Horn-Schunck)		137.088	Relatively 		High
 * cvCalcOpticalFlowBM(Block Matching)		283.032	Relatively		Low
 * calcOpticalFlowSF(Simple Flow Algorithm)	9 seconds per frame		Very high
 */



using namespace cv;  
using namespace std;  
      
#define UNKNOWN_FLOW_THRESH 1e9  
      


// Create colors for marking motion

void makecolorwheel(vector<Scalar> &colorwheel)  
{  
    int RY = 15;  
    int YG = 6;  
    int GC = 4;  
    int CB = 11;  
    int BM = 13;  
    int MR = 6;  
      
    int i;  
      
    for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255,       255*i/RY,     0));  
    for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255-255*i/YG, 255,       0));  
    for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0,         255,      255*i/GC));  
    for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0,         255-255*i/CB, 255));  
    for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255*i/BM,      0,        255));  
    for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255,       0,        255-255*i/MR));  
}  
     

// Mark motion with specific color
void motionToColor(Mat flow, Mat &color)  
{  
    if (color.empty())  
        color.create(flow.rows, flow.cols, CV_8UC3);  
      
    static vector<Scalar> colorwheel; //Scalar r,g,b  
    if (colorwheel.empty())  
        makecolorwheel(colorwheel);  
    
        // determine motion range:  
    float maxrad = -1;  
      
        // Find max flow to normalize fx and fy  
    for (int i= 0; i < flow.rows; ++i)   
    {  
        for (int j = 0; j < flow.cols; ++j)   
        {  
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);  
            float fx = flow_at_point[0];  
            float fy = flow_at_point[1];  
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
                continue;  
            float rad = sqrt(fx * fx + fy * fy);  
            maxrad = maxrad > rad ? maxrad : rad;  
        }  
    }  
      
    for (int i= 0; i < flow.rows; ++i)   
    {  
        for (int j = 0; j < flow.cols; ++j)   
        {  
            uchar *data = color.data + color.step[0] * i + color.step[1] * j;  
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);  
      
            float fx = flow_at_point[0] / maxrad;  
            float fy = flow_at_point[1] / maxrad;  
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
            {  
                data[0] = data[1] = data[2] = 0;  
                continue;  
            }  
            float rad = sqrt(fx * fx + fy * fy);  
      
            float angle = atan2(-fy, -fx) / CV_PI;  
            float fk = (angle + 1.0) / 2.0 * (colorwheel.size()-1);  
            int k0 = (int)fk;  
            int k1 = (k0 + 1) % colorwheel.size();  
            float f = fk - k0;  
            for (int b = 0; b < 3; b++)   
            {  
                float col0 = colorwheel[k0][b] / 255.0;  
                float col1 = colorwheel[k1][b] / 255.0;  
                float col = (1 - f) * col0 + f * col1;  
                if (rad <= 1)  
                    col = 1 - rad * (1 - col); // increase saturation with radius  
                else  
                    col *= .75; // out of range  
                data[2 - b] = (int)(255.0 * col);  
            }  
        }  
    }  
}  
      
      
// Gunnar Farneback algorithm     

int fb()   
{  
    VideoCapture cap;  
    cap.open("vtest.avi");  
    //cap.open("test_02.wmv");  
    
    if( !cap.isOpened() )  
        return -1;  
    
    Mat prevgray, gray, flow, cflow, frame;  
    namedWindow("Gunnar Farneback", 1);  
    
    Mat motion2color;  
    int frameCnt = cap.get(CV_CAP_PROP_FRAME_COUNT);
    for(int i = 0; i < frameCnt; i ++)  
    {  
        double t = (double)cvGetTickCount();  
    
        cap >> frame;  
        cvtColor(frame, gray, CV_BGR2GRAY);  
        imshow("original", frame);  
    
        if( prevgray.data )  
        {  
            
            calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0); 
            motionToColor(flow, motion2color);  
            imshow("Gunnar Farneback", motion2color);  
        }  
        if(waitKey(10)>=0)  
            break;  
        std::swap(prevgray, gray);  

    } 
    destroyAllWindows();
    return 0;  
} 



// Horn-Schunck algorithm
int hs()  
{   
    IplImage * frame=NULL;
    int i, j, dx, dy, rows, cols;
    IplImage *src_img1=NULL, *src_img2=NULL, *dst_img1=NULL, *dst_img2=NULL;
    CvMat *velx, *vely;
    CvTermCriteria criteria;

    CvCapture *input_video = cvCreateFileCapture("vtest.avi");//read vtest.avi
    if (input_video == NULL)
    {

        /* Either the video didn't exist OR it uses a codec OpenCV
        * doesn't support.
        */
        fprintf(stderr, "Error: Can't open video.\n");
        return -1; 
    }
    cvQueryFrame( input_video ); 
    cvSetCaptureProperty( input_video, CV_CAP_PROP_POS_AVI_RATIO, 1. );
    /* Now that we're at the end, read the AVI position in frames */
    number_of_frames = (int) cvGetCaptureProperty( input_video, CV_CAP_PROP_POS_FRAMES ); 
    cvSetCaptureProperty( input_video, CV_CAP_PROP_POS_FRAMES, 0. ); 

    long current_frame = 0;

    for(int index = 0; index < number_of_frames; index++)
    {
       // cout<<"rao index :"<<index << " total :" << number_of_frames << endl;
        cvSetCaptureProperty( input_video, CV_CAP_PROP_POS_FRAMES, current_frame );
        frame = cvQueryFrame( input_video );
        dst_img2 =(IplImage *) cvClone (frame);
        if (frame == NULL)
        {
            /* Why did we get a NULL frame? We shouldn't be at the end. */
            //fprintf(stderr, "Error: Hmm. The end came sooner than we thought.\n");
            return -1;
        }
; 
        IplImage* frame1_1C = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);//创建目标图像
        cvCvtColor(frame,frame1_1C,CV_BGR2GRAY);//cvCvtColor(src,des,CV_BGR2GRAY)
        frame = cvQueryFrame( input_video );
        if (frame == NULL)
        {
            fprintf(stderr, "Error: Hmm. The end came sooner than we thought.\n");
            return -1;
        }
        IplImage* frame2_1C = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);//创建目标图像
        cvCvtColor(frame,frame2_1C,CV_BGR2GRAY);//cvCvtColor(src,des,CV_BGR2GRAY)
        cols = frame1_1C->width;
        rows = frame1_1C->height;
        velx = cvCreateMat (rows, cols, CV_32FC1);
        vely = cvCreateMat (rows, cols, CV_32FC1);
        cvSetZero (velx);
        cvSetZero (vely);
        criteria = cvTermCriteria (CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.1);
        cvCalcOpticalFlowHS (frame1_1C, frame2_1C, 0, velx, vely, 100.0, criteria);


        for (i = 0; i < cols; i += 5) {
            for (j = 0; j < rows; j += 5) {
                dx = (int) cvGetReal2D (velx, j, i);
                dy = (int) cvGetReal2D (vely, j, i);
                if ((abs(dx) > 2 && abs(dx) < 15)&&(abs(dy) > 2 && abs(dy) < 15))
                {
                    if(dx < 0)
                        dx -= 5;
                    else 
                        dx +=5;
                    if(dy < 0)
                        dy -= 5;
                    else 
                        dy +=5;
                    //printf("raoyonghui %d  %d\n", dx, dy);
                    cvLine (dst_img2, cvPoint (i, j), cvPoint (i + dx, j + dy), CV_RGB (255, 0, 0), 1, CV_AA, 0);
                }
            }
        }
        cvReleaseImage (&frame1_1C);
        cvReleaseImage (&frame2_1C);
        cvNamedWindow ("Horn-Schunck", 1);
        cvShowImage ("Horn-Schunck", dst_img2);
        cvReleaseImage (&dst_img2);
        
        cvReleaseMat(&velx);
        cvReleaseMat(&vely);
        int key_pressed;
        key_pressed = cvWaitKey(10);
        if (key_pressed == 'b' || key_pressed == 'B')
            break;
       
        /* Don't run past the front/end of the AVI. */
        //if (current_frame < 0) current_frame = 0;
        if (current_frame >= number_of_frames - 1)
            break;
        current_frame=current_frame+1;
        
    }
    cvDestroyWindow ("Horn-Schunck");
    cvReleaseImage (&src_img1);
    cvReleaseImage (&src_img2);
    cvReleaseImage (&dst_img1);
    cvReleaseImage (&dst_img2);
    cvReleaseMat (&velx);
    cvReleaseMat (&vely);
}





// Vars for block matching
IplImage *image[2];
IplImage *source[2];
CvSize size;
IplImage *velocityX, velocityY;


// Block matching algorithm

int bm()  
{
    /* Create an object that decodes the input video stream. */
	CvCapture *input_video = cvCaptureFromFile(
		"vtest.avi");
	if (input_video == NULL)
	{
		fprintf(stderr, "Error: Can't open video.\n");
		return -1;
	}

    CvSize frame_size;
	frame_size.height = (int) cvGetCaptureProperty(input_video, CV_CAP_PROP_FRAME_HEIGHT );
	frame_size.width = (int) cvGetCaptureProperty(input_video, CV_CAP_PROP_FRAME_WIDTH );

    	/* Determine the number of frames in the AVI. */
	long number_of_frames;
	/* Go to the end of the AVI (ie: the fraction is "1") */
	cvSetCaptureProperty(input_video, CV_CAP_PROP_POS_AVI_RATIO, 1.);
	/* Now that we're at the end, read the AVI position in frames */
	number_of_frames = ( int ) cvGetCaptureProperty (input_video, CV_CAP_PROP_POS_FRAMES);
	/* Return to the beginning */
	cvSetCaptureProperty (input_video, CV_CAP_PROP_POS_FRAMES, 0.);

    IplImage *source[ 2 ], *image[ 2 ];
    

    int current_frame = 0;
    while(current_frame < number_of_frames - 1)
    {
        source[ 0 ] = cvCreateImage (frame_size, IPL_DEPTH_8U, 1);
        source[ 1 ] = cvCreateImage (frame_size, IPL_DEPTH_8U, 1);
        cvSetCaptureProperty( input_video, CV_CAP_PROP_POS_FRAMES, current_frame);

        IplImage * frame = cvQueryFrame( input_video );

        if (frame == NULL) {
            /* Why did we get a NULL frame?  We shouldn't be at the end. */
            fprintf(stderr, "Error: Hmm. The end came sooner than we thought.\n");
            return -1;
        }

        image[0] = cvCloneImage(frame);
        
        ++current_frame;
        cvSetCaptureProperty( input_video, CV_CAP_PROP_POS_FRAMES, current_frame);

        frame = cvQueryFrame( input_video );
        if (frame == NULL) {
            fprintf(stderr,"Error: Hmm. The end came sooner than we thought.\n");
            return -1;
        }

        //cvConvertImage(frame, image[ 1 ], CV_CVTIMG_FLIP);
        image[1] = cvCloneImage(frame);

        cvCvtColor (image[ 0 ], source[ 0 ], CV_BGR2GRAY);
        cvCvtColor (image[ 1 ], source[ 1 ], CV_BGR2GRAY);

        cvNamedWindow("Original", CV_WINDOW_AUTOSIZE);
        cvShowImage("Original", frame);

        CvSize shiftSize, maxrange, blockSize;
        IplImage *velx, *vely;
         
        int blockWidth = 7;
        int shiftWidth = 3;
        blockSize.height =blockSize.width = blockWidth;
        shiftSize.height =shiftSize.width = shiftWidth;
        maxrange.height = maxrange.width = 12;
        CvSize velSize =
        {
            (source[0]->width - blockWidth + shiftWidth)/shiftWidth,
            (source[0]->height - blockWidth + shiftWidth)/shiftWidth
        };
        velx = cvCreateImage(velSize,IPL_DEPTH_32F,1);
        vely = cvCreateImage(velSize,IPL_DEPTH_32F,1); 
        cvSetZero(velx);
        cvSetZero(vely);
        cvCalcOpticalFlowBM(source[0], source[1], blockSize, shiftSize, maxrange, 1 , velx, vely);

        cvNamedWindow ("Horizontal", 0);
        cvShowImage ("Horizontal", velx);

        cvNamedWindow ("Vertical", 0);
        cvShowImage ("Vertical", vely);

        cvWaitKey( 10 );
        cvReleaseImage( &source[ 1 ] );
        cvReleaseImage( &source[ 0 ] );
        cvReleaseImage( &image[ 1 ] );
        cvReleaseImage( &image[ 0 ] );
        cvReleaseImage( &vely );
        cvReleaseImage( &velx );
    }
    cvDestroyWindow( "Original" );
    cvDestroyWindow( "Horizontal" );
    cvDestroyWindow( "Vertical" );
}


// Lucas-Kanade algorithm
int lucas() 
{
    cout<<"press r to generate trackinngn point" << endl;

    VideoCapture cap;
    TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
    Size subPixWinSize(10,10), winSize(31,31);

    const int MAX_COUNT = 500;
    bool needToInit = false;
    bool nightMode = false;


    cap.open("vtest.avi");
    if( !cap.isOpened() )
    {
        cout << "Could not initialize capturing...\n";
        return 0;
    }
    
    int frameCnt = cap.get(CV_CAP_PROP_FRAME_COUNT);

    namedWindow( "Lucas-Kanade", 1 );

    Mat gray, prevGray, image;
    vector<Point2f> points[2];
    //int i = 0;
    for(int i = 0; i < frameCnt; i ++)
    {
        Mat frame;
        cap >> frame;
        if( frame.empty() )
            break;

        frame.copyTo(image);
        cvtColor(image, gray, COLOR_BGR2GRAY);

        if( nightMode )
            image = Scalar::all(0);

        if( needToInit )
        {
            // automatic initialization
            goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
            cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);
            addRemovePt = false;
        }
        else if( !points[0].empty() )
        {
            vector<uchar> status;
            vector<float> err;
            if(prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                                 3, termcrit, 0, 0.001);
            size_t i, k;
            for( i = k = 0; i < points[1].size(); i++ )
            {
                if( addRemovePt )
                {
                    if( norm(point - points[1][i]) <= 5 )
                    {
                        addRemovePt = false;
                        continue;
                    }
                }

                if( !status[i] )
                    continue;

                points[1][k++] = points[1][i];
                circle( image, points[1][i], 3, Scalar(0,255,0), -1, 8);
            }
            points[1].resize(k);
        }

        if( addRemovePt && points[1].size() < (size_t)MAX_COUNT )
        {
            vector<Point2f> tmp;
            tmp.push_back(point);
            cornerSubPix( gray, tmp, winSize, cvSize(-1,-1), termcrit);
            points[1].push_back(tmp[0]);
            addRemovePt = false;
        }

        needToInit = false;
        imshow("Lucas-Kanade", image);


        char c = (char)waitKey(10);
        
        if( c == 'q' )
            break;
        switch( c )
        {
        case 'r':
            needToInit = true;
            break;
        case 'c':
            points[0].clear();
            points[1].clear();
            break;
        case 'n':
            nightMode = !nightMode;
            break;
        }

        std::swap(points[1], points[0]);
        cv::swap(prevGray, gray);
   
    }
    destroyAllWindows();
    return 0;
}




#define UNKNOWN_FLOW_THRESH 1e9  
 
// Simple Flow algorithm
int sf()   
{  
    VideoCapture cap;  
    cap.open("vtest.avi");  
    
    if( !cap.isOpened() )  
        return -1;  
    
    Mat img, prevImg, frame;  
    
    Mat motion2color; 
    int frameCnt = cap.get(CV_CAP_PROP_FRAME_COUNT); 
    
    for(int i = 0; i < frameCnt; i ++)  
    {  
        double t = (double)cvGetTickCount();  
    
        cap >> frame;  
        imshow("original", frame);  
        img = frame.clone();
        namedWindow("Simple Flow Algorithm");
        if( prevImg.data )  
        {  
                
            Mat flow;
            calcOpticalFlowSF(prevImg, img, flow, 6, 2, 2);
            
            Mat xy[2];
            split(flow, xy);
            //calculate angle and magnitude
            Mat magnitude, angle;
            cartToPolar(xy[0], xy[1], magnitude, angle, true);
            //translate magnitude to range [0;1]
            double mag_max;
            minMaxLoc(magnitude, 0, &mag_max);
            magnitude.convertTo(magnitude, -1, 1.0/mag_max);
                //build hsv image
            Mat _hsv[3], hsv;
            _hsv[0] = angle;
            _hsv[1] = Mat::ones(angle.size(), CV_32F);
            _hsv[2] = magnitude;
            merge(_hsv, 3, hsv);
            //convert to BGR and show
            Mat bgr;//CV_32FC3 matrix
            cvtColor(hsv, bgr, COLOR_HSV2BGR);
            imshow("Simple Flow Algorithm", bgr);
            
        }  
          
        std::swap(prevImg, img);  
        
        if(waitKey(10)>=0)  
            break;  
    }  
    return 0;  
} 
    
    

// Show usage description

static void help()
{
    // print a welcome message, and the OpenCV version
    cout << "\nThis is a demo of optical flow algorithms\n"
            "Using OpenCV "<< endl;
    cout << "1 calcOpticalFlowPyrLK(Lucas-Kanade)\n";
    cout << "2 calcOpticalFlowFarneback(Gunnar Farneback, dense)\n";
    cout << "3 CalcOpticalFlowHS(Horn-Schunck)\n";
    cout << "4 cvCalcOpticalFlowBM(Block Matching)\n";
    cout << "5 calcOpticalFlowSF(Simple Flow Algorithm)\n";
    cout << "Press q to exit\n";
}

    
int main(int argc, char**argv)
      
{
    while (1)
    {
        help();
        char input;
        cin>>input;
        
        double t = (double)cvGetTickCount(); 
        
        if(input == 'q')
            return 0;
        if(input == '1')
            lucas();
        if(input == '2')
            fb();
        if(input == '3')
            hs();
        if(input == '4')
            bm();
        if(input == '5')
            sf();
        t = (double)cvGetTickCount() - t;  
        cout << endl << "Finished! cost time: " << t / ((double)cvGetTickFrequency()*1000.) << endl;
    }
   
}
      
      
