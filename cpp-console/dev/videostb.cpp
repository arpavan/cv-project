#include <iostream>
#include <cstdlib>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/video.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <fstream>

using namespace cv;
using namespace std;

string inFile;
VideoCapture cap;

class Tracker {
    vector<Point2f> trackedFeatures;
    Mat             prevGray;
public:
    bool            freshStart;
    Mat_<float>     rigidTransform;

    Tracker():freshStart(true) {
        rigidTransform = Mat::eye(3,3,CV_32FC1); //affine 2x3 in a 3x3 matrix
    }

    void processImage(Mat& img) {
        Mat gray; cvtColor(img,gray,CV_BGR2GRAY);
        vector<Point2f> corners;
        if(trackedFeatures.size() < 200) {
            goodFeaturesToTrack(gray,corners,300,0.01,10);
            cout << "found " << corners.size() << " features\n";
            for (int i = 0; i < corners.size(); ++i) {
                trackedFeatures.push_back(corners[i]);
            }
        }

        if(!prevGray.empty()) {
            vector<uchar> status; vector<float> errors;
            calcOpticalFlowPyrLK(prevGray,gray,trackedFeatures,corners,status,errors,Size(10,10));

            if(countNonZero(status) < status.size() * 0.8) {
                cout << "cataclysmic error \n";
                rigidTransform = Mat::eye(3,3,CV_32FC1);
                trackedFeatures.clear();
                prevGray.release();
                freshStart = true;
                return;
            } else
                freshStart = false;

            Mat_<float> newRigidTransform = estimateRigidTransform(trackedFeatures,corners,false);
            Mat_<float> nrt33 = Mat_<float>::eye(3,3);
            newRigidTransform.copyTo(nrt33.rowRange(0,2));
            rigidTransform *= nrt33;

            trackedFeatures.clear();
            for (int i = 0; i < status.size(); ++i) {
                if(status[i]) {
                    trackedFeatures.push_back(corners[i]);
                }
            }
        }

        gray.copyTo(prevGray);
    }
};

VideoCapture init_input(char* in)
{
    inFile = string(in);
    VideoCapture cap(inFile); //capture the video from file
    if (!cap.isOpened())  // if not success, exit program
    {
        cout << "Cannot open file" << endl;
        exit(0);
    }
    return cap;    
}


VideoWriter init_output(void)
{
    string outFile;
    stringstream outFilestream;
    outFilestream << inFile << "Processed" << ".avi";
    outFile = outFilestream.str();
    Size S = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH), (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));    // Acquire input size
    int fourcc = CV_FOURCC('H','2','6','4');
    VideoWriter outputVideo;
    outputVideo.open(outFile, fourcc, 30, S, true);  
    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << outFile << endl;
        exit(0);
    }
    return outputVideo;
}

// process frames as they come
Mat doProcessing(Mat src)
{
    // Mat orig_copy,orig_warped;
    // original.copyTo(orig_copy);
    // tracker.processImage(orig_copy);

    // Mat invTrans = tracker.rigidTransform.inv(DECOMP_SVD);
    // warpAffine(orig_copy,orig_warped,invTrans.rowRange(0,2),Size());
    int thresh = 240;

    Mat dst, dst_norm, dst_norm_scaled, src_gray;
    dst = Mat::zeros( src.size(), CV_32FC1 );

    cvtColor( src, src_gray, CV_BGR2GRAY );

    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    /// Detecting corners
    cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

    /// Normalizing
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );

    int count = 0;
    /// Drawing a circle around corners
    for( int j = 0; j < dst_norm.rows ; j++ )
    {
        for( int i = 0; i < dst_norm.cols; i++ )
        {
            if( (int) dst_norm.at<float>(j,i) > thresh )
            {
                circle( src, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
                count++;

            }
        }
    }
    cout << count << endl;
    return src;
}


int main(int argc, char **argv)
{

    /* Initialize VideoCapture */
    cap = init_input(argv[1]);

    /* Initialize VideoWriter */
    VideoWriter outputVideo = init_output();

    /* Processing variables */
    Mat imgOriginal,orig_warped,tmp;
    Tracker tracker;

    Mat currFrame, prevFrame, resFrame;

    while (1)
    {
        
        /* Video Capture */
        bool bSuccess = cap.read(currFrame); // read a new frame from video
        if(!bSuccess) {
            break;
        }
        
        /* Begin processing */
        resFrame = doProcessing(currFrame);    

        /* End Processing */   
        
        /* Record output */
        outputVideo << resFrame;
		
    	if(waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            cout << "esc key is pressed by user" << endl;
            break; 
        }
        imshow("Result", resFrame);
        
    }

    cout << "Processing ended" << endl;
    cap.release();
    
    return 0;
}