#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace std;
using namespace cv;

struct TransformParam
{
    TransformParam() {}
    TransformParam(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
    }

    double dx;
    double dy;
    double da; // angle
};

struct Trajectory
{
    Trajectory() {}
    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }

    double x;
    double y;
    double a; // angle
};

int main(int argc, char **argv)
{
    if(argc < 2) {
        cout << "./VideoStab [video.avi]" << endl;
        return 0;
    }

    VideoCapture cap(argv[1]);
    if(!cap.isOpened())  // if not success, exit program
    {
        cout << "Cannot open file" << endl;
        exit(0);
    }

    cout << "FPS " << cap.get(CV_CAP_PROP_FPS) << endl;

    string outFile;
    stringstream outFilestream;
    outFilestream << argv[1] << "Processed" << ".avi";
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



    Mat cur, cur_grey;
    Mat prev, prev_grey;

    cap >> prev;
    cvtColor(prev, prev_grey, COLOR_BGR2GRAY);

    // Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
    vector <TransformParam> prev_to_cur_transform; // previous to current

    int k=1;
    int max_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    Mat last_T, lastGoodFrame;

    while(true) {
        cap >> cur;

        if(cur.data == NULL) {
            break;
        }
        waitKey(20);
        
        cvtColor(cur, cur_grey, COLOR_BGR2GRAY);

        // vector from prev to cur
        vector <Point2f> prev_corner, cur_corner;
        vector <Point2f> prev_corner2, cur_corner2;
        vector <uchar> status;
        vector <float> err;

        // corner detection using the Shi-Tomasi method
        goodFeaturesToTrack(prev_grey, prev_corner, 200, 0.01, 30);
        
        // Calculate optical flow using Lucas-Kanade
        calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner, status, err);

        // weed out bad matches // status = status vector stores 1/0 depending on the detection of flow for the given set of features
        for(size_t i=0; i < status.size(); i++) {
            if(status[i]) {
                prev_corner2.push_back(prev_corner[i]);
                cur_corner2.push_back(cur_corner[i]);
            }
        }

        // translation + rotation only
        Mat T = estimateRigidTransform(prev_corner2, cur_corner2, false); // false = rigid transform, no scaling/shearing

        double dx = -1;
        double dy = -1;
        double da = -1;

        // TODO: DONT COPY
        if(T.data != NULL) {
            //write these frames to new video

            //cur.copyTo(lastGoodFrame);
            // cap.set(CV_CAP_PROP_FPS, 30);
            // cout << "FPS1 " << cap.get(CV_CAP_PROP_FPS) << endl;
    
            // decompose T
            dx = T.at<double>(0,2);
            dy = T.at<double>(1,2);
            da = atan2(T.at<double>(1,0), T.at<double>(0,0));
            imshow("original", cur);
            outputVideo << cur;

            //left shift by 1
	        cur.copyTo(prev);
    	    cur_grey.copyTo(prev_grey);
        } else {
            cout << "no transform detected" << endl;
            last_T.copyTo(T);
            // cap.set(CV_CAP_PROP_FPS, 45);
            // cout << "FPS2 " << cap.get(CV_CAP_PROP_FPS) << endl;
            imshow("original", prev);
            continue;
        }
        T.copyTo(last_T);

        cout << "dx transform " << dx << endl;
        cout << "dy transform " << dy << endl;
        
        // getting the previous-to-current transformation for all frames
        prev_to_cur_transform.push_back(TransformParam(dx, dy, da));

        cout << "Frame: " << k << "/" << max_frames << " - good optical flow: " << prev_corner2.size() << endl;
        k++;
    }

    cap.release();
}