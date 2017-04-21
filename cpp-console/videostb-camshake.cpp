#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace std;
using namespace cv;

const int SMOOTHING_RADIUS = 50; // size of the fixed bounds to determine the degree of stabilization effect. Higher the better.

struct Transformation
{
    Transformation() {}
    Transformation(double _dx, double _dy, double _da) {
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
    outFilestream << argv[1] << "Temp" << ".avi";
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

    vector <Transformation> prev_to_cur_transform; // previous to current

    int k=1;
    int max_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    Mat last_T, lastGoodFrame;

    // pass-1: Process the entire video to elimiate jerks.
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

        // eliminate any zero matches // status = status vector stores 1/0 depending on the detection of flow for the given set of features
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
        prev_to_cur_transform.push_back(Transformation(dx, dy, da));

        k++;
    }
    outputVideo.release();
    cap.release();
    //Pass-1 ends.

    // Pre-pass-2: Computing and manipulating the transformations between frames to prepare for stabilization
    // from Pass-1 compute the transformation between all frames and compute the trajectory of the features through the frames.
    // get transformations between frames
    double a = 0;
    double x = 0;
    double y = 0;

    vector <Trajectory> trajectory; // trajectory at all frames

    for(size_t i=0; i < prev_to_cur_transform.size(); i++) {
        x += prev_to_cur_transform[i].dx;
        y += prev_to_cur_transform[i].dy;
        a += prev_to_cur_transform[i].da;

        trajectory.push_back(Trajectory(x,y,a));

    }    

    // smooth out the trajectory computed above using predetermined bounds.
    vector <Trajectory> smoothed_trajectory; // trajectory at all frames

    for(size_t i=0; i < trajectory.size(); i++) {
        double sum_x = 0;
        double sum_y = 0;
        double sum_a = 0;
        int count = 0;

        for(int j=-SMOOTHING_RADIUS; j <= SMOOTHING_RADIUS; j++) {
            if(i+j >= 0 && i+j < trajectory.size()) {
                sum_x += trajectory[i+j].x;
                sum_y += trajectory[i+j].y;
                sum_a += trajectory[i+j].a;

                count++;
            }
        }

        double avg_a = sum_a / count;
        double avg_x = sum_x / count;
        double avg_y = sum_y / count;

        smoothed_trajectory.push_back(Trajectory(avg_x, avg_y, avg_a));

    }

    // Compute the "new" frame-to-frame transformations to apply to the video frameset. These newly computed transformations
    // stabilize the video for vibrations
    vector <Transformation> new_prev_to_cur_transform;
    a = 0;
    x = 0;
    y = 0;

    for(size_t i=0; i < prev_to_cur_transform.size(); i++) {
        x += prev_to_cur_transform[i].dx;
        y += prev_to_cur_transform[i].dy;
        a += prev_to_cur_transform[i].da;

        double diff_x = smoothed_trajectory[i].x - x;
        double diff_y = smoothed_trajectory[i].y - y;
        double diff_a = smoothed_trajectory[i].a - a;

        double dx = prev_to_cur_transform[i].dx + diff_x;
        double dy = prev_to_cur_transform[i].dy + diff_y;
        double da = prev_to_cur_transform[i].da + diff_a;

        new_prev_to_cur_transform.push_back(Transformation(dx, dy, da));

    }

    VideoCapture cap2(outFile);
    if(!cap2.isOpened())  // if not success, exit program
    {
        cout << "cap2 Cannot open file" << endl;
        exit(0);
    }

    // Final pass - Iterate through all the frames and apply the above computed transformations
    // Apply affine warping with the computed transformations.
    cap2.set(CV_CAP_PROP_POS_FRAMES, 0);
    Mat T(2,3,CV_64F);

	string outFile2;
    stringstream outFilestream2;
    outFilestream2 << argv[1] << "Processed" << ".avi";
    outFile2 = outFilestream2.str();
	outputVideo.open(outFile2, fourcc, 30, S, true);


    k=0;
    while(k < max_frames-1) { // we cant get a valid transform for the last frame so skip it
        cap2 >> cur;

        if(cur.data == NULL) {
            break;
        }

        // extracting the updated transformation matrix (rotation, translation)
        T.at<double>(0,0) = cos(new_prev_to_cur_transform[k].da);
        T.at<double>(0,1) = -sin(new_prev_to_cur_transform[k].da);
        T.at<double>(1,0) = sin(new_prev_to_cur_transform[k].da);
        T.at<double>(1,1) = cos(new_prev_to_cur_transform[k].da);

        T.at<double>(0,2) = new_prev_to_cur_transform[k].dx;
        T.at<double>(1,2) = new_prev_to_cur_transform[k].dy;


        Mat cur2;

        // warp the current frame based on the computed transformation
        warpAffine(cur, cur2, T, cur.size());

        waitKey(20);

        k++;

        outputVideo << cur2;
    }

    cap2.release();
    outputVideo.release();

    return 0;
}