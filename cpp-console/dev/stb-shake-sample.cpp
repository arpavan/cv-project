#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace std;
using namespace cv;

// This video stablisation smooths the global trajectory using a sliding average window

const int SMOOTHING_RADIUS = 30; // In frames. The larger the more stable the video, but less reactive to sudden panning
const int HORIZONTAL_BORDER_CROP = 2; // In pixels. Crops the border to reduce the black borders from stabilisation being too noticeable.

// 1. Get previous to current frame transformation (dx, dy, da) for all frames
// 2. Accumulate the transformations to get the image trajectory
// 3. Smooth out the trajectory using an averaging window
// 4. Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
// 5. Apply the new transformation to the video

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

    // For further analysis
    // ofstream out_transform("prev_to_cur_transformation.txt");
    // ofstream out_trajectory("trajectory.txt");
    // ofstream out_smoothed_trajectory("smoothed_trajectory.txt");
    // ofstream out_new_transform("new_prev_to_cur_transformation.txt");

    VideoCapture cap(argv[1]);
    assert(cap.isOpened());

    cout << "FPS " << cap.get(CV_CAP_PROP_FPS) << endl;
    Mat cur, cur_grey;
    Mat prev, prev_grey;

    cap >> prev;
    cvtColor(prev, prev_grey, COLOR_BGR2GRAY);

    // Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
    vector <TransformParam> prev_to_cur_transform; // previous to current

    int k=1;
    int max_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    Mat last_T, lastGoodFrame;
    int avgOpticalFlow = 0;
    int sumOpticalFlow = 0;
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

        int currOpticalFlow = prev_corner2.size();
        sumOpticalFlow += currOpticalFlow;
        avgOpticalFlow = sumOpticalFlow/k;
        cout << "Frame: " << k << "/" << max_frames << " - good optical flow: " << currOpticalFlow << " - avg optical flow: " << avgOpticalFlow << endl;
        k++;

        // translation + rotation only
        Mat T = estimateRigidTransform(prev_corner2, cur_corner2, false); // false = rigid transform, no scaling/shearing

        // in rare cases no transform is found. We'll just use the last known good transform.
        // TODO: DONT COPY
        if(T.data == NULL) {
            cout << "no transform detected" << endl;
            //exit(0);
            last_T.copyTo(T);
            continue;
            // cap.set(CV_CAP_PROP_FPS, 45);
            // cout << "FPS2 " << cap.get(CV_CAP_PROP_FPS) << endl;
            // imshow("original", prev);
            //continue;
        }
            //write these frames to new video

            //cur.copyTo(lastGoodFrame);
            // cap.set(CV_CAP_PROP_FPS, 30);
            // cout << "FPS1 " << cap.get(CV_CAP_PROP_FPS) << endl;
    
            // decompose T
            double dx = T.at<double>(0,2);
            double dy = T.at<double>(1,2);
            double da = atan2(T.at<double>(1,0), T.at<double>(0,0));
            //outputVideo << cur;
            T.copyTo(last_T);
            //left shift by 1
            cur.copyTo(prev);
            cur_grey.copyTo(prev_grey);
            // imshow("current", cur);
            // imshow("prev", prev);
            cout << "dx transform " << dx << endl;
            cout << "dy transform " << dy << endl;
            // getting the previous-to-current transformation for all frames
            prev_to_cur_transform.push_back(TransformParam(dx, dy, da));
            imshow("current", cur);
            imshow("prev", prev);

    }

    // Step 2 - Accumulate the transformations to get the image trajectory

    // Accumulated frame to frame transform
    double a = 0;
    double x = 0;
    double y = 0;

    vector <Trajectory> trajectory; // trajectory at all frames

    for(size_t i=0; i < prev_to_cur_transform.size(); i++) {
        x += prev_to_cur_transform[i].dx;
        y += prev_to_cur_transform[i].dy;
        a += prev_to_cur_transform[i].da;

        trajectory.push_back(Trajectory(x,y,a));

        // out_trajectory << (i+1) << " " << x << " " << y << " " << a << endl;
    }

    // the actual sliding window part for eliminating jerks
    // Step 3 - Smooth out the trajectory using an averaging window
    // can use a sliding window here instead of averaging window just for the jerks
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

        // out_smoothed_trajectory << (i+1) << " " << avg_x << " " << avg_y << " " << avg_a << endl;
    }

    // Step 4 - Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
    vector <TransformParam> new_prev_to_cur_transform;

    // Accumulated frame to frame transform
    a = 0;
    x = 0;
    y = 0;

    for(size_t i=0; i < prev_to_cur_transform.size(); i++) {
        x += prev_to_cur_transform[i].dx;
        y += prev_to_cur_transform[i].dy;
        a += prev_to_cur_transform[i].da;

        // target - current
        double diff_x = smoothed_trajectory[i].x - x;
        double diff_y = smoothed_trajectory[i].y - y;
        double diff_a = smoothed_trajectory[i].a - a;

        double dx = prev_to_cur_transform[i].dx + diff_x;
        double dy = prev_to_cur_transform[i].dy + diff_y;
        double da = prev_to_cur_transform[i].da + diff_a;

        new_prev_to_cur_transform.push_back(TransformParam(dx, dy, da));

        //out_new_transform << (i+1) << " " << dx << " " << dy << " " << da << endl;
    }

    // Step 5 - Apply the new transformation to the video
    cap.set(CV_CAP_PROP_POS_FRAMES, 0);
    Mat T(2,3,CV_64F);

    // optional, no need of cropping for now
    int vert_border = HORIZONTAL_BORDER_CROP * prev.rows / prev.cols; // get the aspect ratio correct

    k=0;
    while(k < max_frames-1) { // we cant get a valid transform for the last frame so skip it
        cap >> cur;

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

        // optional - crop and resize to eliminate the null outer edges
        cur2 = cur2(Range(vert_border, cur2.rows-vert_border), Range(HORIZONTAL_BORDER_CROP, cur2.cols-HORIZONTAL_BORDER_CROP));
        // optional - Resize cur2 back to cur size, for better side by side comparison
        resize(cur2, cur2, cur.size());

        // optional - Now draw the original and stablised side by side for coolness
        Mat canvas = Mat::zeros(cur.rows, cur.cols*2+10, cur.type());
        cur.copyTo(canvas(Range::all(), Range(0, cur2.cols)));
        cur2.copyTo(canvas(Range::all(), Range(cur2.cols+10, cur2.cols*2+10)));

        // optional - If too big to fit on the screen, then scale it down by 2, hopefully it'll fit :)
        if(canvas.cols > 1920) {
            resize(canvas, canvas, Size(canvas.cols/2, canvas.rows/2));
        }

        // imshow("before and after", canvas);
        cout << "frame " << k << endl;
        //imshow("processed", cur2);

        //char str[256];
        //sprintf(str, "images/%08d.jpg", k);
        //imwrite(str, canvas);

        waitKey(20);

        k++;
    }

    return 0;
}