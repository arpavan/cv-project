package me.cv.unstutter;

import android.icu.lang.UCharacter;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;

import org.bytedeco.javacpp.avcodec;
import org.bytedeco.javacpp.indexer.FloatArrayIndexer;
import org.bytedeco.javacpp.opencv_video;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.FrameRecorder;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_imgproc.*;

import java.util.ArrayList;

import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.goodFeaturesToTrack;
import static org.bytedeco.javacpp.opencv_optflow.calcOpticalFlowSF;
import static org.bytedeco.javacpp.opencv_video.estimateRigidTransform;


class TransformParam {
        TransformParam() {}
        TransformParam(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
        }

        double dx;
        double dy;
        double da; // angle
}


class Trajectory{
        Trajectory() {}
        Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
        }

        double x;
        double y;
        double a; // angle
}

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        final String TAG = "unStutter";


        // Initialize video capture
        String baseDir = Environment.getExternalStorageDirectory().getAbsolutePath();
        Log.d(TAG,"directory "+baseDir);
        String input = "Video1";
        String ext = ".avi";
        String inputPath = baseDir+"/"+input+ext;


        FrameGrabber aviGrabber = new FFmpegFrameGrabber(inputPath);
        try {
            aviGrabber.start();
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }
        OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();

        double FPS = aviGrabber.getFrameRate();
        int h = aviGrabber.getImageHeight();
        int w = aviGrabber.getImageWidth();
        int c = aviGrabber.getVideoCodec();
        String format = aviGrabber.getFormat();

        Log.d(TAG,"videocapture init complete");

        // Initialize video writer
        String outputPath = baseDir + "/" + input + "processed" + ext;
        FFmpegFrameRecorder recorder = new FFmpegFrameRecorder(outputPath,w,h,1);
        recorder.setVideoCodec(c);
        recorder.setFormat(format);
        recorder.setFrameRate(FPS);
        try {
            recorder.start();
        } catch (FrameRecorder.Exception e) {
            e.printStackTrace();
        }

        Log.d(TAG,"videowriter init complete");

        Log.d(TAG,"input file "+inputPath);
        Log.d(TAG,"output file "+outputPath);

        Mat mat = new Mat();

        Mat cur = new Mat();
        Mat cur_grey = new Mat();
        Mat prev = new Mat();
        Mat prev_grey = new Mat();
        Mat last_T = new Mat();
        Mat lastGoodFrame = new Mat();


        // start processing frames
        Frame frame = null;
        try {
            frame = aviGrabber.grab();
            prev = converterToMat.convert(frame);
        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }

        cvtColor(prev, prev_grey, COLOR_BGR2GRAY);

        ArrayList<TransformParam> prev_to_cur_transform = new ArrayList<TransformParam>();

        int k=1;
        int max_frames = aviGrabber.getLengthInFrames();
        int avgOpticalFlow = 0;
        int sumOpticalFlow = 0;


        for(;;) {

            // 1. read frames
            // Grab an image Frame from the video file
            try {
                frame = aviGrabber.grab();
                if(frame!=null) {
                    cur = converterToMat.convert(frame);
                } else {
                    aviGrabber.stop();
                    recorder.stop();
                    recorder.release();
                    aviGrabber.release();
                    break;
                }
            } catch (FrameGrabber.Exception e) {
                e.printStackTrace();
                try {
                    aviGrabber.release();
                    recorder.release();
                    Log.d(TAG,"processing stopped");
                    break;
                } catch (Exception e1) {
                    e1.printStackTrace();
                }
            } catch (FrameRecorder.Exception e) {
                e.printStackTrace();
            }


            Log.d(TAG,"processing...");
            // 2. process frames
            cvtColor(cur, cur_grey, COLOR_BGR2GRAY);

            // vector from prev to cur
            Mat prev_corner = new Mat();
            Mat cur_corner = new Mat();
            Mat prev_corner2 = new Mat();
            Mat cur_corner2 = new Mat();
            Mat status = new Mat();
            Mat err = new Mat();

            goodFeaturesToTrack(prev_grey, prev_corner, 200, 0.01, 30);


            opencv_video.calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner, status, err);

            // weed out bad matches // status = status vector stores 1/0 depending on the detection of flow for the given set of features
            for(int i=0; i < status.rows(); i++) {
                if(status.row(i).col(0)) {
                    prev_corner2.push_back(prev_corner[i]);
                    cur_corner2.push_back(cur_corner[i]);
                }
            }

            int currOpticalFlow = prev_corner2.size();
            sumOpticalFlow += currOpticalFlow;
            avgOpticalFlow = sumOpticalFlow/k;
            //cout << "Frame: " << k << "/" << max_frames << " - good optical flow: " << currOpticalFlow << " - avg optical flow: " << avgOpticalFlow << endl;
            k++;

            // translation + rotation only
            Mat T = estimateRigidTransform(prev_corner2, cur_corner2, false); // false = rigid transform, no scaling/shearing

            // in rare cases no transform is found. We'll just use the last known good transform.
            // TODO: DONT COPY
            if(T.empty()) {
                //cout << "no transform detected" << endl;
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
            double dx = T.row(0).col(2);
            double dy = T.row(1).col(2);
            double da = Math.atan2(T.row(1).col(0), T.row(0).col(0));
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


            // 3. Record frames
            Frame outFrame;
            try {
                outFrame = converterToMat.convert(mat);
                recorder.record(outFrame);
            } catch (Exception e) {
                e.printStackTrace();
                try {
                    aviGrabber.release();
                    recorder.release();
                    Log.d(TAG,"processing stopped");
                    break;
                } catch (Exception e1) {
                    e1.printStackTrace();
                }
            }
        }

        
        Log.d(TAG,"processing complete");
    }
}
