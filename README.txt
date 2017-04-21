# cv-project
Code repository for CSc 8980 Computer Vision course project  

## Console application  

### Prerequisites
Running the console application requires C++ and OpenCV 2.4+ installed and configured on the computer. The following URL provides instrutions to setup OpenCV 2.4 on a Ububtu Linux environment: http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html  

### Running the program
1. Navigate to the folder cv-project/cpp-console  
2. Compile the "videostb-camshake.cpp" file using the below command:  
		g++ videostb-camshake.cpp -L/usr/local/opt/opencv3/lib/ -o videostb-camshake `pkg-config --cflags --libs opencv`
3. Upon successful compilation, execute the program by providing the input video file as a command line argument.
		E.g. ./videostb-camshake video1.mp4
4. The processed video will be placed in the same location as the input video with a "Processed" suffix. E.g. video1Processed.avi


For the android port the app package will be made available.