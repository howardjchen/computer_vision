#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <opencv2/opencv.hpp>
 
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/legacy/legacy.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
//#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/core/core.hpp> // imread
#include <time.h>

 using namespace std;
 using namespace cv;

 /** Function Headers */
 void detectAndDisplay( Mat frame );

 /** Global variables */

 /*
    haarcascade_cat.xml 
    traincascade/cascade.xml
    haarcascade_frontalface_alt.xml
 */
String face_cascade_name;
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";  //  haarcascade_frontalface_alt.xml
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);


int main( int argc, const char** argv )
{
    CvCapture* capture;
    Mat frame;
    double start, end;
    clock_t clock();
    int cmd = atoi(argv[2]);

    if(cmd == 1)
        face_cascade_name = "traincascade/cascade.xml";             
    else if(cmd == 2)
        face_cascade_name = "haarcascade_final2.xml";               
    else if(cmd == 3)
        face_cascade_name = "haarcascade_frontalface_alt.xml"; 
    else if(cmd == 4)
        face_cascade_name = "traincascade_final2/cascade.xml";                  


    start = clock();

    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };


    //-- 3. REad form image
    frame = imread( argv[1] ,1);
    if( !frame.empty() )
    { 
        imwrite("/Users/chungyunhsiao/Documents/MATLAB/final/AllData/TestData/TestImage.jpg",frame);
        imwrite("TestImage.jpg",frame);
        detectAndDisplay( frame ); 
        end = clock();
    }
    else
    { 
        printf(" --(!) No image -- Break!");
    }

    printf("execution time = %f\n", (end - start)/CLOCKS_PER_SEC );

   // waitKey(0);

    return 0;
}


void detectAndDisplay( Mat frame )
{
    FILE *fpout;
    FILE *ftemp;
    std::vector<Rect> faces;
    Mat frame_gray;

    //fpout = fopen("/Users/chungyunhsiao/Documents/MATLAB/final/AllData/TestData/data.txt","w");
    fpout = fopen("data.txt","w");

    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );


    for( size_t i = 0; i < faces.size(); i++ )
    {
        if(i == 0)
        {    
            Point point1(faces[i].x-10, faces[i].y-30);
            Point point2(faces[i].x + faces[i].width+20, faces[i].y + faces[i].height+10);
            rectangle(
                    frame,
                    point1,
                    point2,
                    Scalar( 255, 0, 255 ),
                    2,
                    8,
                    0 );
            printf("faces[%ld] = ( %d , %d ) %d %d  \n",i,faces[i].x-10, faces[i].y-30, faces[i].width+20, faces[i].height+10 );
            fprintf(fpout, "%d %d %d %d \n",faces[i].x-10, faces[i].y-30, faces[i].width+20, faces[i].height+10 ); 
        }
        else
        {
            Point point1(faces[i].x, faces[i].y);
            Point point2(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
            rectangle(
                        frame,
                        point1,
                        point2,
                        Scalar( 255, 0, 255 ),
                        2,
                        8,
                        0 );
            printf("faces[%ld] = ( %d , %d ) %d %d  \n",i,faces[i].x, faces[i].y, faces[i].width, faces[i].height );
            fprintf(fpout, "%d %d %d %d \n",faces[i].x, faces[i].y, faces[i].width, faces[i].height ); 
        }
    }

    cout << "Number of faces being detected = " << faces.size() << endl;
    //ftemp = fopen("/Users/chungyunhsiao/Documents/MATLAB/final/AllData/TestData/temp.txt","w");
   // imshow( window_name, frame );
   // imwrite("output.jpg", frame);
 }
