#include <opencv/highgui.h>


#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgcodecs.hpp> // imread
#include <opencv2/highgui.hpp> // imshow, waitKey
#include <opencv2/opencv_modules.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <cstdlib>
 
using namespace std;
using namespace cv;
 
//
int main( void )
{
    //source image
    char* img1_file = "DSC_0363.jpg";
    char* img2_file = "DSC_0364.jpg";
 
    // image read
    Mat tmp = cv::imread( img1_file, 1 );
    Mat in  = cv::imread( img2_file, 1 );
 
    /* threshold      = 0.04;
       edge_threshold = 10.0;
       magnification  = 3.0;    */
 
    // SIFT feature detector and feature extractor
    cv::SiftFeatureDetector::detector( 0.05, 5.0 );
    cv::SiftDescriptorExtractor::extractor( 3.0 );
 
    // Feature detection
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
    detector.detect( tmp, keypoints1 );
    detector.detect( in, keypoints2 );
 
    // Feature display
    Mat feat1,feat2;
    drawKeypoints(tmp,keypoints1,feat1,Scalar(255, 255, 255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(in,keypoints2,feat2,Scalar(255, 255, 255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imwrite( "feat1.bmp", feat1 );
    imwrite( "feat2.bmp", feat2 );
    int key1 = keypoints1.size();
    int key2 = keypoints2.size();
    printf("Keypoint1=%d \nKeypoint2=%d \n", key1, key2);
 
    // Feature descriptor computation
    Mat descriptor1,descriptor2;
    extractor.compute( tmp, keypoints1, descriptor1 );
    extractor.compute( in, keypoints2, descriptor2 );

    printf("Descriptor1=(%d,%d) \nDescriptor2=(%d,%d)", descriptor1.size().height,descriptor1.size().width, descriptor2.size().height,descriptor2.size().width);

    system("pause");
 
    return 0;
}
