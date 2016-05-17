#include <opencv2/opencv.hpp>
 
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

//#include <opencv2/imgcodecs.hpp> // imread
#include <opencv2/core/core.hpp> // imread
#include <opencv2/highgui/highgui.hpp> // imshow, waitKey

#include <math.h>
#include <cmath>
#include <limits.h>

#include <sstream>
#include <fstream>

#define K_KNN 4  // k for KNN
#define RANSAC_DISTANCE 5

#define OBJECT_IMG "object_11.bmp"
#define TARGET_IMG "object_12.bmp"
 
using namespace std;
using namespace cv;
 
/*****************************************************************************

1. This function check the type of Mat
2. input  : Mat.type()
3. output : char of ty.c_str()
4. usage  : 
            string ty =  type2str( M.type() );
            printf("Matrix H: %s %dx%d \n", ty.c_str(), M.cols, M.rows );

******************************************************************************/
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

float EucliDistance(Point& p, Point& q) 
{
    Point diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

double ComputeDistance(double x1, double y1, double x2, double y2)
{
    double distance;
    double x_diff = x1-x2;
    double y_diff = y1-y2;
    
    distance = sqrt(x_diff*x_diff + y_diff*y_diff);
    return distance;
}

int main(  )
{
    int diff_vector = 0;
    int KeyPoint_Neighborhood[991][1250];
    int K_NearestNeighbor[991][K_KNN];

    Mat ObjectImage = cv::imread( OBJECT_IMG, 1 );  //type : 8UC3
    Mat TargetImage = cv::imread( TARGET_IMG, 1 );
 
    /* threshold      = 0.04;
       edge_threshold = 10.0;
       magnification  = 3.0;    */
 
    // SIFT feature detector and feature extractor
    cv::SiftFeatureDetector detector( 0.05, 5.0 );
    cv::SiftDescriptorExtractor extractor( 3.0 );
 
    // Feature detection
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
    detector.detect( ObjectImage, keypoints1 );
    detector.detect( TargetImage, keypoints2 );
 
    // Feature display
    Mat feat1,feat2;
    drawKeypoints(ObjectImage,keypoints1,feat1,Scalar(255, 255, 255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(TargetImage,keypoints2,feat2,Scalar(255, 255, 255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imwrite( "feat1.bmp", feat1 );
    imwrite( "feat2.bmp", feat2 );

    int key1 = keypoints1.size();   //object
    int key2 = keypoints2.size();   //target
    printf("Keypoint1 = %d \n",key1);
    printf("Keypoint2 = %d \n",key2);
 
    // Feature descriptor computation
    Mat descriptor1,descriptor2;
    extractor.compute( ObjectImage, keypoints1, descriptor1 );
    extractor.compute( TargetImage, keypoints2, descriptor2 );

    printf("Descriptor1=(%d,%d)\n", descriptor1.size().height,descriptor1.size().width);
    printf("Descriptor2=(%d,%d)\n", descriptor2.size().height,descriptor2.size().width);


    for (size_t i = 0; i < key1; ++i)   //need threading
    {
        for (size_t j = 0; j < key2; ++j)
        {
            for (size_t k = 0; k < 128; ++k)
            {
                diff_vector +=  (descriptor1.at<float>(i,k) - descriptor2.at<float>(j,k))*(descriptor1.at<float>(i,k) - descriptor2.at<float>(j,k));    
            }
            KeyPoint_Neighborhood[i][j] = sqrt(diff_vector);
            diff_vector = 0;
        }
    }

    int Min_Distance;
    int Min_Index = 0;

    for (size_t i = 0; i < key1; ++i)   //need threading7
    {   
        for (size_t j = 0; j < K_KNN; ++j)
        {
            Min_Distance = KeyPoint_Neighborhood[i][0]; 
            for (size_t k = 0; k < key2; ++k)
            {
                if(Min_Distance > KeyPoint_Neighborhood[i][k])
                {
                    Min_Distance = KeyPoint_Neighborhood[i][k];
                    Min_Index = k;
                } 
            }
            K_NearestNeighbor[i][j] = Min_Index;
            KeyPoint_Neighborhood[i][Min_Index] = INT_MAX;
        }
    }
    
    /*for (size_t i = 0; i < 4; ++i)
    {
        cout <<  keypoints1[i].pt.x << " " << keypoints2[i].pt.y << endl;
    }*/

    /*for (int i = 0; i < key1; ++i)
    {
        printf("K_NearestNeighbor[%d] = ",i);
        for (size_t j = 0; j < K_KNN; ++j)
            printf(" %d ",K_NearestNeighbor[i][j]);
        printf("\n");
    }*/


    std::vector<Point2f> obj;
    std::vector<Point2f> scene[256];
    

    for (size_t i = 0; i < 4; ++i)
        obj.push_back( keypoints1[i].pt );

    int Extract_Index[256][4];
    int m = 0;
    int IndexForKNN;
    int IndexForScene; 
    int H_InlierNumber[256];

    memset(H_InlierNumber,0,sizeof(int));


    for (int i= 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                for( int l = 0; l < 4 ; l++ )
                {
                    Extract_Index[m][0] = i;
                    Extract_Index[m][1] = j;
                    Extract_Index[m][2] = k;
                    Extract_Index[m][3] = l;
                    m++;
                }


    Mat H[256];

    for (int i = 0; i < 256; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            IndexForKNN = Extract_Index[i][j];
            IndexForScene = K_NearestNeighbor[j][IndexForKNN];
            scene[i].push_back( keypoints2[IndexForScene].pt );
            //cout << IndexForScene << " ";
        }
        //printf("\n");
        H[i] = findHomography(obj, scene[i]);
    }



    
    double Candidate[3];
    Mat TargetKeypoint[key1];
    //std::vector<Point2d> Target[key1];
    //std::vector<Point2d> Object[key2];
    double Object_X, Object_Y;
    double Target_X, Target_Y;
    double distance;

    int Max_InlierNumber = 0;
    int Max_InlierIndex = 0;

    for (int i = 0; i < 256; ++i)
    {
        for (int j = 0; j < key1; j++)
        {
            Candidate[0] = keypoints1[j].pt.x;
            Candidate[1] = keypoints1[j].pt.y;
            Candidate[2] = 1;

            Mat Before(3,1,CV_64FC1,Candidate);
            TargetKeypoint[j] = H[i]*Before;
            
            /****************** Computing inlier using RANSAC ***************/           
            
            Target_X = TargetKeypoint[j].at<double>(0,0);
            Target_Y = TargetKeypoint[j].at<double>(0,1);
            
            for (int k = 0; k < key2; k++)
            {
                Object_X = keypoints2[k].pt.x;
                Object_Y = keypoints2[k].pt.y;
                distance = ComputeDistance(Target_X,Target_Y,Object_X,Object_Y);
                //printf("distance = %f \n",distance );

                if (distance < RANSAC_DISTANCE)
                {
                    H_InlierNumber[i]++;
                    break;
                }
            }
            /*********************************************************/
        }

        if (H_InlierNumber[i] > Max_InlierNumber)
        {
            Max_InlierNumber = H_InlierNumber[i];
            Max_InlierIndex = i;
        }
        //printf("H_InlierNumber[%d] = %d\n",i,H_InlierNumber[i]);
    }
    //cout << TargetKeypoint[0] << endl; 
    //cout << TargetKeypoint[0].at<double>(1,0) << endl;
    //cout << Object[0] << endl;
    //printf("the max H[%d] = %d\n",Max_InlierIndex,Max_InlierNumber );

    Mat Reconvered_H(H[Max_InlierIndex]);
    int ObjectImage_row, ObjectImage_col;

    ObjectImage_row = ObjectImage.rows;
    ObjectImage_col = ObjectImage.cols;

    Mat WarpingImage = Mat::zeros(708, 1024,CV_8UC3);
    //WarpingImage.create(ObjectImage.rows, ObjectImage.cols,ObjectImage.type());
    //Mat WarpingImage = TargetImage.clone();
    //Mat WarpingImage;
    //Mat WarpingImage(ObjectImage.rows, ObjectImage.cols,ObjectImage.type());


    printf("object : row = %d, col = %d\n",ObjectImage.rows, ObjectImage.cols );
    printf("warping : row = %d, col = %d\n",WarpingImage.rows, WarpingImage.cols );

    for (int i = 0; i <= ObjectImage.rows-1 ; i++)     //rows
    {
        for (int j = 0; j <= ObjectImage.cols-1; j++) //cols
        {
            WarpingImage.at<Vec3b>(i,j) = ObjectImage.at<Vec3b>(i,j);
        }
    }

    imshow("objectimg",ObjectImage);
    imshow("Warping",WarpingImage);
    waitKey(0);

    return 0;
}