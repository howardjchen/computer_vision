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

#include <pthread.h>
#include <omp.h>

#include <string.h>

#define K_KNN 4  // k for KNN
#define RANSAC_DISTANCE 3
#define ITERATIVE 20

#define H_START_NUM 0
#define H_END_NUM 4

#define OBJECT_IMG "object_11.bmp"
#define TARGET_IMG "object_12.bmp"
 
long thread_count;
long long n = 256;

int key1,key2;
std::vector<cv::KeyPoint> keypoints1, keypoints2;
std::vector<cv::KeyPoint> keypoints3, keypoints4;

cv :: Mat H[256];
int H_InlierNumber[256]; 


using namespace std;
using namespace cv;

void* RANSAC_Thread(void* rank);

Mat ComputeH(int n, Point2f *p1, Point2f *p2)
{ 
    int i; 
    CvMat *A = cvCreateMat(2*n, 9, CV_64FC1); 
    CvMat *U = cvCreateMat(2*n, 2*n, CV_64FC1); 
    CvMat *D = cvCreateMat(2*n, 9, CV_64FC1); 
    CvMat *V = cvCreateMat(9, 9, CV_64FC1); 
    Mat H_local;
    cvZero(A); 
    
    for(i = 0; i < 1; i++)
    { 
        // 2*i row 
        cvmSet(A,2*i,3,-p1[i].x); 
        cvmSet(A,2*i,4,-p1[i].y); 
        cvmSet(A,2*i,5,-1); 
        cvmSet(A,2*i,6,p2[i].y*p1[i].x); 
        cvmSet(A,2*i,7,p2[i].y*p1[i].y); 
        cvmSet(A,2*i,8,p2[i].y); 
        // 2*i+1 row 
        cvmSet(A,2*i+1,0,p1[i].x); 
        cvmSet(A,2*i+1,1,p1[i].y); 
        cvmSet(A,2*i+1,2,1); 
        cvmSet(A,2*i+1,6,-p2[i].x*p1[i].x); 
        cvmSet(A,2*i+1,7,-p2[i].x*p1[i].y); 
        cvmSet(A,2*i+1,8,-p2[i].x); 
    } 
    // SVD 
    // The flags cause U and V to be returned transpose
    // Therefore, in OpenCV, A = U^T D V 
    cvSVD(A, D, U, V, CV_SVD_U_T|CV_SVD_V_T);  

    // take the last column of V^T, i.e., last row of V
    /*for(i=0; i<9; i++) 
        cvmSet(H_local, i/3, i%3, cvmGet(V, 8, i)); */
      
    cvReleaseMat(&A); 
    cvReleaseMat(&U); 
    cvReleaseMat(&D); 
    cvReleaseMat(&V); 

    return H_local;
} 
 
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

Mat FirstProcess( Mat ObjectImage, Mat TargetImage)
{
    int diff_vector = 0;

    //Mat ObjectImage = cv::imread( OBJECT_IMG, 1 );  //type : 8UC3
    //Mat TargetImage = cv::imread( TARGET_IMG, 1 );
 
    /* threshold      = 0.04;
       edge_threshold = 10.0;
       magnification  = 3.0;    */
 
    // SIFT feature detector and feature extractor
    cv::SiftFeatureDetector detector( 0.05, 5.0 );
    cv::SiftDescriptorExtractor extractor( 3.0 );
 
    // Feature detection
    detector.detect( ObjectImage, keypoints1 );
    detector.detect( TargetImage, keypoints2 );
 
    // Feature display
    Mat feat1,feat2;
    drawKeypoints(ObjectImage,keypoints1,feat1,Scalar(255, 255, 255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(TargetImage,keypoints2,feat2,Scalar(255, 255, 255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imwrite( "feat1.bmp", feat1 );
    imwrite( "feat2.bmp", feat2 );

    key1 = keypoints1.size();   //object
    key2 = keypoints2.size();   //target
    printf("Keypoint1 = %d \n",key1);
    printf("Keypoint2 = %d \n",key2);
 
    // Feature descriptor computation
    Mat descriptor1,descriptor2;
    extractor.compute( ObjectImage, keypoints1, descriptor1 );
    extractor.compute( TargetImage, keypoints2, descriptor2 );

    printf("Descriptor1=(%d,%d)\n", descriptor1.size().height,descriptor1.size().width);
    printf("Descriptor2=(%d,%d)\n", descriptor2.size().height,descriptor2.size().width);


    //int KeyPoint_Neighborhood[key1][key2];
    //int K_NearestNeighbor[key1][K_KNN];

    int **KeyPoint_Neighborhood;    //int KeyPoint_Neighborhood[key1][key2];
    int **K_NearestNeighbor;   //int K_NearestNeighbor[key1][K_KNN];

    KeyPoint_Neighborhood = (int **)malloc(key1*sizeof(int*));
    for (int i = 0; i < key1; ++i)
        KeyPoint_Neighborhood[i] = (int *)malloc(key2*sizeof(int));

    K_NearestNeighbor = (int **)malloc(key1*sizeof(int*));
    for (int i = 0; i < key1; ++i)
        K_NearestNeighbor[i] = (int *)malloc(K_KNN*sizeof(int));


    cout << "computing KNN" << endl;

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


    std::vector<Point2f> obj[256];
    std::vector<Point2f> scene[256];
    

    int Extract_Index[256][4];
    int m = 0;
    int IndexForKNN;
    int IndexForScene; 


    cout << "arrange the index of neighbors from KNN" << endl;
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


    memset(H_InlierNumber,0,sizeof(int));
    Mat H_local[256];
    int Extract = 0;



    //cout << "computing Homography" << endl;
    /*for (int i = 0; i < 256; ++i)
    {
        for (int j = H_START_NUM; j < H_END_NUM; ++j)
        {
            IndexForKNN = Extract_Index[i][Extract];
            IndexForScene = K_NearestNeighbor[j][IndexForKNN];
            scene[i].push_back( keypoints2[IndexForScene].pt );
            Extract++;
            if(Extract > 3)
                Extract = 0;
            //cout << IndexForScene << " ";
        }
        //printf("\n");
        H_local[i] = findHomography(obj, scene[i]);
        H[i] = H_local[i].clone();
    }*/


    /*for (size_t i = H_START_NUM; i < H_END_NUM; ++i)
        obj.push_back( keypoints1[i].pt );*/

    int count = 0;
    Mat Candidate_H[ITERATIVE];
    int Candidate_InlierNumber[ITERATIVE];
    int store[256][4];
    int Candidate_store[ITERATIVE][4];

    while(count < ITERATIVE)
    {
        cout << "computing Homography" << endl;
        for (int i = 0; i < 256; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                int CorespondIndex = (rand()% key1);
                store[i][j] = CorespondIndex;
                IndexForKNN = Extract_Index[i][Extract];
                IndexForScene = K_NearestNeighbor[CorespondIndex][IndexForKNN];
                scene[i].push_back( keypoints2[IndexForScene].pt );
                obj[i].push_back( keypoints1[CorespondIndex].pt );
                Extract++;
                if(Extract > 3)
                    Extract = 0;
                //printf("CorespondIndex = %d ",CorespondIndex );
            }
            //printf("\n");
            H[i] = findHomography(obj[i], scene[i]);
            obj[i].clear();
            scene[i].clear();
        }



        
        double Candidate[3];
        //double Candidate[3];
        Mat TargetKeypoint[key1];
        double Object_X, Object_Y;
        double Target_X, Target_Y;
        double distance;

        int Max_InlierNumber = 0;
        int Max_InlierIndex = 0;

        cout << "Computing the best Homography using RANSAC" << endl;

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

        printf("the best Candidate Homography[%d] = %d\n",count,Max_InlierNumber );
        Candidate_H[count] = H[Max_InlierIndex].clone();
        Candidate_InlierNumber[count] = Max_InlierNumber;

        for (int ii = 0; ii < 4; ++ii)
        {
            Candidate_store[count][ii] = store[Max_InlierIndex][ii];
        }
        
        count++;
    }

    int Max_Candidate_InlierNumber = 0;
    int Max_Candidate_InlierIndex = 0;

    for (int i = 0; i < ITERATIVE; ++i)
    {
        if (Candidate_InlierNumber[i] > Max_Candidate_InlierNumber)
        {
            Max_Candidate_InlierNumber = Candidate_InlierNumber[i];
            Max_Candidate_InlierIndex = i;
        }
    }

    printf("the best candidate H[%d] is : %d \n",Max_Candidate_InlierIndex,Max_Candidate_InlierNumber);
    printf("store_corespond = %d %d %d %d\n",Candidate_store[Max_Candidate_InlierIndex][0],Candidate_store[Max_Candidate_InlierIndex][1],Candidate_store[Max_Candidate_InlierIndex][2],Candidate_store[Max_Candidate_InlierIndex][3]);

    Mat Reconvered_H(Candidate_H[Max_Candidate_InlierIndex]);

    cout << Reconvered_H << endl;


    //Mat WarpingImage = Mat::zeros(ObjectImage.rows, ObjectImage.cols,CV_8UC3);
    Mat WarpingImage = TargetImage.clone();
    Mat WarpingPoint;
    double Candidate[3];

    //printf("object : row = %d, col = %d\n",ObjectImage.rows, ObjectImage.cols );
    //printf("warping : row = %d, col = %d\n",WarpingImage.rows, WarpingImage.cols );
    printf("Warping\n");

    for (int i = 0; i <= ObjectImage.rows-1 ; i++)      //rows
    {
        for (int j = 0; j <= ObjectImage.cols-1; j++)   //cols
        {
            if(ObjectImage.at<Vec3b>(i,j)[0] != 255 && ObjectImage.at<Vec3b>(i,j)[1] != 255 && ObjectImage.at<Vec3b>(i,j)[2] != 255)
            {
                Candidate[0] = i;
                Candidate[1] = j;
                Candidate[2] = 1;
                Mat Before(3,1,CV_64FC1,Candidate);
                WarpingPoint = Reconvered_H*Before;
                int x = WarpingPoint.at<double>(0,0);
                int y = WarpingPoint.at<double>(0,1);
                if(x > 0 && y > 0)
                    WarpingImage.at<Vec3b>(x,y) = ObjectImage.at<Vec3b>(i,j);
            }    
        }
    }

   /* Mat Result = TargetImage.clone();
    Point2f src_center(WarpingImage.cols/2.0F, WarpingImage.rows/2.0F);
    Mat rot_mat = getRotationMatrix2D(src_center, 0, 1.0);
    Mat AfterRotation;
    warpAffine(WarpingImage, AfterRotation, rot_mat, WarpingImage.size());

    for (int i = 0; i <= AfterRotation.rows-1 ; i++)     //rows
    {
        for (int j = 0; j <= AfterRotation.cols-1; j++) //cols
        {
            if(AfterRotation.at<Vec3b>(i,j)[0] != 0 && AfterRotation.at<Vec3b>(i,j)[1] != 0 && AfterRotation.at<Vec3b>(i,j)[2] != 0)
            {
                Result.at<Vec3b>(i,j) = AfterRotation.at<Vec3b>(i,j);
            }    
        }
    }*/

    //Mat result = TargetImage.clone();
    //WarpingImage.copyTo(result);
    //WarpingImage.copyTo(result(Rect(0, 0, WarpingImage.cols, WarpingImage.rows)));
    //imshow("result", result);


    //imshow("ObjectImage",ObjectImage);
    //imshow("WarpingImage",WarpingImage);
    //imshow("Result",Result);
    //imshow("TargetImage",TargetImage)

    free(K_NearestNeighbor);
    free (KeyPoint_Neighborhood);

    return WarpingImage;
}

Mat SecondProcess( Mat ObjectImage, Mat TargetImage)
{
    int diff_vector = 0;

    printf("object rows = %d, cols = %d\n",ObjectImage.rows, ObjectImage.cols );
    printf("Target rows = %d, cols = %d\n",TargetImage.rows, TargetImage.cols );

    //Mat ObjectImage = cv::imread( OBJECT_IMG, 1 );  //type : 8UC3
    //Mat TargetImage = cv::imread( TARGET_IMG, 1 );
 
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

    
    int **KeyPoint_Neighborhood;    //int KeyPoint_Neighborhood[key1][key2];
    int **K_NearestNeighbor;   //int K_NearestNeighbor[key1][K_KNN];

    KeyPoint_Neighborhood = (int **)malloc(key1*sizeof(int*));
    for (int i = 0; i < key1; ++i)
        KeyPoint_Neighborhood[i] = (int *)malloc(key2*sizeof(int));

    K_NearestNeighbor = (int **)malloc(key1*sizeof(int*));
    for (int i = 0; i < key1; ++i)
        K_NearestNeighbor[i] = (int *)malloc(K_KNN*sizeof(int));


    cout << "computing KNN" << endl;

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


    std::vector<Point2f> obj[256];
    std::vector<Point2f> scene[256];


    int Extract_Index[256][4];
    int m = 0;
    int IndexForKNN;
    int IndexForScene; 
    int H_InlierNumber[256];

    memset(H_InlierNumber,0,sizeof(int));



    cout << "arrange the index of neighbors from KNN" << endl;
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
    int Extract = 0;

    int count = 0;
    Mat Candidate_H[ITERATIVE];
    int Candidate_InlierNumber[ITERATIVE];
    int store[256][4];
    int Candidate_store[ITERATIVE][4];

    while(count < ITERATIVE)
    {
        cout << "computing Homography" << endl;
        for (int i = 0; i < 256; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                int CorespondIndex = (rand()% key1);
                store[i][j] = CorespondIndex;
                IndexForKNN = Extract_Index[i][Extract];
                IndexForScene = K_NearestNeighbor[CorespondIndex][IndexForKNN];
                scene[i].push_back( keypoints2[IndexForScene].pt );
                obj[i].push_back( keypoints1[CorespondIndex].pt );
                Extract++;
                if(Extract > 3)
                    Extract = 0;
                //printf("CorespondIndex = %d ",CorespondIndex );
            }
            //printf("\n");
            H[i] = findHomography(obj[i], scene[i]);
            obj[i].clear();
            scene[i].clear();
        }



        
        double Candidate[3];
        //double Candidate[3];
        Mat TargetKeypoint[key1];
        double Object_X, Object_Y;
        double Target_X, Target_Y;
        double distance;

        int Max_InlierNumber = 0;
        int Max_InlierIndex = 0;

        cout << "Computing the best Homography using RANSAC" << endl;
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

        printf("the best Candidate Homography[%d] = %d\n",count,Max_InlierNumber );
        Candidate_H[count] = H[Max_InlierIndex].clone();
        Candidate_InlierNumber[count] = Max_InlierNumber;

        for (int ii = 0; ii < 4; ++ii)
        {
            Candidate_store[count][ii] = store[Max_InlierIndex][ii];
        }
        
        count++;
    }

    int Max_Candidate_InlierNumber = 0;
    int Max_Candidate_InlierIndex = 0;

    for (int i = 0; i < ITERATIVE; ++i)
    {
        if (Candidate_InlierNumber[i] > Max_Candidate_InlierNumber)
        {
            Max_Candidate_InlierNumber = Candidate_InlierNumber[i];
            Max_Candidate_InlierIndex = i;
        }
    }

    printf("the best candidate H[%d] is : %d \n",Max_Candidate_InlierIndex,Max_Candidate_InlierNumber);
    printf("store_corespond = %d %d %d %d\n",Candidate_store[Max_Candidate_InlierIndex][0],Candidate_store[Max_Candidate_InlierIndex][1],Candidate_store[Max_Candidate_InlierIndex][2],Candidate_store[Max_Candidate_InlierIndex][3]);

    Mat Reconvered_H(Candidate_H[Max_Candidate_InlierIndex]);

    cout << Reconvered_H << endl;






    //Mat WarpingImage = Mat::zeros(ObjectImage.rows, ObjectImage.cols,CV_8UC3);
    Mat WarpingImage = TargetImage.clone();


    //printf("object : row = %d, col = %d\n",ObjectImage.rows, ObjectImage.cols );
    //printf("warping : row = %d, col = %d\n",WarpingImage.rows, WarpingImage.cols );
    printf("Warping\n");
    Mat WarpingPoint;
    double Candidate[3];

    for (int i = 0; i <= ObjectImage.rows-1 ; i++)      //rows
    {
        for (int j = 0; j <= ObjectImage.cols-1; j++)   //cols
        {
            if(ObjectImage.at<Vec3b>(i,j)[0] != 255 && ObjectImage.at<Vec3b>(i,j)[1] != 255 && ObjectImage.at<Vec3b>(i,j)[2] != 255)
            {
                Candidate[0] = i;
                Candidate[1] = j;
                Candidate[2] = 1;
                Mat Before(3,1,CV_64FC1,Candidate);
                WarpingPoint = Reconvered_H*Before;
                int x = WarpingPoint.at<double>(0,0);
                int y = WarpingPoint.at<double>(0,1);
                if ( x >= 0 && y >= 0)
                    WarpingImage.at<Vec3b>(x,y) = ObjectImage.at<Vec3b>(i,j);     
            }    
        }
    }

    /*Mat Result = TargetImage.clone();
    Point2f src_center(WarpingImage.cols/2.0F, WarpingImage.rows/2.0F);
    Mat rot_mat = getRotationMatrix2D(src_center, 0, 1.0);
    Mat AfterRotation;

    warpAffine(WarpingImage, AfterRotation, rot_mat, WarpingImage.size());

    for (int i = 0; i <= AfterRotation.rows-1 ; i++)     //rows
    {
        for (int j = 0; j <= AfterRotation.cols-1; j++) //cols
        {
            if(AfterRotation.at<Vec3b>(i,j)[0] != 0 && AfterRotation.at<Vec3b>(i,j)[1] != 0 && AfterRotation.at<Vec3b>(i,j)[2] != 0)
            {
                Result.at<Vec3b>(i,j) = AfterRotation.at<Vec3b>(i,j);
            }    
        }
    }*/

    //Mat result = TargetImage.clone();
    //WarpingImage.copyTo(result);
    //WarpingImage.copyTo(result(Rect(0, 0, WarpingImage.cols, WarpingImage.rows)));
    //imshow("result", result);


    //imshow("ObjectImage",ObjectImage);
    //imshow("WarpingImage",WarpingImage);
    //imshow("Result",Result);
    //imshow("TargetImage",TargetImage)

    free(KeyPoint_Neighborhood);
    free(K_NearestNeighbor);

    return WarpingImage;
}

Mat FirstProcess_Pthread( Mat ObjectImage, Mat TargetImage)
{
    int diff_vector = 0;

    long       thread;  /* Use long in case of a 64-bit system */
    pthread_t* thread_handles;

    thread_handles = (pthread_t*) malloc (thread_count*sizeof(pthread_t)); 
    //Mat ObjectImage = cv::imread( OBJECT_IMG, 1 );  //type : 8UC3
    //Mat TargetImage = cv::imread( TARGET_IMG, 1 );
 
    /* threshold      = 0.04;
       edge_threshold = 10.0;
       magnification  = 3.0;    */
 
    // SIFT feature detector and feature extractor
    cv::SiftFeatureDetector detector( 0.05, 5.0 );
    cv::SiftDescriptorExtractor extractor( 3.0 );
 
    // Feature detection
    //std::vector<cv::KeyPoint> keypoints1, keypoints2;
    detector.detect( ObjectImage, keypoints1 );
    detector.detect( TargetImage, keypoints2 );
 
    // Feature display
    Mat feat1,feat2;
    drawKeypoints(ObjectImage,keypoints1,feat1,Scalar(255, 255, 255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(TargetImage,keypoints2,feat2,Scalar(255, 255, 255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imwrite( "feat1.bmp", feat1 );
    imwrite( "feat2.bmp", feat2 );

    key1 = keypoints1.size();   //object
    key2 = keypoints2.size();   //target
    printf("Keypoint1 = %d \n",key1);
    printf("Keypoint2 = %d \n",key2);
 
    // Feature descriptor computation
    Mat descriptor1,descriptor2;
    extractor.compute( ObjectImage, keypoints1, descriptor1 );
    extractor.compute( TargetImage, keypoints2, descriptor2 );

    printf("Descriptor1=(%d,%d)\n", descriptor1.size().height,descriptor1.size().width);
    printf("Descriptor2=(%d,%d)\n", descriptor2.size().height,descriptor2.size().width);


    //int KeyPoint_Neighborhood[key1][key2];
    //int K_NearestNeighbor[key1][K_KNN];

    int **KeyPoint_Neighborhood;    //int KeyPoint_Neighborhood[key1][key2];
    int **K_NearestNeighbor;   //int K_NearestNeighbor[key1][K_KNN];

    KeyPoint_Neighborhood = (int **)malloc(key1*sizeof(int*));
    for (int i = 0; i < key1; ++i)
        KeyPoint_Neighborhood[i] = (int *)malloc(key2*sizeof(int));

    K_NearestNeighbor = (int **)malloc(key1*sizeof(int*));
    for (int i = 0; i < key1; ++i)
        K_NearestNeighbor[i] = (int *)malloc(K_KNN*sizeof(int));


    cout << "computing KNN" << endl;

    //omp_set_num_threads(4);
    //OMP_SET_NUM_THREADS(4)

    #pragma omp parallel shared(descriptor1,descriptor2,KeyPoint_Neighborhood) private(i,j,k,diff_vector)
    {
        #pragma for schedule(static)
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
    

    for (size_t i = H_START_NUM; i < H_END_NUM; ++i)
        obj.push_back( keypoints1[i].pt );

    int Extract_Index[256][4];
    int m = 0;
    int IndexForKNN;
    int IndexForScene; 
    //int H_InlierNumber[256];

    memset(H_InlierNumber,0,sizeof(int));



    cout << "arrange the index of neighbors from KNN" << endl;
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


    int Extract = 0;

    cout << "computing Homography" << endl;
    /*for (int i = 0; i < 256; ++i)
    {
        for (int j = H_START_NUM; j < H_END_NUM; ++j)
        {
            IndexForKNN = Extract_Index[i][Extract];
            IndexForScene = K_NearestNeighbor[j][IndexForKNN];
            scene[i].push_back( keypoints2[IndexForScene].pt );
            Extract++;
            if(Extract > 3)
                Extract = 0;
            //cout << IndexForScene << " ";
        }
        //printf("\n");
        H[i] = findHomography(obj, scene[i]);
    }*/


    for (int i = 0; i < 256; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            int CorespondIndex = (rand()% key1);
            IndexForKNN = Extract_Index[i][Extract];
            IndexForScene = K_NearestNeighbor[CorespondIndex][IndexForKNN];
            scene[i].push_back( keypoints2[IndexForScene].pt );
            Extract++;
            if(Extract > 3)
                Extract = 0;
            printf("CorespondIndex = %d ",CorespondIndex );
        }
        printf("\n");
        H[i] = findHomography(obj, scene[i]);
    }


    cout << "Computing the best Homography using RANSAC" << endl;

    for (thread = 0; thread < thread_count; thread++)  
        pthread_create(&thread_handles[thread], NULL,RANSAC_Thread, (void*)thread);

    for (thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread],NULL);

    
    int Max_InlierIndex = 0;
    int Max_InlierNumber = 0;

    for (int i = 0; i < 256; ++i)
    {
        if (Max_InlierNumber < H_InlierNumber[i])
        {
            Max_InlierNumber = H_InlierNumber[i];
            Max_InlierIndex = i;
        }
    }

    //cout << TargetKeypoint[0] << endl; 
    //cout << TargetKeypoint[0].at<double>(1,0) << endl;
    //cout << Object[0] << endl;

    printf("the best Homography[%d] = %d\n",Max_InlierIndex,Max_InlierNumber );
    Mat Reconvered_H(H[Max_InlierIndex]);


    //Mat WarpingImage = Mat::zeros(ObjectImage.rows, ObjectImage.cols,CV_8UC3);
    Mat WarpingImage = TargetImage.clone();
    Mat WarpingPoint;


    printf("Warping\n");
    double Candidate[3];

    for (int i = 0; i <= ObjectImage.rows-1 ; i++)      //rows
    {
        for (int j = 0; j <= ObjectImage.cols-1; j++)   //cols
        {
            if(ObjectImage.at<Vec3b>(i,j)[0] != 255 && ObjectImage.at<Vec3b>(i,j)[1] != 255 && ObjectImage.at<Vec3b>(i,j)[2] != 255)
            {
                Candidate[0] = i;
                Candidate[1] = j;
                Candidate[2] = 1;
                Mat Before(3,1,CV_64FC1,Candidate);
                WarpingPoint = Reconvered_H*Before;
                int x = WarpingPoint.at<double>(0,0);
                int y = WarpingPoint.at<double>(0,1);
                if(x >=0 && y>= 0)
                    WarpingImage.at<Vec3b>(x,y) = ObjectImage.at<Vec3b>(i,j);
            }    
        }
    }

   /* Mat Result = TargetImage.clone();
    Point2f src_center(WarpingImage.cols/2.0F, WarpingImage.rows/2.0F);
    Mat rot_mat = getRotationMatrix2D(src_center, 0, 1.0);
    Mat AfterRotation;
    warpAffine(WarpingImage, AfterRotation, rot_mat, WarpingImage.size());

    for (int i = 0; i <= AfterRotation.rows-1 ; i++)     //rows
    {
        for (int j = 0; j <= AfterRotation.cols-1; j++) //cols
        {
            if(AfterRotation.at<Vec3b>(i,j)[0] != 0 && AfterRotation.at<Vec3b>(i,j)[1] != 0 && AfterRotation.at<Vec3b>(i,j)[2] != 0)
            {
                Result.at<Vec3b>(i,j) = AfterRotation.at<Vec3b>(i,j);
            }    
        }
    }*/

    //Mat result = TargetImage.clone();
    //WarpingImage.copyTo(result);
    //WarpingImage.copyTo(result(Rect(0, 0, WarpingImage.cols, WarpingImage.rows)));
    //imshow("result", result);


    //imshow("ObjectImage",ObjectImage);
    //imshow("WarpingImage",WarpingImage);
    //imshow("Result",Result);
    //imshow("TargetImage",TargetImage)

    free(K_NearestNeighbor);
    free (KeyPoint_Neighborhood);
    free(thread_handles);

    return WarpingImage;
}

Mat SecondProcess_Pthread( Mat ObjectImage, Mat TargetImage)
{
    int diff_vector = 0;

    long       thread;  /* Use long in case of a 64-bit system */
    pthread_t* thread_handles;

    thread_handles = (pthread_t*) malloc (thread_count*sizeof(pthread_t)); 
    //Mat ObjectImage = cv::imread( OBJECT_IMG, 1 );  //type : 8UC3
    //Mat TargetImage = cv::imread( TARGET_IMG, 1 );
 
    /* threshold      = 0.04;
       edge_threshold = 10.0;
       magnification  = 3.0;    */
 
    // SIFT feature detector and feature extractor
    cv::SiftFeatureDetector detector( 0.05, 5.0 );
    cv::SiftDescriptorExtractor extractor( 3.0 );
 
    // Feature detection
    //std::vector<cv::KeyPoint> keypoints1, keypoints2;
    detector.detect( ObjectImage, keypoints1 );
    detector.detect( TargetImage, keypoints2 );
 
    // Feature display
    Mat feat1,feat2;
    drawKeypoints(ObjectImage,keypoints1,feat1,Scalar(255, 255, 255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(TargetImage,keypoints2,feat2,Scalar(255, 255, 255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imwrite( "feat1.bmp", feat1 );
    imwrite( "feat2.bmp", feat2 );

    key1 = keypoints1.size();   //object
    key2 = keypoints2.size();   //target
    printf("Keypoint1 = %d \n",key1);
    printf("Keypoint2 = %d \n",key2);
 
    // Feature descriptor computation
    Mat descriptor1,descriptor2;
    extractor.compute( ObjectImage, keypoints1, descriptor1 );
    extractor.compute( TargetImage, keypoints2, descriptor2 );

    printf("Descriptor1=(%d,%d)\n", descriptor1.size().height,descriptor1.size().width);
    printf("Descriptor2=(%d,%d)\n", descriptor2.size().height,descriptor2.size().width);


    //int KeyPoint_Neighborhood[key1][key2];
    //int K_NearestNeighbor[key1][K_KNN];

    int **KeyPoint_Neighborhood;    //int KeyPoint_Neighborhood[key1][key2];
    int **K_NearestNeighbor;   //int K_NearestNeighbor[key1][K_KNN];

    KeyPoint_Neighborhood = (int **)malloc(key1*sizeof(int*));
    for (int i = 0; i < key1; ++i)
        KeyPoint_Neighborhood[i] = (int *)malloc(key2*sizeof(int));

    K_NearestNeighbor = (int **)malloc(key1*sizeof(int*));
    for (int i = 0; i < key1; ++i)
        K_NearestNeighbor[i] = (int *)malloc(K_KNN*sizeof(int));


    cout << "computing KNN" << endl;

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
    

    for (size_t i = H_START_NUM; i < H_END_NUM; ++i)
        obj.push_back( keypoints1[i].pt );

    int Extract_Index[256][4];
    int m = 0;
    int IndexForKNN;
    int IndexForScene; 
    //int H_InlierNumber[256];

    memset(H_InlierNumber,0,sizeof(int));



    cout << "arrange the index of neighbors from KNN" << endl;
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


    int Extract = 0;

    cout << "computing Homography" << endl;
    for (int i = 0; i < 256; ++i)
    {
        for (int j = H_START_NUM; j < H_END_NUM; ++j)
        {
            IndexForKNN = Extract_Index[i][Extract];
            IndexForScene = K_NearestNeighbor[j][IndexForKNN];
            scene[i].push_back( keypoints2[IndexForScene].pt );
            Extract++;
            if(Extract > 3)
                Extract = 0;
            //cout << IndexForScene << " ";
        }
        //printf("\n");
        H[i] = findHomography(obj, scene[i]);
    }

    cout << "Computing the best Homography using RANSAC" << endl;

    for (thread = 0; thread < thread_count; thread++)  
        pthread_create(&thread_handles[thread], NULL,RANSAC_Thread, (void*)thread);

    for (thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread],NULL);

    
    int Max_InlierIndex = 0;
    int Max_InlierNumber = 0;

    for (int i = 0; i < 256; ++i)
    {
        if (Max_InlierNumber < H_InlierNumber[i])
        {
            Max_InlierNumber = H_InlierNumber[i];
            Max_InlierIndex = i;
        }
    }

    //cout << TargetKeypoint[0] << endl; 
    //cout << TargetKeypoint[0].at<double>(1,0) << endl;
    //cout << Object[0] << endl;

    printf("the best Homography[%d] = %d\n",Max_InlierIndex,Max_InlierNumber );
    Mat Reconvered_H(H[Max_InlierIndex]);


    //Mat WarpingImage = Mat::zeros(ObjectImage.rows, ObjectImage.cols,CV_8UC3);
    Mat WarpingImage = TargetImage.clone();
    Mat WarpingPoint;

    //printf("object : row = %d, col = %d\n",ObjectImage.rows, ObjectImage.cols );
    //printf("warping : row = %d, col = %d\n",WarpingImage.rows, WarpingImage.cols );
    printf("Warping\n");

    double Candidate[3];

    for (int i = 0; i <= ObjectImage.rows-1 ; i++)      //rows
    {
        for (int j = 0; j <= ObjectImage.cols-1; j++)   //cols
        {
            if(ObjectImage.at<Vec3b>(i,j)[0] != 255 && ObjectImage.at<Vec3b>(i,j)[1] != 255 && ObjectImage.at<Vec3b>(i,j)[2] != 255)
            {
                Candidate[0] = i;
                Candidate[1] = j;
                Candidate[2] = 1;
                Mat Before(3,1,CV_64FC1,Candidate);
                WarpingPoint = Reconvered_H*Before;
                int x = WarpingPoint.at<double>(0,0);
                int y = WarpingPoint.at<double>(0,1);
                if(x >= 0 && y>=0)
                    WarpingImage.at<Vec3b>(x,y) = ObjectImage.at<Vec3b>(i,j);
            }    
        }
    }

   /* Mat Result = TargetImage.clone();
    Point2f src_center(WarpingImage.cols/2.0F, WarpingImage.rows/2.0F);
    Mat rot_mat = getRotationMatrix2D(src_center, 0, 1.0);
    Mat AfterRotation;
    warpAffine(WarpingImage, AfterRotation, rot_mat, WarpingImage.size());

    for (int i = 0; i <= AfterRotation.rows-1 ; i++)     //rows
    {
        for (int j = 0; j <= AfterRotation.cols-1; j++) //cols
        {
            if(AfterRotation.at<Vec3b>(i,j)[0] != 0 && AfterRotation.at<Vec3b>(i,j)[1] != 0 && AfterRotation.at<Vec3b>(i,j)[2] != 0)
            {
                Result.at<Vec3b>(i,j) = AfterRotation.at<Vec3b>(i,j);
            }    
        }
    }*/

    //Mat result = TargetImage.clone();
    //WarpingImage.copyTo(result);
    //WarpingImage.copyTo(result(Rect(0, 0, WarpingImage.cols, WarpingImage.rows)));
    //imshow("result", result);


    //imshow("ObjectImage",ObjectImage);
    //imshow("WarpingImage",WarpingImage);
    //imshow("Result",Result);
    //imshow("TargetImage",TargetImage)

    free(K_NearestNeighbor);
    free (KeyPoint_Neighborhood);
    free(thread_handles);

    return WarpingImage;
}

Mat SecondProcess_Pthread_v2( Mat ObjectImage, Mat TargetImage, Mat ReturnImage)
{
    int diff_vector = 0;

    long       thread;  /* Use long in case of a 64-bit system */
    pthread_t* thread_handles;

    thread_handles = (pthread_t*) malloc (thread_count*sizeof(pthread_t)); 
    //Mat ObjectImage = cv::imread( OBJECT_IMG, 1 );  //type : 8UC3
    //Mat TargetImage = cv::imread( TARGET_IMG, 1 );
 
    /* threshold      = 0.04;
       edge_threshold = 10.0;
       magnification  = 3.0;    */
 
    // SIFT feature detector and feature extractor

    cv::SiftFeatureDetector detector( 0.05, 5.0 );
    cv::SiftDescriptorExtractor extractor( 3.0 );
 
    // Feature detection
    //std::vector<cv::KeyPoint> keypoints1, keypoints2;

    cout << "before keypoint" << endl;
    keypoints1 = keypoints3;
    keypoints2 = keypoints3;
    detector.detect( ObjectImage, keypoints1 );
    detector.detect( TargetImage, keypoints2 );
 
    // Feature display
    Mat feat1,feat2;
    /*drawKeypoints(ObjectImage,keypoints1,feat1,Scalar(255, 255, 255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(TargetImage,keypoints2,feat2,Scalar(255, 255, 255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imwrite( "feat1.bmp", feat1 );
    imwrite( "feat2.bmp", feat2 );*/

    key1 = keypoints1.size();   //object
    key2 = keypoints2.size();   //target
    printf("Keypoint1 = %d \n",key1);
    printf("Keypoint2 = %d \n",key2);
 
    // Feature descriptor computation
    Mat descriptor1,descriptor2;
    extractor.compute( ObjectImage, keypoints1, descriptor1 );
    extractor.compute( TargetImage, keypoints2, descriptor2 );

    printf("Descriptor1=(%d,%d)\n", descriptor1.size().height,descriptor1.size().width);
    printf("Descriptor2=(%d,%d)\n", descriptor2.size().height,descriptor2.size().width);


    //int KeyPoint_Neighborhood[key1][key2];
    //int K_NearestNeighbor[key1][K_KNN];

    int **KeyPoint_Neighborhood;    //int KeyPoint_Neighborhood[key1][key2];
    int **K_NearestNeighbor;   //int K_NearestNeighbor[key1][K_KNN];

    KeyPoint_Neighborhood = (int **)malloc(key1*sizeof(int*));
    for (int i = 0; i < key1; ++i)
        KeyPoint_Neighborhood[i] = (int *)malloc(key2*sizeof(int));

    K_NearestNeighbor = (int **)malloc(key1*sizeof(int*));
    for (int i = 0; i < key1; ++i)
        K_NearestNeighbor[i] = (int *)malloc(K_KNN*sizeof(int));


    cout << "computing KNN" << endl;

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
    

    for (size_t i = H_START_NUM; i < H_END_NUM; ++i)
        obj.push_back( keypoints1[i].pt );

    int Extract_Index[256][4];
    int m = 0;
    int IndexForKNN;
    int IndexForScene; 
    //int H_InlierNumber[256];

    memset(H_InlierNumber,0,sizeof(int));



    cout << "arrange the index of neighbors from KNN" << endl;
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


    int Extract = 0;

    cout << "computing Homography" << endl;
    for (int i = 0; i < 256; ++i)
    {
        for (int j = H_START_NUM; j < H_END_NUM; ++j)
        {
            IndexForKNN = Extract_Index[i][Extract];
            IndexForScene = K_NearestNeighbor[j][IndexForKNN];
            scene[i].push_back( keypoints2[IndexForScene].pt );
            Extract++;
            if(Extract > 3)
                Extract = 0;
            //cout << IndexForScene << " ";
        }
        //printf("\n");
        H[i] = findHomography(obj, scene[i]);
    }

    cout << "Computing the best Homography using RANSAC" << endl;

    for (thread = 0; thread < thread_count; thread++)  
        pthread_create(&thread_handles[thread], NULL,RANSAC_Thread, (void*)thread);

    for (thread = 0; thread < thread_count; thread++)
        pthread_join(thread_handles[thread],NULL);

    
    int Max_InlierIndex = 0;
    int Max_InlierNumber = 0;

    for (int i = 0; i < 256; ++i)
    {
        if (Max_InlierNumber < H_InlierNumber[i])
        {
            Max_InlierNumber = H_InlierNumber[i];
            Max_InlierIndex = i;
        }
    }

    //cout << TargetKeypoint[0] << endl; 
    //cout << TargetKeypoint[0].at<double>(1,0) << endl;
    //cout << Object[0] << endl;

    printf("the best Homography[%d] = %d\n",Max_InlierIndex,Max_InlierNumber );
    Mat Reconvered_H(H[Max_InlierIndex]);


    Mat WarpingImage = Mat::zeros(ObjectImage.rows, ObjectImage.cols,CV_8UC3);
    //Mat WarpingImage = ReturnImage.clone();
    Mat WarpingPoint;

    //printf("object : row = %d, col = %d\n",ObjectImage.rows, ObjectImage.cols );
    //printf("warping : row = %d, col = %d\n",WarpingImage.rows, WarpingImage.cols );
    printf("Warping\n");

    double Candidate[3];

    for (int i = 0; i <= ObjectImage.rows-1 ; i++)      //rows
    {
        for (int j = 0; j <= ObjectImage.cols-1; j++)   //cols
        {
            if(ObjectImage.at<Vec3b>(i,j)[0] != 255 && ObjectImage.at<Vec3b>(i,j)[1] != 255 && ObjectImage.at<Vec3b>(i,j)[2] != 255)
            {
                Candidate[0] = i;
                Candidate[1] = j;
                Candidate[2] = 1;
                Mat Before(3,1,CV_64FC1,Candidate);
                WarpingPoint = Reconvered_H*Before;
                int x = WarpingPoint.at<double>(0,0);
                int y = WarpingPoint.at<double>(0,1);
                if(x >= 0)
                    WarpingImage.at<Vec3b>(x-100,y+200) = ObjectImage.at<Vec3b>(i,j);
            }    
        }
    }

    Mat Result = ReturnImage.clone();
    Point2f src_center(WarpingImage.cols/2.0F, WarpingImage.rows/2.0F);
    Mat rot_mat = getRotationMatrix2D(src_center, 180, 1.0);
    Mat AfterRotation;
    warpAffine(WarpingImage, AfterRotation, rot_mat, WarpingImage.size());

    for (int i = 0; i <= AfterRotation.rows-1 ; i++)     //rows
    {
        for (int j = 0; j <= AfterRotation.cols-1; j++) //cols
        {
            if(AfterRotation.at<Vec3b>(i,j)[0] != 0 && AfterRotation.at<Vec3b>(i,j)[1] != 0 && AfterRotation.at<Vec3b>(i,j)[2] != 0)
            {
                Result.at<Vec3b>(i,j) = AfterRotation.at<Vec3b>(i,j);
            }    
        }
    }

    //Mat result = TargetImage.clone();
    //WarpingImage.copyTo(result);
    //WarpingImage.copyTo(result(Rect(0, 0, WarpingImage.cols, WarpingImage.rows)));
    //imshow("result", result);


    //imshow("ObjectImage",ObjectImage);
    //imshow("WarpingImage",WarpingImage);
    //imshow("Result",Result);
    //imshow("TargetImage",TargetImage)

    free(K_NearestNeighbor);
    free (KeyPoint_Neighborhood);
    free(thread_handles);

    return Result;
}


void* RANSAC_Thread(void* rank)
{
    long my_rank = (long) rank;
    long long i;
    long long part = n / thread_count;
    long long my_first_i = part * my_rank;
    long long my_last_i = my_first_i + part;

    double Candidate[3];
    Mat TargetKeypoint[key1];
    double Object_X, Object_Y;
    double Target_X, Target_Y;
    double distance;

    int Max_InlierNumber = 0;
    int Max_InlierIndex = 0;

    //printf("thread %ld is working : %lld - %lld\n",my_rank,my_first_i,my_last_i );

    for (int i = my_first_i; i < my_last_i; ++i)
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

                if (distance < RANSAC_DISTANCE)
                {
                    H_InlierNumber[i]++;
                    break;
                }
            }
            /*********************************************************/
        }
    }
}


int main(int argc, char const *argv[])
{
    Mat ObjectImage = cv::imread( argv[1], 1 );  //type : 8UC3
    //Mat ObjectImage2 = cv::imread( argv[2], 1 );
    Mat TargetImage = cv::imread( argv[2], 1 );
    Mat ResultImage;
    Mat TempImage;
    int cmd = strtol(argv[3],NULL,10);
    double start, end;
    clock_t clock();
    thread_count = strtol(argv[4],NULL,10);

    start = clock();
    if (cmd == 1)
    {
        printf("================ using method 1 ===============\n");
        ResultImage = FirstProcess(ObjectImage,TargetImage);
    }
    else if(cmd == 2)
    {
        printf("================= using method 2================\n");
        ResultImage = SecondProcess(ObjectImage,TargetImage);
    }
    else if(cmd == 3)
    {
        printf("================= using method 1 pthread : %ld ================\n",thread_count);
        ResultImage = FirstProcess_Pthread(ObjectImage, TargetImage);
    }
    else if(cmd == 4)
    {
        printf("================= using method 2 pthread : %ld ================\n",thread_count);
        ResultImage = SecondProcess_Pthread(ObjectImage, TargetImage);
    }
    /*else if(cmd == 5)
    {
        TempImage = FirstProcess_Pthread(ObjectImage, TargetImage);
        ResultImage = SecondProcess_Pthread_v2(ObjectImage2, TargetImage,TempImage);
    }*/

    imshow("result",ResultImage);
    imwrite( "result.bmp", ResultImage );

    end = clock();
    printf("the time elasped = %f s\n",(end-start)/CLOCKS_PER_SEC);
    waitKey(0);

    return 0;
}

