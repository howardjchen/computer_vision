#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <vector>

using namespace std;
using namespace cv;
int main()
{
	Mat img_1 = imread("object.jpg");
	Mat img_2 = imread("target.jpg");

	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

	////-- Step 1: Detect the keypoints:
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	f2d->detect(img_1, keypoints_1);
	f2d->detect(img_2, keypoints_2);

	//-- Step 2: Calculate descriptors (feature vectors)    
	Mat descriptors_1, descriptors_2;
	f2d->compute(img_1, keypoints_1, descriptors_1);
	f2d->compute(img_2, keypoints_2, descriptors_2);

	Mat feat1, feat2;
	drawKeypoints(img_1, keypoints_1, feat1);
	drawKeypoints(img_2, keypoints_2, feat2);
	imshow("Object_KeyPoint.jpg", feat1);
	imshow("Target_KeyPoint.jpg", feat2);
	imwrite("Object_KeyPoint.jpg", feat1);
	imwrite("Target_KeyPoint.jpg", feat2);
	int key1 = keypoints_1.size();
	int key2 = keypoints_2.size();
	printf("Keypoint1=%d \nKeypoint2=%d \n", key1, key2);
	printf("Descriptor1=(%d,%d) \nDescriptor2=(%d,%d)", descriptors_1.size().height, descriptors_1.size().width, descriptors_2.size().height, descriptors_2.size().width);


	waitKey(0);
	return 0;

}