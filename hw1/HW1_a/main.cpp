#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp> // imread
#include <opencv2/highgui.hpp> // imshow, waitKey
#include <stdio.h>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>



#include <vector>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

using namespace cv;
using namespace std;

#define INTEGRAL_STEP 2

void outputply( Mat &f)
{
	//FILE *fp1;
	fstream fp1;
	fp1.open("apple.ply",ios::out);

	fp1 << "ply" << endl;
	fp1 << "format ascii 1.0" << endl;
	fp1 << "comment alpha=1.0" << endl;
	fp1 << "element vertex 14400" << endl;
	fp1 << "property float x" << endl;
	fp1 << "property float y" << endl;
	fp1 << "property float z" << endl;
	fp1 << "property uchar red" << endl;
	fp1 << "property uchar green" << endl;
	fp1 << "property uchar blue" << endl;
	fp1 << "end_header" << endl;

	//fp1 = fopen("depth_map.txt","w");

	for (int j = 0; j < 120; ++j)
	{
		for (int i = 0; i < 120; ++i)
		{
			//fprintf(fp1, " %d %d %f 255 255 255 \n",i,j,f.at<float>(i,j));
			fp1<< j <<" " << i  << " "<< f.at<float>(j,i) <<" " <<"255 255 255" << endl;
		}
	}
}


/*void OutputAsMatlabCode(Mat &f,string filename)
{
	std::fstream file;

	file.open(filename,iso::out);
	file<<"figure;surf([";
	for(int y=0;y<120;y++)
	{
		for(int x=0;x<120;x++)
		{
			if(f.at<float>(y,x)<200&&f.at<float>(y,x)>-200)
				file<<f.at<float>(y,x)<<" ";
			else 
				file<<0<<" ";
			if(x==119)
				file<<";";
		}
		file<<"\n";
	}
	file<<"]);";
	file.close();
}
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/

template<typename Method, typename F, typename Float>

double integrate(F f, Float a, Float b, int steps, Method m)
{
  double s = 0;
  double h = (b-a)/steps;
  for (int i = 0; i < steps; ++i)
    s += m(f, a + h*i, h);
  return h*s;
}
 
// methods
class rectangular
{
public:
  enum position_type { left, middle, right };
  rectangular(position_type pos): position(pos) {}
  template<typename F, typename Float>
  double operator()(F f, Float x, Float h) const
  {
    switch(position)
    {
    case left:
      return f(x);
    case middle:
      return f(x+h/2);
    case right:
      return f(x+h);
    }
  }
private:
  const position_type position;
};
 
class trapezium
{
public:
  template<typename F, typename Float>
   double operator()(F f, Float x, Float h) const
  {
    return (f(x) + f(x+h))/2;
  }
};
 
class simpson
{
public:
  template<typename F, typename Float>
   double operator()(F f, Float x, Float h) const
  {
    return (f(x) + 4*f(x+h/2) + f(x+h))/6;
  }
};


/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
/**************************************************************************/
void flip(Mat& m,char type)
{
	Mat afterflip(120,120,CV_32FC1,Scalar(0));
	if(type=='y')
	{
		for(int y=0;y<120;y++)
		{
			for(int x=0;x<120;x++)
			{
				afterflip.at<float>(119-y,x)=m.at<float>(y,x);
			}
		}
	}
	else if(type=='x')
	{
		for(int y=0;y<120;y++)
		{
			for(int x=0;x<120;x++)
			{
				afterflip.at<float>(y,119-x)=m.at<float>(y,x);
			}
		}
	}
	m=afterflip;
}



//得到該點的強度值
Mat *GetIntensity(Mat *pic,int y,int x)
{
	float Array[6];
	//cout<<"\n("<<y<<","<<x<<")\t";
	for(int i=0;i<6;i++)
	{
		Array[i]=pic[i].at<uchar>(y,x);
		//cout<<Array[i]<<" ";
	}
	//回傳6*1 matrix 因為有六個光源
	Mat Intensity_local(6,1,CV_32FC1,&Array);
	Mat *Intensity_p=new Mat;

	Intensity_local.copyTo(*Intensity_p);
	return Intensity_p;
}

Mat partial_derive(120,120,CV_32FC3);

/////////////////////
///先從y方向積分/////
/////////////////////
double u=0;
double partialf_partialy_ydirection(double y) 
{ 
	return partial_derive.at<Vec3f>(y,u)[1]; 
}
double partialf_partialx_ydirection(double x) 
{ 
	return partial_derive.at<Vec3f>(0,x)[0]; 
}
double f_u_v_ydirection(double u_,double v)
{
	u=u_;
	return  integrate(partialf_partialx_ydirection, 0.0, u_, u_*INTEGRAL_STEP, trapezium())+integrate(partialf_partialy_ydirection, 0.0, v, v*INTEGRAL_STEP, trapezium());
}
/////////////////////
///先從x方向積分/////
/////////////////////
double partialf_partialy_xdirection(double y) 
{ 
	return partial_derive.at<Vec3f>(y,0)[1]; 
}
double v=0;
double partialf_partialx_xdirection(double x) 
{ 
	return partial_derive.at<Vec3f>(v,x)[0]; 
}
double f_u_v_xdirection(double u,double v_)
{
	v=v_;
	return  integrate(partialf_partialy_xdirection, 0.0, v_, v_*INTEGRAL_STEP, trapezium())+integrate(partialf_partialx_xdirection, 0.0, u, u*INTEGRAL_STEP, trapezium());
	
}


void RemoveOffset(Mat &m)
{
	float Array[18]={	-0.1250	,0		,0.1250,
						-0.1250	,0.1250	,0,
						1.5000	,-0.25	,-0.2500};
	Mat A_inv(3,3,CV_32FC1,&Array);
	Mat b(3,1,CV_32FC1),x(3,1,CV_32FC1);
	b.at<float>(0)=m.at<float>(2,2);
	b.at<float>(2)=m.at<float>(2,10);
	b.at<float>(1)=m.at<float>(10,2);
	//cout<<b<<endl;
	x=A_inv*b;
	//cout<<x<<endl;
	Mat plane(120,120,CV_32FC1);
	for(int y=0;y<120;y++)
	{
		for(int xx=0;xx<120;xx++)
		{
			plane.at<float>(y,xx)=x.at<float>(0)*xx+x.at<float>(1)*y+x.at<float>(2);
		}
	}
	//OutputAsMatlabCode(plane,"plane.txt");
	m=m-plane;
}


//高度值*-1
Mat &UpsideDown(Mat &m)
{
	for(int y=0;y<m.rows;y++)
	{
		for(int x=0;x<m.cols;x++)
		{
			m.at<float>(y,x)=m.at<float>(y,x)*-1;
		}
	}
	return m;
}


int main()
{
    
	Mat img[6];

	img[0] = imread("pic1.bmp", IMREAD_GRAYSCALE);
	img[1] = imread("pic2.bmp", IMREAD_GRAYSCALE);
	img[2] = imread("pic3.bmp", IMREAD_GRAYSCALE);
	img[3] = imread("pic4.bmp", IMREAD_GRAYSCALE);
	img[4] = imread("pic5.bmp", IMREAD_GRAYSCALE);
	img[5] = imread("pic6.bmp", IMREAD_GRAYSCALE);



	float Array[18]={  	30,60,9,
						30,50,9,
						30,40,9,
						30,30,9,
						30,20,9,
						30,10,9};
						
	Mat U(6,3,CV_32FC1,Array);
	Mat U_normalized(6,3,CV_32FC1);

	//cout << U << endl;

	float base[6];

	for (int x = 0; x < 6; ++x)
	{
		//printf("%f %f %f\n",U.at<float>(x,0),U.at<float>(x,1),U.at<float>(x,2) );
		base[x] = U.at<float>(x,0)*U.at<float>(x,0)	+U.at<float>(x,1)*U.at<float>(x,1) + U.at<float>(x,2)*U.at<float>(x,2);	
		base[x] = sqrt(base[x]);
		U_normalized.at<float>(x,0) = U.at<float>(x,0) / base[x];
		U_normalized.at<float>(x,1) = U.at<float>(x,1) / base[x];
		U_normalized.at<float>(x,2) = U.at<float>(x,2) / base[x];
	}
	
	//cout << U_normalized << endl;

	Mat U_trans(3,6,CV_32FC1);
	transpose(U,U_trans);
	
	Mat UTU_1UT=Mat(U_trans*U).inv()*U_trans;
	//cout<<UTU_1UT<<endl;


    //3 channel normal (未被normalize)存法向量的x y z
	Mat b(120,120,CV_32FC3);
	Mat normal(120,120,CV_32FC3);
	Mat rho(120,120,CV_32FC1);

	
	//共120*120個點
	//cout<<UTU_1UT<<endl;
	//cout<<(*GetIntensity(img,60,60))<<endl;
	//cout<<UTU_1UT*(*GetIntensity(img,60,60))<<endl;

	//cout<<"surface normal:"<<endl;

	for(int y = 0; y < 120; y++)
	{
		for(int x = 0; x < 120; x++)
		{
			Mat tmp_for_get_element(UTU_1UT*(*GetIntensity(img,y,x)));
			for(int z = 0; z < 3; z++)
			{
				//將得到的向量存到b 
				b.at<Vec3f>(y,x)[z] = tmp_for_get_element.at<float>(z);
			}
			//將b拆成 N ,rho
			//normal是向量 用vec3f存
			normal.at<Vec3f>(y,x)=cv::normalize(b.at<Vec3f>(y,x));
			//rho是b的長度,用一個float存就好
			rho.at<float>(y,x)=cv::norm(b.at<Vec3f>(y,x));
			//normalize後得到	(∂f/∂x,  ∂f/∂y,1)
			//[2]就是Nc,
			if(normal.at<Vec3f>(y,x)[2] != 0.0)
			{
				partial_derive.at<Vec3f>(y,x) = normal.at<Vec3f>(y,x) / normal.at<Vec3f>(y,x)[2];
			}
			else
			{
				partial_derive.at<Vec3f>(y,x) = 0.0;
				//fp1 << "y,x =  "<< x << " " << y  << "  ";
				//fp1 << normal.at<Vec3f>(y,x)  << "  ";
				//fp1 << img[0].at<long>(y,x) << endl;
			}
		}
	}

	imshow("normal",normal);

	//cout << normal.at<Vec3f>(60,60) << endl;

	Mat depth_data(120,120,CV_32FC1);

	for (int x = 0; x < 120; ++x)
	{
		for (int y = 0; y < 120; ++y)
		{
			if(normal.at<Vec3f>(y,x)[2] == 0.0)
			{
				depth_data.at<float>(y,x) = 0.0;
			}
			else
			{
				depth_data.at<float>(y,x) = -(normal.at<Vec3f>(y,x)[0])*x / (normal.at<Vec3f>(y,x)[2]) - (normal.at<Vec3f>(y,x)[1])*y / (normal.at<Vec3f>(y,x)[2]);
				if(depth_data.at<float>(y,x) < 0)
				{
					depth_data.at<float>(y,x) = depth_data.at<float>(y,x)/(-10);
				}
				else
				{
					depth_data.at<float>(y,x) = depth_data.at<float>(y,x)/10;
				}
			}
		}
	}

	outputply(depth_data);


	cout<<"========================================================" << endl;

	/*Mat afterfilp(partial_derive);
	//開始積分平面,從X方向開始
	Mat f_1(120,120,CV_32FC1);
	for(int y=0;y<120;y++)
	{
		for(int x=0;x<120;x++)
		{
			f_1.at<float>(y,x)=f_u_v_xdirection(y,x);
		}
	}
	//開始積分平面,從X方向開始
	cv::flip(partial_derive,afterfilp,1);
	partial_derive = afterfilp;

	Mat f_2(120,120,CV_32FC1);
	for(int y = 0; y < 120; y++)
	{
		for(int x = 0;x < 120; x++)
		{
			f_2.at<float>(y,x)=f_u_v_xdirection(y,x);
		}
	}
	
	//開始積分平面,從Y方向開始
	Mat f_3(120,120,CV_32FC1);
	for(int y=0;y<120;y++)
	{
		for(int x=0;x<120;x++)
		{
			f_3.at<float>(y,x)=f_u_v_ydirection(y,x);
		}
	}

	//開始積分平面,從Y方向開始
	cv::flip(partial_derive,afterfilp,0);
	partial_derive=afterfilp;
	Mat f_4(120,120,CV_32FC1);
	for(int y=0;y<120;y++)
	{
		for(int x=0;x<120;x++)
		{
			f_4.at<float>(y,x)=f_u_v_ydirection(y,x);
		}
	}


	//移除offset
	/*RemoveOffset(f_1);
	RemoveOffset(f_2);
	RemoveOffset(f_3);
	RemoveOffset(f_4);
	//翻轉使起點一樣
	flip(f_2,'y');
	flip(f_3,'y');
	flip(f_4,'x');
	flip(f_4,'y');
	//經過中值濾波
	medianBlur(f_1,f_1,3);
	medianBlur(f_2,f_2,3);
	medianBlur(f_3,f_3,3);
	medianBlur(f_4,f_4,3);*/

	//outputply(f_2);

	///////////////////////////////

	///////////////////////////////
	///////產生color normal map////
	///////////////////////////////
	
	normal=normal*127;
	normal=normal+Mat(120,120,CV_32FC3,Scalar(127,127,127));
	Mat color_normal_map(120,120,CV_8UC3);
	normal.convertTo(color_normal_map,CV_8UC3);
	imshow("color normal map",color_normal_map);

	cout<<"end"<<endl;


    waitKey(0); // Wait for the user to press a key.
    //return 0;
}
