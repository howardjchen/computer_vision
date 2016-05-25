#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include "NumericalIntegration.cpp"
#define INTEGRAL_STEP 2
using namespace cv;
using namespace std;


void OutputAsMatlabCode(Mat &f,string filename)
{
	ofstream file;
	file.open(filename,ios::out);
	file<<"figure;surf([";
	for(int y=0;y<120;y++)
	{
		for(int x=0;x<120;x++)
		{
			if(f.at<float>(y,x)<20&&f.at<float>(y,x)>-10)
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
double f_u_v_ydirection(double v,double u_)
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
double f_u_v_xdirection(double v_,double u)
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
void RemoveSpecular(Mat &m,int threshold_x,int threshold_y,int threshold_z)
{
	for(int z=0;z<3;z++)
	{
		for(int y=0;y<120;y++)
		{
			
			for(int x=0;x<120;x++)
			{
				
				if(z==0)
				{
					if((int)m.at<Vec3b>(y,x)[z]>threshold_x)
					{
						m.at<Vec3b>(y,x)[z]=threshold_x;
					}
				}
				else if(z==1)
				{
					if((int)m.at<Vec3b>(y,x)[z]>threshold_y)
					{
						m.at<Vec3b>(y,x)[z]=threshold_y;
					}
				}
				else if(z==2)
				{
					if((int)m.at<Vec3b>(y,x)[z]>threshold_z)
					{
						m.at<Vec3b>(y,x)[z]=threshold_z;
					}
				}
			}
		}
	}

}
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
int main( int argc, char** argv )
{
	
	//Mat picture=imread("test.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//imshow("test",picture);
	//cout<<picture.rows<<" "<<picture.cols<<endl;
	
	//用來轉換字串
	stringstream ss;
	string pic_name;
	//儲存圖片的矩陣
	Mat pic[6];
	//讀入六張圖並顯示
	for(int i=0;i<6;i++)
	{
		ss.clear();
		ss.str(std::string());
		ss<<"apple"<<i<<".bmp"<<endl;
		ss>>pic_name;
		//以灰階圖讀入
		pic[i] = imread(pic_name, CV_LOAD_IMAGE_COLOR);
		//if(! pic[i].data )
		imshow(pic_name,pic[i]);
		//namedWindow( pic_name, CV_WINDOW_AUTOSIZE );
		// Create a window for display.
		RemoveSpecular(pic[i],140,105,150);
		//cout<<type2str(pic[i].type())<<endl;
		imwrite("de-specular"+pic_name,pic[i]);
		imshow(pic_name,pic[i]);
	}
	Mat pic_2[6];
	for(int i=0;i<6;i++)
	{
		ss.clear();
		ss.str(std::string());
		ss<<"de-specularapple"<<i<<".bmp"<<endl;
		ss>>pic_name;
		//以灰階圖讀入
		pic_2[i] = imread(pic_name, CV_LOAD_IMAGE_GRAYSCALE);
		//imshow(pic_name,pic_2[i]);
	}


	




	//cout<<pic[0]<<endl;


	
	//事先用matlab 算出pseudo inverse那串矩陣
	//float Array[18]={  -0.004520786201502 ,  0.010567032169654,  -0.006686047906550 , -0.008125182562643 ,  0.001998884506948 ,  0.006652883154767,
	//				  -0.019177520525019  , 0.023721010273291  , 0.013503608213505  ,-0.011538670200703  ,-0.002899015151684  ,-0.004410709182530,
	//				   0.002924817082408  ,-0.001328892845353 , -0.000293629534994  , 0.002211217475984  , 0.001304151598502  , 0.001581694334445};
	//Mat UTU_1UT(3,6,CV_32FC1,Array);
	float Array[18]={ 30,60,9,
						30,50,9,
						30,40,9,
						30,20,9,
						30,10,9};

	Mat U(6,3,CV_32FC1,Array);
	Mat U_trans(3,6,CV_32FC1);
	transpose(U,U_trans);
	Mat UTU_1UT=Mat(U_trans*U).inv(DECOMP_SVD)*U_trans;
	//檢查
	//cout<<UTU_1UT<<endl;

	//cout<<"UTU_1UT"<<UTU_1UT<<endl;
	//3 channel normal (未被normalize)存法向量的x y z
	Mat b(120,120,CV_32FC3);
	Mat normal(120,120,CV_32FC3);
	Mat rho(120,120,CV_32FC1);
	
	//cout<<normal.at<Vec3f>(10,10)[0];
	//共120*120個點
	//cout<<UTU_1UT<<endl;
	//cout<<(*GetIntensity(pic,60,60))<<endl;
	//cout<<UTU_1UT*(*GetIntensity(pic,60,60))<<endl;

	//cout<<"surface normal:"<<endl;
	for(int y=0;y<120;y++)
	{
		
		for(int x=0;x<120;x++)
		{
			
			Mat tmp_for_get_element(UTU_1UT*(*GetIntensity(pic_2,y,x)));
			for(int z=0;z<3;z++)
			{
				//將得到的向量存到b 
				b.at<Vec3f>(y,x)[z]=tmp_for_get_element.at<float>(z);
			}
			//將b拆成 N ,rho
			//normal是向量 用vec3f存
			normal.at<Vec3f>(y,x)=cv::normalize(b.at<Vec3f>(y,x));
			//rho是b的長度,用一個float存就好
			rho.at<float>(y,x)=cv::norm(b.at<Vec3f>(y,x));
			//normalize後得到	(∂f/∂x,  ∂f/∂y,1)
			//[2]就是Nc,
			partial_derive.at<Vec3f>(y,x)=normal.at<Vec3f>(y,x)/normal.at<Vec3f>(y,x)[2];
		}
	}

	cout<<"partial_derive: "<<partial_derive.size()<<endl;
	cout<<"pic[0]: "<<pic[0].size()<<endl;


	//經過中值濾波
	//medianBlur(partial_derive,partial_derive,3);


	//for(int i=0;i<120;i++)
	//	cout<<partial_derive.at<Vec3f>(60,i)[1]<<endl;

	Mat afterfilp(partial_derive);
	//開始積分平面,從X方向開始
	Mat f_1(120,120,CV_32FC1);
	cout<<"//開始積分平面,方向 (1)"<<endl;
	for(int y=1;y<120;y++)
	{
		for(int x=0;x<120;x++)
		{
			//cout<<y<<" "<<x<<endl;
			f_1.at<float>(y,x)=f_u_v_xdirection(y,x);
		}
	}
	cout<<partial_derive.size()<<endl;
	cout<<"//開始積分平面,方向 (2)"<<endl;
	//開始積分平面,從X方向開始
	cv::flip(partial_derive,afterfilp,1);
	partial_derive=afterfilp;
	Mat f_2(120,120,CV_32FC1);
	for(int y=0;y<120;y++)
	{
		for(int x=0;x<120;x++)
		{
			f_2.at<float>(y,x)=f_u_v_xdirection(y,x);
		}
	}
	
	cout<<"//開始積分平面,方向 (3)"<<endl;
	//開始積分平面,從Y方向開始
	Mat f_3(120,120,CV_32FC1);
	for(int y=0;y<120;y++)
	{
		for(int x=0;x<120;x++)
		{
			f_3.at<float>(y,x)=f_u_v_ydirection(y,x);
		}
	}
	
	cout<<"//開始積分平面,方向 (4)"<<endl;
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
	cout<<"//移除offset"<<endl;
	RemoveOffset(f_1);
	RemoveOffset(f_2);
	RemoveOffset(f_3);
	RemoveOffset(f_4);
	//翻轉使起點一樣
	cout<<"//翻轉使起點一樣"<<endl;
	flip(f_2,'y');
	flip(f_3,'y');
	flip(f_4,'x');
	flip(f_4,'y');
	//經過中值濾波
	cout<<"//經過中值濾波"<<endl;
	medianBlur(f_1,f_1,3);
	medianBlur(f_2,f_2,3);
	medianBlur(f_3,f_3,3);
	medianBlur(f_4,f_4,3);
	//將結果以matlab 矩陣格式輸出,以檢查
	cout<<"//將結果以matlab 矩陣格式輸出,以檢查"<<endl;
	OutputAsMatlabCode(UpsideDown(f_1),"matrix_01.txt");
	OutputAsMatlabCode(           f_2,"matrix_02.txt");
	OutputAsMatlabCode(             f_3,"matrix_03.txt");
	OutputAsMatlabCode(UpsideDown(f_4),"matrix_04.txt");

	///////////////////////////////
	///////產生color normal map////
	///////////////////////////////
	cout<<"//產生color normal map"<<endl;
	normal=normal*127;
	normal=normal+Mat(120,120,CV_32FC3,Scalar(127,127,127));
	Mat color_normal_map(120,120,CV_8UC3);
	normal.convertTo(color_normal_map,CV_8UC3);
	imshow("color normal map",color_normal_map);
	
	

	/*////////////////////////////////
	/////remove  ball  specular/////
	////////////////////////////////
	ofstream file;
	Mat specular=imread("specular.bmp",CV_LOAD_IMAGE_COLOR);
	imshow("specular",specular);
	file.open("specular_rgbspace.txt",ios::out);
	//將結果以matlab 矩陣格式輸出,以檢查
	for(int z=0;z<3;z++)
	{
		file<<"specular"<<"(:,:,"<<z+1<<")=[";
		for(int y=0;y<120;y++)
		{
			for(int x=0;x<120;x++)
			{
				file<<(int)specular.at<Vec3b>(y,x)[z]<<" ";
				if(x==119)
					file<<";";
			}
			file<<"\n";
		}
		file<<"];";
	}
	file<<"x=specular(:,:,1)"<<endl;
	file<<"y=specular(:,:,2)"<<endl;
	file<<"z=specular(:,:,3)"<<endl;
	file<<"x=x(:);"<<endl;
	file<<"y=y(:);"<<endl;
	file<<"z=z(:);"<<endl;
	file<<"scatter3(x,y,z);"<<endl;
	file.close();


	//imshow("remove specular",specular);


	
	////////////////////////////////
	////remove  orange  specular////
	////////////////////////////////
	Mat speclar_apple=imread("speclar_apple.bmp",CV_LOAD_IMAGE_COLOR);
	imshow("speclar_apple",speclar_apple);
	file.open("speclar_apple_rgbspace.txt",ios::out);
	//將結果以matlab 矩陣格式輸出,以檢查
	for(int z=0;z<3;z++)
	{
		file<<"specular"<<"(:,:,"<<z+1<<")=[";
		for(int y=0;y<120;y++)
		{
			for(int x=0;x<120;x++)
			{
				file<<(int)speclar_apple.at<Vec3b>(y,x)[z]<<" ";
				if(x==119)
					file<<";";
			}
			file<<"\n";
		}
		file<<"];";
	}
	file<<"x=specular(:,:,1);"<<endl;
	file<<"y=specular(:,:,2);"<<endl;
	file<<"z=specular(:,:,3);"<<endl;
	file<<"x=x(:);"<<endl;
	file<<"y=y(:);"<<endl;
	file<<"z=z(:);"<<endl;
	file<<"scatter3(x,y,z);"<<endl;
	file.close();
	*/

	/////////////////////////
	/////remove specular/////
	/////////////////////////



	
	cout<<"end"<<endl;
    waitKey(0);
    return 0;
}


//////////////////////
///Matlab code////////
//////////////////////
/*

A_img=((A+20)./70).*254;
A_img=uint8(ceil(A_img));
//顯示深度圖
imshow(A_img);

H = fspecial('gaussian');
A_gaussian = imfilter(A_img,H,'replicate');
//使用高斯濾波
imshow(A_gaussian);
//使用中值濾波
imshow(medfilt2(A_img,[3 7]));

figure;imshow((medfilt2(A_img,[7 3])))

surf2stl('bunny_original.stl',1,1,A);
surf2stl('bunny_gaussian.stl',1,1,double(A_gaussian));
surf2stl('bunny_medfilt.stl',1,1,double(medfilt2(A_img)));


A_dft = fft2(A_img); 
A_dft_Mag = abs(20.*log(1.+double(A_dft))); 
A_dft_Mag =fftshift(A_dft_Mag);
imshow(uint8(A_dft_Mag));


Ax 平面的系數    
0.367844661688861
  -0.283776614579324
  -0.083806173504734

Ay
x =

   0.367896057347670
  -0.283819354838710
  -0.082963440860212


for ii=1:120
X(ii,:)=[1:120];
end
for ii=1:120
Y(:,ii)=[1:120]';
end

 Zx=0.367844661688861*X+-0.283776614579324*Y+-0.083806173504734;
 Zy=0.367896057347670*X+-0.283819354838710*Y+-0.082963440860212;

*/