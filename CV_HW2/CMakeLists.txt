cmake_minimum_required(VERSION 2.8)
project( hw2 )
find_package( OpenCV REQUIRED )
add_executable( hw2 SIFT_opencv2.4x.cpp )
target_link_libraries( hw2 ${OpenCV_LIBS} )

