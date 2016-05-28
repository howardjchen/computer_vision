cmake_minimum_required(VERSION 2.8)
project( hw2 )
find_package( OpenCV REQUIRED )
add_executable( hw2 main.cpp )
target_link_libraries( hw2 ${OpenCV_LIBS} )


