cmake_minimum_required(VERSION 3.1)
project(yolo_trt VERSION 1.0)
enable_language(CUDA)

set(CMAKE_CXX_COMPILIER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -Wno-write-strings")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

#cuda
#export PATH=/usr/local/cuda-11.0/bin:$PATH
#include_directories(/usr/local/cuda/include)
#link_directories(/usr/local/cuda/lib64)
find_package(CUDA REQUIRED)

#opencv
#set(OpenCV_DIR /opt/opencv-4.1.2/share/OpenCV)
find_package(OpenCV 4 REQUIRED)

#detector lib (cuda)
file(GLOB_RECURSE sources src/*.cpp src/*.cu)
add_library(${PROJECT_NAME} SHARED ${sources})
target_include_directories(
	${PROJECT_NAME}
	PRIVATE
	include/
	${OpenCV_INCLUDE_DIRS}
	${CUDA_TOOLKIT_ROOT_DIR}/include
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	)
target_link_libraries(
	${PROJECT_NAME}
	nvinfer
	nvinfer_plugin
	nvcaffe_parser
	"stdc++fs"
	${OpenCV_LIBS}
	)

#sample
add_executable(${PROJECT_NAME}_sample samples/sample_detector.cpp)
target_include_directories(${PROJECT_NAME}_sample PRIVATE include/)
target_link_libraries(${PROJECT_NAME}_sample ${PROJECT_NAME} ${OpenCV_LIBS})
