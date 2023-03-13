CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(camera_collect)

FIND_PACKAGE(CUDA REQUIRED)

ENABLE_LANGUAGE(CUDA)


INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
SET(CUDA_SEPARABLE_COMPILATION ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -std=c++11 -Wall")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=compute_87 -code=sm_87")
LINK_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
LINK_DIRECTORIES(/usr/lib/aarch64-linux-gnu/tegra/)
LINK_DIRECTORIES(/usr/lib/aarch64-linux-gnu/)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/)
LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/lib/)

SET(JETSON_ROOT_DIR /usr/src/jetson_multimedia_api/)
INCLUDE_DIRECTORIES(${JETSON_ROOT_DIR}/include)
INCLUDE_DIRECTORIES(${JETSON_ROOT_DIR}/include/libjpeg-8b)

SET(JETSON_CLASS_DIR ${JETSON_ROOT_DIR}/samples/common/classes)
SET(SRC_FILE src/camera_collect_worker.cpp
			 src/cfg.cpp
             ${JETSON_CLASS_DIR}/NvLogging.cpp
             ${JETSON_CLASS_DIR}/NvBuffer.cpp
             ${JETSON_CLASS_DIR}/NvJpegEncoder.cpp
             ${JETSON_CLASS_DIR}/NvElement.cpp
             ${JETSON_CLASS_DIR}/NvElementProfiler.cpp)

ADD_LIBRARY(camera_collect_worker SHARED ${SRC_FILE})
ADD_EXECUTABLE(camera_collect main.cpp)
TARGET_LINK_LIBRARIES(camera_collect  camera_collect_worker syncv4l2 yuv nvjpeg)

