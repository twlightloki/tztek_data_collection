CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(camera_collect)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(Protobuf REQUIRED)

ENABLE_LANGUAGE(CUDA)


INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
SET(CUDA_SEPARABLE_COMPILATION ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -std=c++11 -Wall")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=compute_87 -code=sm_87")
LINK_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
LINK_DIRECTORIES(/usr/lib/aarch64-linux-gnu/tegra/)
LINK_DIRECTORIES(/usr/lib/aarch64-linux-gnu/)

INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIRS})

SET(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${SRC_DIR})
LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/lib/)

SET(JETSON_ROOT_DIR /usr/src/jetson_multimedia_api/)
INCLUDE_DIRECTORIES(${JETSON_ROOT_DIR}/include)
INCLUDE_DIRECTORIES(${JETSON_ROOT_DIR}/include/libjpeg-8b)

SET(PROTO_DIR ${CMAKE_SOURCE_DIR}/protos)
SET(PROTO_FILE ${PROTO_DIR}/error_code.proto
               ${PROTO_DIR}/header.proto
               ${PROTO_DIR}/sensor_image.proto
               ${PROTO_DIR}/message.proto)
PROTOBUF_GENERATE_CPP(PROTO_C PROTO_H ${PROTO_FILE})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})


SET(JETSON_CLASS_DIR ${JETSON_ROOT_DIR}/samples/common/classes)
SET(SRC_FILE ${PROTO_C}
             ${SRC_DIR}/camera_collect_worker.cpp
             ${SRC_DIR}/gnss_collect_worker.cpp
             ${SRC_DIR}/pb_writer.cpp
			 ${SRC_DIR}/cfg.cpp
			 ${SRC_DIR}/rs232.c
             ${JETSON_CLASS_DIR}/NvLogging.cpp
             ${JETSON_CLASS_DIR}/NvBuffer.cpp
             ${JETSON_CLASS_DIR}/NvJpegEncoder.cpp
             ${JETSON_CLASS_DIR}/NvElement.cpp
             ${JETSON_CLASS_DIR}/NvElementProfiler.cpp)

ADD_LIBRARY(camera_collect_worker SHARED ${SRC_FILE})
ADD_EXECUTABLE(camera_collect main.cpp)
TARGET_LINK_LIBRARIES(camera_collect  camera_collect_worker syncv4l2 yuv nvjpeg protobuf)

