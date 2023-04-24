#include "zmq.hpp"
#include <iostream>
#include <string>
#include <thread>
#include "message.pb.h"

#include "sensor_image.pb.h"
#include "gnss.pb.h"
#include "imu.pb.h"
#include "pose.pb.h"
#include "pointcloud.pb.h"


template <typename T> std::string Parse(const common::SingleMessage &message) {
    T data;
    data.ParseFromString(message.content());
    return data.DebugString();
}

template <> std::string Parse<drivers::CompressedImage>(const common::SingleMessage &message) {
    drivers::CompressedImage data;
    data.ParseFromString(message.content());
    data.clear_data();
    return data.DebugString();
}

template <> std::string Parse<drivers::PointCloud>(const common::SingleMessage &message) {
    drivers::PointCloud data;
    data.ParseFromString(message.content());
    data.clear_point();
    return data.DebugString();
}


int main(int argc, char** argv) {
    zmq::context_t context(1);
    std::cout << zmq_strerror(zmq_errno()) << std::endl;
    zmq::socket_t socket(context, zmq::socket_type::sub);
    std::cout << zmq_strerror(zmq_errno()) << std::endl;
    socket.connect(std::string("tcp://localhost:") + argv[1]);
    std::cout << zmq_strerror(zmq_errno()) << std::endl;
    char filter[] = "byd66";
    socket.set(zmq::sockopt::subscribe, filter);
    std::cout << zmq_strerror(zmq_errno()) << std::endl;
    std::string sensor_name = argv[2];
    while (true) {
        zmq::message_t request;
        zmq::recv_result_t rtn = socket.recv(request);
        common::SingleMessage message;
        message.ParseFromString(request.to_string().substr(6, request.size() - 6));
        if (message.sensor_name() == sensor_name) {
            std::string debug_str;
            if (sensor_name == "camera" || sensor_name == "camera_visual") {
                debug_str = Parse<drivers::CompressedImage>(message);
            } else if (sensor_name == "gnss") {
                debug_str = Parse<drivers::gnss::Gnss>(message);
            } else if (sensor_name == "lidar") {
                debug_str = Parse<drivers::PointCloud>(message);
            }
            std::cout << "\x1B[2J\x1B[H" << "message size: " << request.size() << std::endl << debug_str << std::endl;
        }    
    }

    return 0;
}
