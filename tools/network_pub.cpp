#include "zmq.hpp"
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>

int main(int argc, char** argv) {
    zmq::context_t context(1);
    std::cout << zmq_strerror(zmq_errno()) << std::endl;
    zmq::socket_t socket(context, zmq::socket_type::pub);
    std::cout << zmq_strerror(zmq_errno()) << std::endl;
    socket.bind("tcp://*:5556");
    std::cout << zmq_strerror(zmq_errno()) << std::endl;
    while (true) {
        std::string msg = "byd66 hello world";
        zmq::message_t message(msg.size());
        memcpy(message.data(), msg.data(), msg.size());
        socket.send(message);
        usleep(100000);
    }


    return 0;
}
