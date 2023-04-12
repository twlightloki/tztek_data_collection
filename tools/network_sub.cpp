#include "zmq.hpp"
#include <iostream>
#include <string>
#include <thread>

int main(int argc, char** argv) {
    zmq::context_t context(1);
    std::cout << zmq_strerror(zmq_errno()) << std::endl;
    zmq::socket_t socket(context, zmq::socket_type::sub);
    std::cout << zmq_strerror(zmq_errno()) << std::endl;
    socket.connect("tcp://localhost:" + argv[1]);
    std::cout << zmq_strerror(zmq_errno()) << std::endl;
    char filter[] = "byd66";
    socket.setsockopt(ZMQ_SUBSCRIBE, filter, 5);
    std::cout << zmq_strerror(zmq_errno()) << std::endl;
    while (true) {
        zmq::message_t request;
        socket.recv(request);
        std::cout<< request.size() << std::endl;
    }


    return 0;
}
