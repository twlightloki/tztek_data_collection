#include "zmq.hpp"
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>

int main(int argc, char** argv) {
    std::string port = argv[1];
    std::string msg = argv[2];
    zmq::context_t context(1);
    zmq::socket_t socket(context, zmq::socket_type::req);
    socket.connect(std::string("tcp://localhost:") + port);
    zmq::const_buffer buf(msg.data(), msg.size());
    socket.send(buf);
    std::cout << "send msg: " << msg << std::endl;
    zmq::message_t request;
    zmq::recv_result_t rtn = socket.recv(request);
    std::cout << "recv[" << *rtn << "]: " << request.to_string() << std::endl;
    return 0;
}
