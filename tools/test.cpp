#include <iostream>
#include <string>
#include <thread>
#include <fstream>
#include "message.pb.h"

#include "sensor_image.pb.h"
#include "gnss.pb.h"
#include "imu.pb.h"
#include "pose.pb.h"
#include "pointcloud.pb.h"





int main(int argc, char** argv) {

    std::ifstream inf(argv[1], std::ios::binary);

    common::Chunk chunk;
    chunk.ParseFromIstream(&inf);
    std::cout << chunk.messages_size() << std::endl;
    for (int i1 = 0; i1 < chunk.messages_size(); i1 ++) {
        const common::SingleMessage &message = chunk.messages(i1);
        if (message.sensor_name() == "gnss_raw") {
            drivers::gnss::RawData raw;
            raw.ParseFromString(message.content());
            std::cout << "data len: " << raw.data().size() << std::endl;
            uint8_t check_sum = 0;
            for (int i2 = 0; i2 < raw.data().size(); i2 ++) {
                check_sum ^= raw.data()[i2];
                std::cout << i2 << " " << int(raw.data()[i2]) << " " << int(check_sum) << std::endl;
                for (int i3 = 7; i3 >= 0; i3 --) {
                    std::cout << int(raw.data()[i2] >> i3 & 1);
                }
                std::cout << " ";
                for (int i3 = 7; i3 >= 0; i3 --) {
                    std::cout << int(check_sum >> i3 & 1);
                }
                std::cout << std::endl;
 
            }
        }
    }

    return 0;
}
