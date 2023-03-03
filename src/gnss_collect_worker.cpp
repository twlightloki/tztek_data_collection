#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <assert.h>
#include <iostream>
#include "gnss_collect_worker.h"
#include "rs232.h"
#include "gnss.pb.h"
#include "imu.pb.h"
#include "pose.pb.h"
#include <fstream>

using namespace std::chrono;
using namespace drivers::gnss;

GNSSCollectWorker::GNSSCollectWorker(const int port, const int bdrate, const std::shared_ptr<PBWriter>& writer):
    port_(port), bdrate_(bdrate),
    writer_(writer) {
    }

GNSSCollectWorker::~GNSSCollectWorker() {
}

bool GNSSCollectWorker::Init() {
    init_ = true;
    char mode[]={'8','N','1',0};
    CHECK(RS232_OpenComport(port_, bdrate_, mode, 0) == 0);
    return true;
}

bool GNSSCollectWorker::Release() {
    CHECK(init_);

    if (worker_.get()) {
        stopped_ = true;
        worker_->join();
    }
    INFO_MSG("GNSS worker Released, gps count: " << gps_count_ << ", imu count: " << imu_count_);

    RS232_CloseComport(port_);
    return true;
}

bool GNSSCollectWorker::Open() {
    INFO_MSG("GNSS worker Open");
    CHECK(init_);
    worker_.reset(new std::thread([this](){this->Work();}));
    return true;
}


double RawImuAccel(double raw) {
    return raw / 9.80665 / 655360 * 100;
}

double RawImuGyro(double raw) {
    return raw / 160849.543863 * 100;
}

bool GNSSCollectWorker::Work() {
    INFO_MSG("GNSS worker start");
    CHECK(init_);
    std::string buf;
    buf.resize(buf_size_);
    while (!stopped_) {
        int n = RS232_PollComport(port_, (unsigned char*)(buf.data()), buf_size_);
        if (n > 0) {
            auto ms = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
            double measurement_time = ms.count() / 1e9;
            int skipped_bytes = 0;

            buf[n]='\0';
            int idx = 0;
            //INFO_MSG("head: " << int(buf[0]) << " " << int(buf[1]) << " " << int(buf[2]));
            while (idx < n) {
                std::string content;
                if (buf[idx] == '$') {
                    std::string nema_raw_data;
                    unsigned char checksum = 0;
                    int idx2 = idx + 1;
                    std::vector<std::string> split_result;
                    int last = 0;
                    while (buf[idx2] != '*' && buf[idx2] != 0x0A && buf[idx2] != 0x0D) {
                        if (buf[idx2] == ',') {
                            split_result.push_back(buf.substr(last, idx2 - last));
                            last = idx2 + 1;
                        }
                        if (idx2 > idx + 1) {
                            checksum ^= buf[idx2];
                        } else {
                            checksum = buf[idx2];
                        }
                        idx2 ++;
                    }
                    nema_raw_data = buf.substr(idx, idx2 - idx);
                    if (last < idx2) {
                        split_result.push_back(buf.substr(last, idx2 - last));
                    }
                    bool nema_checksum = true;
                    int gt_checksum = -1;
                    try {
                        gt_checksum = std::stoi(buf.substr(idx2 + 1, 2), nullptr, 16);
                    } catch (...) {
                        nema_checksum = false;
                    }

                    if (buf[idx2] == '*' && nema_checksum && gt_checksum == checksum && split_result.size() == 24) {
                        idx = idx2 + 5;

                        //gnss
                        Gnss gnss_data;
                        gnss_data.mutable_header()->set_timestamp_sec(measurement_time);
                        gnss_data.mutable_header()->set_module_name(writer_->ModuleName());
                        gnss_data.mutable_header()->set_sequence_num(gps_count_);
                        gnss_data.set_measurement_time(measurement_time);
                        try {

                            gnss_data.mutable_orientation()->set_x(std::stof(split_result[3]));
                            gnss_data.mutable_orientation()->set_y(std::stof(split_result[4]));
                            gnss_data.mutable_orientation()->set_z(std::stof(split_result[5]));
                            gnss_data.mutable_gyro()->set_x(std::stof(split_result[6]));
                            gnss_data.mutable_gyro()->set_y(std::stof(split_result[7]));
                            gnss_data.mutable_gyro()->set_z(std::stof(split_result[8]));
                            gnss_data.mutable_accel()->set_x(std::stof(split_result[9]));
                            gnss_data.mutable_accel()->set_y(std::stof(split_result[10]));
                            gnss_data.mutable_accel()->set_z(std::stof(split_result[11]));





                            gnss_data.mutable_position()->set_lon(std::stof(split_result[13]));
                            gnss_data.mutable_position()->set_lat(std::stof(split_result[12]));
                            gnss_data.mutable_position()->set_height(std::stof(split_result[14]));
                            gnss_data.mutable_linear_velocity()->set_x(std::stof(split_result[15]));
                            gnss_data.mutable_linear_velocity()->set_y(std::stof(split_result[16]));
                            gnss_data.mutable_linear_velocity()->set_z(std::stof(split_result[17]));
                            gnss_data.set_num_sats(std::stoi(split_result[19]) + std::stoi(split_result[20]));
                        } catch (...) {
                            ERROR_MSG("nema decode fail" << nema_raw_data);
                        }
                        gnss_data.SerializeToString(&content);
                        CHECK(writer_->PushMessage(content, "gnss", measurement_time));
                        RawData gnss_raw;
                        gnss_raw.mutable_header()->set_timestamp_sec(measurement_time);
                        gnss_raw.mutable_header()->set_module_name(writer_->ModuleName());
                        gnss_raw.mutable_header()->set_sequence_num(gps_count_);
                        gnss_raw.set_data(nema_raw_data);
                        INFO_MSG("nema: "  << uint64_t(measurement_time) << " " << nema_raw_data);
                        gnss_raw.SerializeToString(&content);
                        CHECK(writer_->PushMessage(content, "gnss_raw", measurement_time));

                        //pose
                        gps_count_++;
                    } else {
                        skipped_bytes += idx2 - idx;
                        idx = idx2 + 1;
                    }
                } else if (idx + 73 <= n && buf[idx] == 0xAA && buf[idx + 1] == 0x44 && buf[idx + 2] == 0x12) {
                    Imu imu_data;
                    imu_data.mutable_header()->set_timestamp_sec(measurement_time);
                    imu_data.mutable_header()->set_module_name(writer_->ModuleName());
                    imu_data.mutable_header()->set_sequence_num(imu_count_);
                    imu_data.set_measurement_time(measurement_time);
                    const char *p_imu = buf.data() + idx + int(buf[idx + 3]);
                    imu_data.mutable_linear_acceleration()->set_x(RawImuAccel(reinterpret_cast<const int32_t*>(p_imu + 24)[0]));
                    imu_data.mutable_linear_acceleration()->set_y(-RawImuAccel(reinterpret_cast<const int32_t*>(p_imu + 20)[0]));
                    imu_data.mutable_linear_acceleration()->set_z(RawImuAccel(reinterpret_cast<const int32_t*>(p_imu + 16)[0]));
                    imu_data.mutable_angular_velocity()->set_x(RawImuGyro(reinterpret_cast<const int32_t*>(p_imu + 36)[0]));
                    imu_data.mutable_angular_velocity()->set_y(RawImuGyro(-reinterpret_cast<const int32_t*>(p_imu + 32)[0]));
                    imu_data.mutable_angular_velocity()->set_z(RawImuGyro(reinterpret_cast<const int32_t*>(p_imu + 28)[0]));

                    if (imu_count_ % 100 == 0) {
                        INFO_MSG("imu: " << uint64_t(measurement_time) << " : " << 
                        imu_data.linear_acceleration().x() << " " << imu_data.linear_acceleration().y() << " " << imu_data.linear_acceleration().z() << " : " << 
                        imu_data.angular_velocity().x() <<  " " << imu_data.angular_velocity().y() << " " << imu_data.angular_velocity().z());
                    }
                    imu_data.SerializeToString(&content);
                    CHECK(writer_->PushMessage(content, "imu", measurement_time));

                    RawData imu_raw;
                    imu_raw.mutable_header()->set_timestamp_sec(measurement_time);
                    imu_raw.mutable_header()->set_module_name(writer_->ModuleName());
                    imu_raw.mutable_header()->set_sequence_num(imu_count_);
                    imu_raw.set_data(buf.substr(idx, 73));
                    imu_raw.SerializeToString(&content);
                    CHECK(writer_->PushMessage(content, "imu_raw", measurement_time));

                    imu_count_ ++;
                    idx += 73;
                } else {
                    skipped_bytes ++;
                    idx ++;
                }
            }
            if (skipped_bytes > 0) {
                INFO_MSG("n " << n << " skip byte: " << skipped_bytes);
            }
        } else {
            usleep(1000);
        }
    }
    return true;
}
