#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <assert.h>
#include <iostream>
#include "gnss_collect_worker.h"
#include "rs232.h"
#include "gnss.pb.h"
#include "imu.pb.h"

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
    CHECK(RS232_OpenComport(port_, bdrate_, mode, 0) == 0)
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




bool GNSSCollectWorker::Work() {
    INFO_MSG("GNSS worker start");
    CHECK(init_);
    std::vector<unsigned char> buf(buf_size_);
    while (!stopped_) {
        int n = RS232_PollComport(port_, buf.data(), buf_size_);
        auto ms = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
        uint8_t measurement_time = ms.count();
        int skipped_bytes = 0;
        if (n > 0) {
            int idx = 0;
            while (idx < n) {
                std::string content;
                if (buf[idx] == '$') {
                    std::string nema_raw_data;
                    unsigned char check_sum = buf[idx + 1];
                    int idx2 = idx + 2;
                    while (buf[idx2] != '*' && buf[idx2] != 0x0A && buf[idx2] != 0x0D) {
                        nema_raw_data += buf[idx2];
                        check_sum ^= buf[idx2];
                        idx2 ++;
                    }
                    if (buf[idx2] == '*') {
                        idx = idx2 + 4;
                        // TODO CHECKSUM
                        RawData gnss_raw;
                        gnss_raw.mutable_header()->set_timestamp_sec((double)measurement_time / 1000000);
                        gnss_raw.mutable_header()->set_module_name(writer_->ModuleName());
                        gnss_raw.mutable_header()->set_sequence_num(gps_count_++);
                        gnss_raw.mutable_header()->set_camera_timestamp(measurement_time);
                        gnss_raw.set_data(nema_raw_data);
                        gnss_raw.SerializeToString(&content);
                        CHECK(writer_->PushMessage(content, "gnss", measurement_time));
                    } else {
                        skipped_bytes += idx2 - idx;
                        idx = idx2 + 1;
                        ERROR_MSG("nema stop fail" << nema_raw_data);
                    }
                } else if (idx + 76 <= n && buf[idx] == 0xAA && buf[idx + 1] == 0x44 && buf[idx + 1] == 0x12) {
                    Imu imu_data;
                    imu_data.mutable_header()->set_timestamp_sec((double)measurement_time / 1000000);
                    imu_data.mutable_header()->set_module_name(writer_->ModuleName());
                    imu_data.mutable_header()->set_sequence_num(imu_count_++);
                    imu_data.mutable_header()->set_camera_timestamp(measurement_time);
                    imu_data.set_measurement_time((double)measurement_time / 1000000);
                    unsigned char *p_imu = buf.data() + idx + 28;
                    imu_data.mutable_linear_acceleration()->set_x(*(reinterpret_cast<long*>(p_imu + 24)));
                    imu_data.mutable_linear_acceleration()->set_y(-*(reinterpret_cast<long*>(p_imu + 20)));
                    imu_data.mutable_linear_acceleration()->set_x(*(reinterpret_cast<long*>(p_imu + 16)));
                    imu_data.mutable_angular_velocity()->set_x(*(reinterpret_cast<long*>(p_imu + 40)));
                    imu_data.mutable_angular_velocity()->set_y(-*(reinterpret_cast<long*>(p_imu + 36)));
                    imu_data.mutable_angular_velocity()->set_z(*(reinterpret_cast<long*>(p_imu + 32)));
                    imu_data.SerializeToString(&content);
                    CHECK(writer_->PushMessage(content, "imu", measurement_time));
                } else {
                    idx ++;
                    skipped_bytes ++;
                }
            }
        } else {
            usleep(1000);
        }
    }
    return true;
}

