#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <assert.h>
#include <iostream>
#include "gnss_collect_worker.h"
#include "rs232.h"

using namespace std::chrono;

GNSSCollectWorker::GNSSCollectWorker(const int port, const int bdrate, const std::shared_ptr<PBWriter>& writer):
    port_(port), bdrate_(bdrate),
    writer_(writer) {
    }

GNSSCollectWorker::~GNSSCollectWorker() {
}

bool GNSSCollectWorker::Init() {
    init_ = true;
    char mode[]={'8','N','1',0};
    CHECK(RS232_OpenComport(port_, bdrate_, mode, 0))
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
    std::vector<unsigned char> buf(4096);
    while (!stopped_) {
        int n = RS232_PollComport(port_, buf.data(), 4095);
        if (n > 0) {
            INFO_MSG("len: " << n); 
            buf[n] = '\0';
            uint8_t *u8 = buf.data();
            INFO_MSG("mes0: " << *(reinterpret_cast<long*>(u8 + 28)));
            INFO_MSG("mes1: " << *(reinterpret_cast<double*>(u8 + 32)));
            RS232_flushRX(port_);
        } else {
            usleep(1000);
        }
    }
    return true;
}

