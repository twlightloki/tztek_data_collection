#pragma once
#include "data_consumer.h"
#include "common.h"
#include <chrono>



class GNSSCollectWorker {
    public:
        GNSSCollectWorker(const int port, const int bdrate, const std::shared_ptr<DataWriter> &writer);
        ~GNSSCollectWorker();
        bool Release();
        bool Open();
        bool Init();

    private:
        bool Work();
        std::atomic<bool> stopped_{false};
        bool init_{false};
        std::shared_ptr<std::thread> worker_;
        int gps_count_{0};
        int imu_count_{0};
        int buf_size_{4096};
        int port_;
        int bdrate_;
        std::shared_ptr<DataWriter> writer_;
};

