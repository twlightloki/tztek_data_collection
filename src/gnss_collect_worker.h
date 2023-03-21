#pragma once
#include "pb_writer.h"
#include "common.h"
#include <chrono>



class GNSSCollectWorker {
    public:
        GNSSCollectWorker(const int port, const int bdrate, const std::shared_ptr<PBWriter> &writer);
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
        int port_;
        int bdrate_;
        std::shared_ptr<PBWriter> writer_;
};

