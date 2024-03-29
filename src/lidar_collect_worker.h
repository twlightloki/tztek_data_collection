#pragma once
#include "data_consumer.h"
#include "common.h"
#include <chrono>
#include "ros/ros.h"
#include <sensor_msgs/PointCloud2.h>

class LidarCollectWorker {
    public:
        LidarCollectWorker(const std::string& channel, const std::shared_ptr<DataWriter> &writer);
        ~LidarCollectWorker();
        bool Release();
        bool Open();
        bool Init();

        void LaserCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);

    private:

        void RosSpin();
       
        std::string channel_;
        bool init_{false};
        int lidar_count_{0};
        std::shared_ptr<std::thread> ros_thread_;
        std::shared_ptr<DataWriter> writer_;
};

