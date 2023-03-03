#pragma once
#include "pb_writer.h"
#include "common.h"
#include <chrono>
#include "ros/ros.h"
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include "pcl/point_cloud.h"
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl_conversions/pcl_conversions.h>



class LidarCollectWorker {
    public:
        LidarCollectWorker(const std::string& channel, const std::shared_ptr<PBWriter> &writer);
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
        std::shared_ptr<PBWriter> writer_;
};

