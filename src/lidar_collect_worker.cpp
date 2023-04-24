#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <assert.h>
#include <iostream>
#include "lidar_collect_worker.h"
#include "pointcloud.pb.h"

using namespace std::chrono;
using namespace drivers;

LidarCollectWorker::LidarCollectWorker(const std::string& channel, const std::shared_ptr<DataWriter>& writer):
    channel_(channel),
    writer_(writer) {
    }

LidarCollectWorker::~LidarCollectWorker() {
}

bool LidarCollectWorker::Init() {
    int argc = 0;
    char** argv = nullptr;
    ros::init(argc, argv, writer_->ModuleName());
    init_ = true;
    return true;
}

bool LidarCollectWorker::Release() {
    ros::shutdown();
    ros_thread_->join();
    INFO_MSG("Lidar worker Released, lidar count: " << lidar_count_);
    return true;
}

bool LidarCollectWorker::Open() {
    INFO_MSG("Lidar worker Open");
    CHECK(init_);
    ros_thread_.reset(new std::thread([this](){this->RosSpin();}));
    return true;
}

void LidarCollectWorker::RosSpin() {
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe(channel_, 1, &LidarCollectWorker::LaserCallback, this);
    ros::spin();
}


void LidarCollectWorker::LaserCallback(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    ros::Time timestamp = msg->header.stamp;
    double measurement_time = timestamp.toSec();
   
    PointCloud pc;
    pc.mutable_header()->set_timestamp_sec(measurement_time);
    pc.mutable_header()->set_module_name(writer_->ModuleName());
    pc.mutable_header()->set_sequence_num(lidar_count_++);
    pc.mutable_header()->set_lidar_timestamp(measurement_time * 1e9);
    pc.set_frame_id(msg->header.frame_id);
    pc.set_height(msg->height);
    pc.set_width(msg->width);
    pc.set_measurement_time(measurement_time);
    pc.set_is_dense(msg->is_dense);
    
    for (int i1 = 0; i1 < int(msg->height * msg->width); i1 ++) {
      const uint8_t* data = msg->data.data() + i1 * msg->point_step;
      float x = reinterpret_cast<const float*>(data)[0];
      float y = reinterpret_cast<const float*>(data + 4)[0];
      float z = reinterpret_cast<const float*>(data + 8)[0];
      if (isnan(x) || isnan(y) || isnan(z)) {
          continue;
      }
      PointXYZIRT *point = pc.add_point();
      point->set_x(x);
      point->set_y(y);
      point->set_z(z);
      point->set_intensity(reinterpret_cast<const float*>(data + 12)[0]);
      point->set_ring(reinterpret_cast<const uint16_t*>(data + 16)[0]);
      point->set_timestamp(reinterpret_cast<const double*>(data + 18)[0]);
 
    }

    std::string content;
    pc.SerializeToString(&content);
    writer_->PushMessage(content, "lidar", measurement_time);
}

