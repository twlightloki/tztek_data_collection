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

double GPStoUTC(int gps_week, double gps_sec) {
    double weeks = gps_week * 604800 + 315936000;
    return gps_sec + weeks - 18;
}

template<typename T>
double asensing_data_convert(const char* data, double coefficient) {
    return double(reinterpret_cast<const T*>(data)[0]) * coefficient;
}

bool GNSSCollectWorker::Work() {
    INFO_MSG("GNSS worker start");
    CHECK(init_);
    std::string buf;
    buf.resize(buf_size_);
    //std::ofstream ouf("gnss.txt");
    int skipped_bytes = 0;
    int read_bytes = 0;
    while (!stopped_) {
        int n = RS232_PollComport(port_, (unsigned char*)(buf.data()), buf_size_);
        if (n > 0) {
            read_bytes += n;
            //auto ms = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());

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

                        double measurement_time = -1;
                        //gnss
                        Gnss gnss_data;
                        try {
                           
                            int32_t gps_week = std::stof(split_result[1]); 
                            double gps_sec = std::stof(split_result[2]);
                            measurement_time = GPStoUTC(gps_week, gps_sec);
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
                        gnss_data.mutable_header()->set_timestamp_sec(measurement_time);
                        gnss_data.mutable_header()->set_module_name(writer_->ModuleName());
                        gnss_data.mutable_header()->set_sequence_num(gps_count_);
                        gnss_data.set_measurement_time(measurement_time);
                        gnss_data.SerializeToString(&content);
                        CHECK(writer_->PushMessage(content, "gnss", measurement_time));
                        RawData gnss_raw;
                        gnss_raw.mutable_header()->set_timestamp_sec(measurement_time);
                        gnss_raw.mutable_header()->set_module_name(writer_->ModuleName());
                        gnss_raw.mutable_header()->set_sequence_num(gps_count_);
                        gnss_raw.set_data(nema_raw_data);
                        //INFO_MSG("nema: "  << uint64_t(measurement_time * 1000) << " " << nema_raw_data);
                        //ouf << nema_raw_data << "\n";
                        gnss_raw.SerializeToString(&content);
                        CHECK(writer_->PushMessage(content, "gnss_raw", measurement_time));

                        //pose
                        gps_count_++;
                    } else {
                        skipped_bytes += idx2 - idx;
                        idx = idx2 + 1;
                    }
                } else if (idx + 73 <= n && buf[idx] == 0xBD && buf[idx + 1] == 0xDB && buf[idx + 2] == 0x0B) {
                    Imu imu_data;
                    const char *p_imu = buf.data() + idx + int(buf[idx + 3]);
                    int32_t gps_week = reinterpret_cast<const int32_t*>(p_imu + 0)[0]; 
                    double gps_sec = reinterpret_cast<const double*>(p_imu + 4)[0];
                    double measurement_time = GPStoUTC(gps_week, gps_sec);
                    imu_data.mutable_header()->set_timestamp_sec(measurement_time);
                    imu_data.mutable_header()->set_module_name(writer_->ModuleName());
                    imu_data.mutable_header()->set_sequence_num(imu_count_);
                    imu_data.set_measurement_time(measurement_time);
                    imu_data.mutable_linear_acceleration()->set_x(RawImuAccel(reinterpret_cast<const int32_t*>(p_imu + 24)[0]));
                    imu_data.mutable_linear_acceleration()->set_y(-RawImuAccel(reinterpret_cast<const int32_t*>(p_imu + 20)[0]));
                    imu_data.mutable_linear_acceleration()->set_z(RawImuAccel(reinterpret_cast<const int32_t*>(p_imu + 16)[0]));
                    imu_data.mutable_angular_velocity()->set_x(RawImuGyro(reinterpret_cast<const int32_t*>(p_imu + 36)[0]));
                    imu_data.mutable_angular_velocity()->set_y(RawImuGyro(-reinterpret_cast<const int32_t*>(p_imu + 32)[0]));
                    imu_data.mutable_angular_velocity()->set_z(RawImuGyro(reinterpret_cast<const int32_t*>(p_imu + 28)[0]));

                    //if (imu_count_ % 123 == 0) {
                    //    INFO_MSG("imu: " << uint64_t(measurement_time * 1000) << " : " << 
                    //    imu_data.linear_acceleration().x() << " " << imu_data.linear_acceleration().y() << " " << imu_data.linear_acceleration().z() << " : " << 
                    //    imu_data.angular_velocity().x() <<  " " << imu_data.angular_velocity().y() << " " << imu_data.angular_velocity().z());
                    //}
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
                } else if (idx + 62 <= n && buf[idx] == 0xAA && buf[idx + 1] == 0x44 && buf[idx + 2] == 0x12) {
                    const char *p_gnss = buf.data() + idx;
                    idx += 62;
                    uint8_t check_sum = *(p_gnss);
                    bool check_sum_valid = true;
                    for (int i1 = 1; i1 < 62; i1 ++) {
                        check_sum ^= *(p_gnss + idx);
                        if ((i1 == 56 && check_sum != *(p_gnss + 57)) ||
                            (i1 == 61 && check_sum != *(p_gnss + 62))) {
                            check_sum_valid = false;
                        }
                    }
                    if (!check_sum_valid) {
                        skipped_bytes += 62;
                    } else {
                        Gnss gnss_data;
                        
                        int32_t gps_week = reinterpret_cast<const int32_t*>(p_gnss + 58)[0]; 
                        double gps_sec = double(reinterpret_cast<const uint32_t*>(p_gnss + 52)[0]) / 1000;
                        double measurement_time = GPStoUTC(gps_week, gps_sec);

                        gnss_data.mutable_header()->set_timestamp_sec(measurement_time);
                        gnss_data.mutable_header()->set_module_name(writer_->ModuleName());
                        gnss_data.mutable_header()->set_sequence_num(gps_count_);
                        
                        gnss_data.mutable_orientation()->set_x(asensing_data_convert<int16_t>(p_gnss + 7, 360.0 / 32768));
                        gnss_data.mutable_orientation()->set_y(asensing_data_convert<int16_t>(p_gnss + 3, 360.0 / 32768));
                        gnss_data.mutable_orientation()->set_z(asensing_data_convert<int16_t>(p_gnss + 5, 360.0 / 32768));
                        gnss_data.mutable_gyro()->set_x(asensing_data_convert<int16_t>(p_gnss + 9, 300.0 / 32768));
                        gnss_data.mutable_gyro()->set_y(asensing_data_convert<int16_t>(p_gnss + 11, 300.0 / 32768));
                        gnss_data.mutable_gyro()->set_z(asensing_data_convert<int16_t>(p_gnss + 13, 300.0 / 32768));
                        gnss_data.mutable_accel()->set_x(asensing_data_convert<int16_t>(p_gnss + 15, 300.0 / 32768));
                        gnss_data.mutable_accel()->set_y(asensing_data_convert<int16_t>(p_gnss + 17, 300.0 / 32768));
                        gnss_data.mutable_accel()->set_z(asensing_data_convert<int16_t>(p_gnss + 19, 300.0 / 32768));
                        gnss_data.mutable_position()->set_lon(asensing_data_convert<int32_t>(p_gnss + 21, 1e-7));
                        gnss_data.mutable_position()->set_lat(asensing_data_convert<int32_t>(p_gnss + 25, 1e-7));
                        gnss_data.mutable_position()->set_height(asensing_data_convert<int32_t>(p_gnss + 29, 1e-3));
                        gnss_data.mutable_linear_velocity()->set_x(asensing_data_convert<int16_t>(p_gnss + 33, 1e2 / 32768));
                        gnss_data.mutable_linear_velocity()->set_y(asensing_data_convert<int16_t>(p_gnss + 35, 1e2 / 32768));
                        gnss_data.mutable_linear_velocity()->set_z(asensing_data_convert<int16_t>(p_gnss + 37, 1e2 / 32768));
                        gnss_data.set_solution_status(reinterpret_cast<const uint8_t*>(p_gnss + 39)[0]);
                        gnss_data.SerializeToString(&content);
                        CHECK(writer_->PushMessage(content, "gnss", measurement_time));

                        RawData gnss_raw;
                        gnss_raw.mutable_header()->set_timestamp_sec(measurement_time);
                        gnss_raw.mutable_header()->set_module_name(writer_->ModuleName());
                        gnss_raw.mutable_header()->set_sequence_num(gps_count_);
                        gnss_raw.set_data(buf.substr(idx - 62, 62));
                        gnss_raw.SerializeToString(&content);
                        CHECK(writer_->PushMessage(content, "gnss_raw", measurement_time));

                        gps_count_ ++;
                    }

                } else {
                    skipped_bytes ++;
                    idx ++;
                }
            }
            if (read_bytes > 1 * kMBSize) {
                if (skipped_bytes > 0) {
                    WARN_MSG("GNSS Worker Unrecognized Data: " << double(skipped_bytes) / read_bytes * 100 << "%");
                }
                read_bytes = 0;
                skipped_bytes = 0;
            }
        } else {
            usleep(1000);
        }
    }
    return true;
}

