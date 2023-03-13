#pragma once
#include <string>
#include <mutex>
#include <thread>
#include <queue>
#include <atomic>
#include <memory>
#include "syncv4l2.h"
#include <fstream>
#include "NvJpegEncoder.h"


#define CHECK(S) \
    for (bool status = S; status != true;) { \
        ERROR_MSG(__LINE__ << " #" << channel_ << "  check faild"); \
        return false; }


class CameraCollectWorker {
    public:
        CameraCollectWorker(const int &channel, const std::string &str_config);
        ~CameraCollectWorker();
        bool Init();
        bool Release();
        bool Open();
        bool Push(uint64_t mesurement_time,unsigned char *data, int width, int height, int data_len);

        static void Consume(void *op);

        friend void SyncV4l2CallBack(int nChan,struct timespec stTime,int nWidth,int nHeight,unsigned char *pData,int nDatalen,void *pUserData);

    private:
        bool DoConsume();

        SYNCV4L2_TCameraPara camera_params_;
        int channel_;
        int width_;
        int height_;
        int jpeg_quality_;
        bool init_ = false;
        std::shared_ptr<std::thread> consumer_;
        std::atomic<bool> stopped_ {false};
        int buffer_len_{0};
        std::queue<NvBuffer*> free_bufs_;
        std::queue<NvBuffer*> using_bufs_;
        std::queue<uint64_t> mesurement_times_;
        std::mutex buf_mutex_;

        unsigned char *jpeg_buf_{nullptr};
        unsigned long jpeg_buf_size_{0};
        std::shared_ptr<NvJPEGEncoder> jpegenc_;

        int push_count_{0};
        int free_bufs_count_{0};
        uint64_t push_time_{0};

        std::shared_ptr<std::ofstream> file_;
        uint64_t record_time_;
};
