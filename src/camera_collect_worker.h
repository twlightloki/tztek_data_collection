#pragma once
#include "syncv4l2.h"
#include "NvJpegEncoder.h"
#include "data_consumer.h"
#include "common.h"
#include <chrono>



class CameraCollectWorker {
    public:
        CameraCollectWorker(const int channel, const std::string &str_config, 
                            const std::shared_ptr<DataWriter> &writer);
        ~CameraCollectWorker();
        bool Init();
        bool Release();
        bool Open();
        bool Push(uint64_t measurement_time,unsigned char *data, int width, int height, int data_len);

        friend void SyncV4l2CallBack(int nChan,struct timespec stTime,int nWidth,int nHeight,unsigned char *pData,int nDatalen,void *pUserData);

    private:
        bool Consume();

        SYNCV4L2_TCameraPara camera_params_;
        int channel_;
        int width_;
        int height_;
        int jpeg_quality_;
        int interval_{5};
        bool init_ = false;
        std::unique_ptr<std::thread> consumer_;
        std::atomic<bool> stopped_ {false};
        int buffer_len_{0};
        std::queue<uint64_t> measurement_times_;
        std::queue<unsigned char*> using_bufs_;
        std::queue<unsigned char*> free_bufs_;
        std::mutex buf_mutex_;
        int image_count_{0};
        std::chrono::time_point<std::chrono::system_clock> last_;

        int push_count_{0};
        int free_bufs_count_{0};
        uint64_t push_time_{0};

        std::shared_ptr<DataWriter> writer_;
};

