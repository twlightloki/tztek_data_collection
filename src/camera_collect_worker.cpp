#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <assert.h>
#include "cfg.h"
#include <iostream>
#include "camera_collect_worker.h"
#include "libyuv.h"
#include "NvLogging.h"
#include <chrono>

using namespace std::chrono;

void SyncV4l2CallBack(int nChan,struct timespec stTime,int nWidth,int nHeight,unsigned char *pData,int nDatalen,void *pUserData) {
    CameraCollectWorker *worker= reinterpret_cast<CameraCollectWorker *>(pUserData);
    if(worker == NULL)
    {
        return;
    }
    worker->Push(stTime.tv_sec * 1000000000 + stTime.tv_nsec, pData, nWidth, nHeight, nDatalen);
}

CameraCollectWorker::CameraCollectWorker(const int &channel, const std::string &str_config):
    channel_(channel) {
        memset(&camera_params_, 0 ,sizeof(SYNCV4L2_TCameraPara));
        std::string tag = "video" + std::to_string(channel_);

        width_ = CFG_get_section_value_int(str_config.c_str(),tag.c_str(),"width",0) ;
        height_ = CFG_get_section_value_int(str_config.c_str(),tag.c_str(),"height",0) ;
        camera_params_.nWidth = width_;
        camera_params_.nHeight = height_;
        snprintf(camera_params_.szDevName,sizeof(camera_params_.szDevName),"/dev/video%d", channel);

        camera_params_.nTriggerFps = CFG_get_section_value_int(str_config.c_str(),tag.c_str(),"fps",0);
        buffer_len_ = camera_params_.nTriggerFps;
        camera_params_.nTriggerPluseWidth = CFG_get_section_value_int(str_config.c_str(),tag.c_str(),"plusewidth",3);
        camera_params_.nTriggerPluseOffset = CFG_get_section_value_int(str_config.c_str(),tag.c_str(),"offset",0);
        camera_params_.nRender = 0;
        camera_params_.nTriggerOnly = 0;
        jpeg_quality_ = CFG_get_section_value_int(str_config.c_str(),"common", "quality",95);
    }

CameraCollectWorker::~CameraCollectWorker() {
}

bool CameraCollectWorker::Init() {
    for (int i1 = 0; i1 < buffer_len_; i1 ++) {
        NvBuffer *buf = new NvBuffer(V4L2_PIX_FMT_YUV420M, width_, height_, i1);
        buf->allocateMemory();
        free_bufs_.push(buf);
    }
    std::string enc_name = "jpegenc#" + std::to_string(channel_);
    jpegenc_.reset(NvJPEGEncoder::createJPEGEncoder(enc_name.c_str()));

    jpeg_buf_size_ = 10485760; 
    jpeg_buf_ = new unsigned char[jpeg_buf_size_];

    init_ = true;
    INFO_MSG("WORKER" << channel_ << " Init completed");
    return true;
}

bool CameraCollectWorker::Release() {
    CHECK(init_);

    SYNCV4L2_CloseCamera(channel_);
    if (consumer_.get()) {
        stopped_ = true;
        consumer_->join();
    }
    while (free_bufs_.size() > 0) {
        delete free_bufs_.front();
        free_bufs_.pop();
    }
    if (jpeg_buf_) {
        delete jpeg_buf_;
    }
    INFO_MSG("WORKER" << channel_ << " Released");
    return true;
}

bool CameraCollectWorker::Open() {
    INFO_MSG("WORKER" << channel_ << " Open");
    CHECK(init_);
    consumer_.reset(new std::thread(Consume, this));
    SYNCV4L2_OpenCamera(channel_, &camera_params_);
    SYNCV4L2_SetDataCallback(channel_, SyncV4l2CallBack, this);
    return true;
}


bool CameraCollectWorker::Push(uint64_t mesurement_time,unsigned char *data, int width, int height, int data_len) {
    CHECK(init_);
    CHECK(width == width_ && height == height_ && data_len == width * height * 2 && mesurement_time > 0);
    CHECK(data != nullptr);

    {
        //INFO_MSG("WORKER" << channel_ << " Push " << mesurement_time);
        time_point<system_clock> start = system_clock::now();
        std::lock_guard<std::mutex> lock(buf_mutex_);
        CHECK(free_bufs_.size() > 0);
        NvBuffer* buf = free_bufs_.front();
        libyuv::YUY2ToI420(data, 2 * width_,
                buf->planes[0].data, width_,
                buf->planes[1].data, width_ / 2,
                buf->planes[2].data, width_ / 2,
                width_,
                height_);
        free_bufs_.pop();	
        free_bufs_count_ += free_bufs_.size();
        using_bufs_.emplace(buf);
        mesurement_times_.emplace(mesurement_time);
        time_point<system_clock> end = system_clock::now();
        duration<float> elapsed = end - start;
        push_time_ += elapsed.count() * 1000;
        push_count_ += 1;
        if (push_count_ % 25 == 0) {
            INFO_MSG("WORKER#" << channel_ << " avg push time: " << push_time_ / push_count_ << "ms, avg free bufs: " << free_bufs_count_ / push_count_);
            push_count_ = 0;
            push_time_ = 0;
            free_bufs_count_ = 0;
        }
        //INFO_MSG("WORKER" << channel_ << " Push " << mesurement_time << " end");
    }

    return true;
}

void CameraCollectWorker::Consume(void* op) {
    CameraCollectWorker *worker= reinterpret_cast<CameraCollectWorker *>(op);
    worker->DoConsume();
}

bool CameraCollectWorker::DoConsume() {
    INFO_MSG("WORKER" << channel_ << " Consumer start");
    CHECK(init_);
    int buf_used_count = 0;
    int consume_count = 0;
    uint64_t consume_time = 0;
    while (!stopped_) {
        NvBuffer *buf = nullptr;
        uint64_t mesurement_time = 0;
        {
            std::lock_guard<std::mutex> lock(buf_mutex_);
            if (using_bufs_.size() > 0) {
                buf_used_count += using_bufs_.size();
                buf = using_bufs_.front();
                mesurement_time = mesurement_times_.front();
                using_bufs_.pop();
                mesurement_times_.pop();
            }
        }
        if (mesurement_time > 0) {
            time_point<system_clock> start = system_clock::now();
            //INFO_MSG("WORKER" << channel_ << " process frame from " << mesurement_time);

            unsigned long jpeg_size = jpeg_buf_size_;
            int ret = jpegenc_->encodeFromBuffer(*buf, JCS_YCbCr, &jpeg_buf_, jpeg_size, jpeg_quality_);
            CHECK(ret == 0);
            if (jpeg_size > jpeg_buf_size_) {
                jpeg_buf_size_ = jpeg_size;
            }
            time_point<system_clock> end = system_clock::now();
            duration<float> elapsed = end - start;
            consume_time += elapsed.count() * 1000;
            consume_count += 1;
            if (consume_count % 25 == 0) {
                INFO_MSG("WORKER#" << channel_ << " avg consume time: " << consume_time / consume_count << "ms, buf_used: " << buf_used_count / consume_count);
                consume_count = 0;
                consume_time = 0;
                buf_used_count = 0;
            }

#if (1)
//            std::string jpeg_name = std::to_string(channel_) + "_" + std::to_string(mesurement_time / 1000000) + ".jpeg";
//            std::ofstream ouf(jpeg_name, std::ios::out | std::ios::binary);
//            ouf.write(reinterpret_cast<const char*>(jpeg_buf_), jpeg_size);
//            ouf.close();
#else
#endif
            {
                std::lock_guard<std::mutex> lock(buf_mutex_);
                free_bufs_.push(buf);
            }
        } else {
            usleep(1000);
        }
    }
    return true;
}

