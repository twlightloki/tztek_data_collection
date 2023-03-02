#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <assert.h>
#include "cfg.h"
#include <iostream>
#include "camera_collect_worker.h"
#include "NvLogging.h"
#include "sensor_image.pb.h"
#include "kernels.h"
#include <fstream>
#include "cuda_runtime.h"

using namespace std::chrono;
using namespace common;
using namespace drivers;

void SyncV4l2CallBack(int nChan,struct timespec stTime,int nWidth,int nHeight,unsigned char *pData,int nDatalen,void *pUserData) {
    CameraCollectWorker *worker= reinterpret_cast<CameraCollectWorker *>(pUserData);
    if(worker == NULL)
    {
        return;
    }
    worker->Push(stTime.tv_sec * 1000000000 + stTime.tv_nsec, pData, nWidth, nHeight, nDatalen);
}

CameraCollectWorker::CameraCollectWorker(const int channel, const std::string &str_config,
        const std::shared_ptr<PBWriter>& writer):
    channel_(channel), writer_(writer) {
        memset(&camera_params_, 0 ,sizeof(SYNCV4L2_TCameraPara));
        std::string tag = "video" + std::to_string(channel_);

        width_ = CFG_get_section_value_int(str_config.c_str(),tag.c_str(),"width",0) ;
        height_ = CFG_get_section_value_int(str_config.c_str(),tag.c_str(),"height",0) ;
        camera_params_.nWidth = width_;
        camera_params_.nHeight = height_;
        snprintf(camera_params_.szDevName,sizeof(camera_params_.szDevName),"/dev/video%d", channel);

        camera_params_.nTriggerFps = CFG_get_section_value_int(str_config.c_str(),tag.c_str(),"fps",0);
        buffer_len_ = camera_params_.nTriggerFps * interval_;
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
        unsigned char* yuyv_buf;
        cudaMallocHost((void**)&yuyv_buf, width_ * height_ * 2);
        free_bufs_.push(yuyv_buf);
    }
    init_ = true;
    INFO_MSG("WORKER" << channel_ << " Init completed");
    last_ = system_clock::now();
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
        cudaFree(free_bufs_.front());
        free_bufs_.pop();
    }
    INFO_MSG("WORKER" << channel_ << " Released, image_count " << image_count_);
    return true;
}

bool CameraCollectWorker::Open() {
    INFO_MSG("WORKER" << channel_ << " Open");
    CHECK(init_);
    consumer_.reset(new std::thread([this](){this->Consume();}));
    SYNCV4L2_OpenCamera(channel_, &camera_params_);
    SYNCV4L2_SetDataCallback(channel_, SyncV4l2CallBack, this);
    return true;
}


bool CameraCollectWorker::Push(uint64_t measurement_time, unsigned char *data, int width, int height, int data_len) {
    CHECK(init_);
    CHECK(width == width_ && height == height_ && data_len == width * height * 2 && measurement_time > 0);
    CHECK(data != nullptr);

    {
        //        INFO_MSG("WORKER" << channel_ << " Push " << measurement_time);
        time_point<system_clock> start = system_clock::now();
        std::lock_guard<std::mutex> lock(buf_mutex_);
        CHECK(free_bufs_.size() > 0);
        unsigned char* yuyv_buf = free_bufs_.front();
        memcpy(yuyv_buf, data, width_ * height_ * 2);
        //        libyuv::YUY2ToI420(data, 2 * width_,
        //                buf->planes[0].data, width_,
        //                buf->planes[1].data, width_ / 2,
        //                buf->planes[2].data, width_ / 2,
        //                width_,
        //                height_);
        free_bufs_.pop();	
        free_bufs_count_ += free_bufs_.size();
        using_bufs_.emplace(yuyv_buf);
        measurement_times_.emplace(measurement_time);
        time_point<system_clock> end = system_clock::now();
        duration<float> elapsed = end - start;
        push_time_ += elapsed.count() * 1000;
        push_count_ += 1;
        if (push_count_ % buffer_len_ == 0) {
            duration<float> clip = end - last_;
            double fps = double(buffer_len_) / clip.count();
            if (fps < camera_params_.nTriggerFps * 0.95) {
                WARN_MSG("WORKER#" << channel_ << ": fps: " << fps << 
                        " avg push time: " << push_time_ / push_count_ << "ms, avg free bufs: " << free_bufs_count_ / push_count_);
            }
            push_count_ = 0;
            push_time_ = 0;
            free_bufs_count_ = 0;
            last_ = end;
        }
        //        INFO_MSG("WORKER" << channel_ << " Push " << measurement_time << " end ");
    }

    return true;
}

bool CameraCollectWorker::Consume() {
    INFO_MSG("WORKER" << channel_ << " Consumer start");
    CHECK(init_);
    std::unique_ptr<NvJPEGEncoder> jpegenc;
    std::unique_ptr<NvJPEGEncoder> scaled_jpegenc;

    std::string enc_name = "jpegenc#" + std::to_string(channel_);
    jpegenc.reset(NvJPEGEncoder::createJPEGEncoder(enc_name.c_str()));
    if (writer_->AvailVisual()) {
        std::string scaled_enc_name = enc_name + "_scaled";
        scaled_jpegenc.reset(NvJPEGEncoder::createJPEGEncoder(scaled_enc_name.c_str()));
 //       scaled_jpegenc->setScaledEncodeParams(writer_->VisualWidth(), writer_->VisualHeight());
    }
    size_t jpeg_buf_size = 10 * kMBSize;
    unsigned char *jpeg_buf = new unsigned char[jpeg_buf_size];


    int buf_used_count = 0;
    int consume_count = 0;
    double encode_ratio = 0;
    uint64_t consume_time = 0;
    NvBuffer buf(V4L2_PIX_FMT_YUV420M, width_, height_, 0);
    buf.allocateMemory();
 
    NvBuffer scaled_buf(V4L2_PIX_FMT_YUV420M, writer_->VisualWidth(), writer_->VisualHeight(), 0);
    scaled_buf.allocateMemory();
    
    cudaStream_t stream; 
    cudaStreamCreate(&stream);

    while (true) {
        unsigned char *yuyv_buf = nullptr;
        uint64_t measurement_time = 0;
        {
            std::lock_guard<std::mutex> lock(buf_mutex_);
            if (using_bufs_.size() > 0) {
                buf_used_count += using_bufs_.size();
                yuyv_buf = using_bufs_.front();
                measurement_time = measurement_times_.front();
                using_bufs_.pop();
                measurement_times_.pop();
            }
        }
        if (measurement_time > 0) {
            time_point<system_clock> start = system_clock::now();
            if (writer_->AvailDump()) {
                YUYV2To420WithYCExtend(yuyv_buf, buf.planes[0].data, buf.planes[1].data, buf.planes[2].data, 
                                       width_, height_, width_, height_, stream);
            }

            if (writer_->AvailVisual()) {
                YUYV2To420WithYCExtend(yuyv_buf, scaled_buf.planes[0].data, scaled_buf.planes[1].data, scaled_buf.planes[2].data, 
                                       writer_->VisualWidth(), writer_->VisualHeight(), width_, height_, stream);
            }
 
            cudaStreamSynchronize(stream);
            {
                std::lock_guard<std::mutex> lock(buf_mutex_);
                free_bufs_.push(yuyv_buf);
            }

            double measurement_time_sec = (double)measurement_time / 1e9;

            auto EncodeToContent = [&](NvBuffer& buf,
                                       const std::unique_ptr<NvJPEGEncoder> &encoder, std::string& content,
                                       const int quality) {
                uint64_t jpeg_size = jpeg_buf_size;
                int ret = encoder->encodeFromBuffer(buf, JCS_YCbCr, &jpeg_buf, jpeg_size, quality);
                CHECK(ret == 0);
                if (jpeg_size > jpeg_buf_size) {
                    jpeg_buf_size = jpeg_size;
                }
                CompressedImage image;
                image.mutable_header()->set_timestamp_sec(measurement_time_sec);
                image.mutable_header()->set_module_name(writer_->ModuleName());
                image.mutable_header()->set_sequence_num(image_count_);
                image.mutable_header()->set_camera_timestamp(measurement_time);

                image.set_frame_id(camera_params_.szDevName);
                image.set_format("jpeg");
                image.set_data((void*)jpeg_buf, jpeg_size); 
                image.set_measurement_time(measurement_time_sec);
                image.SerializeToString(&content);
                encode_ratio += (double)jpeg_size / (width_ * height_ * 2) / 2;
                return true;
            };

            if (writer_->AvailDump()) {
                std::string content;
                CHECK(EncodeToContent(buf, jpegenc, content, jpeg_quality_));
                CHECK(writer_->PushMessage(content, "camera", measurement_time_sec, PBWriter::ONLY_LOCAL));
            }
            if (writer_->AvailVisual()) {
                std::string scaled_content;
                CHECK(EncodeToContent(scaled_buf, scaled_jpegenc, scaled_content, writer_->VisualQuality()));
                CHECK(writer_->PushMessage(scaled_content, "camera_visual", measurement_time_sec, PBWriter::ONLY_VISUAL));
            }
            image_count_ ++;

#if (0)
            std::string jpeg_name = std::to_string(channel_) + "_" + std::to_string(measurement_time / 1000000) + ".jpeg";
            std::ofstream ouf(jpeg_name, std::ios::out | std::ios::binary);
            ouf.write(reinterpret_cast<const char*>(jpeg_buf), jpeg_size);
            ouf.close();
#else
#endif
            time_point<system_clock> end = system_clock::now();
            duration<float> elapsed = end - start;
            consume_time += elapsed.count() * 1000;
            consume_count += 1;
            double avg_consume_time = consume_time / consume_count;
            if (consume_count % buffer_len_ == 0) {
                if (1000.0 / avg_consume_time < camera_params_.nTriggerFps) {
                    WARN_MSG(measurement_time / 1000000 <<" WORKER#" <<  channel_ << " avg consume time: " << consume_time / consume_count << "ms, buf_used: " << buf_used_count / consume_count << 
                            " encode_ratio = " << encode_ratio / consume_count);
                }
                consume_count = 0;
                consume_time = 0;
                buf_used_count = 0;
                encode_ratio = 0;
            }


        } else {
            usleep(1000);
        }
        {
            std::lock_guard<std::mutex> lock(buf_mutex_);
            if (stopped_ && using_bufs_.size() == 0) {
                break;
            }
        }
    }
    delete jpeg_buf;
    return true;
}

