#include <fcntl.h>
#include <unistd.h>
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "pb_writer.h"

using namespace common;
using namespace std::chrono;

PBWriter::PBWriter(const std::string &module_name, const std::string &sensor_name, 
        const std::string &output_dir,
        const uint64_t file_size):
    module_name_(module_name), sensor_name_(sensor_name),
    output_dir_(output_dir), file_size_(file_size) {
    };
PBWriter::~PBWriter() {};

bool PBWriter::Open() {
    consumer_.reset(new std::thread([this](){Consume();}));
    return true;
};

bool PBWriter::Close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (chunk_->messages_size() > 0) {
            chunks_.push(chunk_);
            record_times_.push(record_time_);
        }
    }
    stopped_ = true;
    if (consumer_.get()) {
        consumer_->join();
    }
    INFO_MSG("close consumer");
    return true;
};


bool PBWriter::PushMessage(const std::string &content, const uint64_t record_time) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (chunk_.get() && current_size_ > file_size_) {
        INFO_MSG("flush file " << sensor_name_ << " count = " << message_count_ << 
                " size = "  <<current_size_);
        chunks_.push(chunk_);
        record_times_.push(record_time_);
        chunk_.reset();
    }
    if (!chunk_.get()) {
        chunk_.reset(new Chunk());
        current_size_ = 0;
        record_time_ = record_time;
    }

    {
        SingleMessage *new_message = chunk_->add_messages();
        new_message->set_sensor_name(sensor_name_);
        new_message->set_time(record_time);
        new_message->set_content(content);
        current_size_ += content.size();
        message_count_ ++;
    }
    return true;
}



bool PBWriter::Consume() {
    while (true) {
        std::shared_ptr<Chunk> chunk;
        uint64_t record_time = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (chunks_.size() > 0) {
                chunk = chunks_.front();
                record_time = record_times_.front();
                chunks_.pop();
                record_times_.pop();
                INFO_MSG(" chunk queue size: " << chunks_.size());
            }
        }
        if (record_time > 0) {
            int fd;
            std::string path = output_dir_ + "/" + module_name_ + "_" + sensor_name_ + "_" + std::to_string(record_time); 
            fd = open(path.data(), O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
            if (fd < 0) {
                ERROR_MSG(path + " file open faild");
                return false;
            }
            time_point<system_clock> start = system_clock::now();
            google::protobuf::io::FileOutputStream raw_output(fd);
            chunk->SerializeToZeroCopyStream(&raw_output);
            time_point<system_clock> end = system_clock::now();
            duration<float> elapsed = end - start;
            float disk_speed = (float)chunk->messages_size() / 1048576 / (elapsed.count() / 1000000);
            if (close(fd) < 0) {
                ERROR_MSG(path + " file close faild");
                return false;
            }
            INFO_MSG("flush file " << sensor_name_ << " finish size(b): " << chunk->messages_size() << " disk speed(mb/s): " << disk_speed);
        } else {
            usleep(1000);
        }


        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_ && chunks_.size() == 0) {
                break;
            }
        }
    }
    return true;
};
