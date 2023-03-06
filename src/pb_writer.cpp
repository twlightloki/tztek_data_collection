#include <fcntl.h>
#include <unistd.h>
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "pb_writer.h"
#include <fstream>

using namespace common;
using namespace std::chrono;

PBWriter::PBWriter(const std::string &module_name, 
        const std::string &output_dir,
        const uint64_t file_size):
    module_name_(module_name), 
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


bool PBWriter::PushMessage(const std::string &content, const std::string &sensor_name, const uint64_t record_time) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (chunk_.get() && current_size_ > file_size_) {
        float size_mb = current_size_ / 1048576;
        INFO_MSG("flush file count = " << message_count_ << 
                " size(mb):" << size_mb << " push speed(mb/s)" << size_mb / (record_time - record_time_) * 1000000000);
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
        new_message->set_sensor_name(sensor_name);
        new_message->set_time(record_time);
        new_message->set_content(content);
        current_size_ += content.size();
        message_count_ ++;
    }
    return true;
}



bool PBWriter::Consume() {
    time_point<system_clock> last = system_clock::now();
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
            std::string path = output_dir_ + "/" + module_name_ + "_" + std::to_string(record_time); 
            std::ofstream ouf(path, std::ios::binary);
            time_point<system_clock> start = system_clock::now();
            chunk->SerializeToOstream(&ouf);
            time_point<system_clock> end = system_clock::now();
            duration<float> saving_elapsed = end - start;
            duration<float> last_chunk_elapsed = end - last;
            last = end;
            float disk_speed = (float)file_size_ / 1048576 / saving_elapsed.count();
            float disk_flow = (float)file_size_ / 1048576 / last_chunk_elapsed.count();
            ouf.close();
            INFO_MSG("flush file estimate disk speed(mb/s): " << disk_speed << ", flow: " << disk_flow);
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
