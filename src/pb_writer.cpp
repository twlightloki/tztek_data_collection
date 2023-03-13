#include <fcntl.h>
#include <unistd.h>
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "pb_writer.h"

using namespace common;

PBWriter::PBWriter(const std::string &module_name, const std::string &sensor_name, 
        const std::string &output_dir,
        const uint64_t file_size):
    module_name_(module_name), sensor_name_(sensor_name),
    output_dir_(output_dir), file_size_(file_size) {
    };
PBWriter::~PBWriter() {};

bool PBWriter::Close() {
    if (writer_.get()) {
        writer_->join();
    }
    if (chunk_.get()) {
        CHECK(WriteFile(std::move(chunk_), record_time_));
    }
    INFO_MSG("closing writer " + module_name_ + "_" + sensor_name_);
    return true;
};


bool PBWriter::PushMessage(const std::string &content, const uint64_t record_time) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (chunk_.get() && current_size_ > file_size_) {
        if (writer_.get()) {
            writer_->join();
        }
        INFO_MSG("flush file " + sensor_name_ + " count = " + std::to_string(message_count_) + 
                " size = " + std::to_string(current_size_));
        std::cout << "0 " << chunk_.get() << std::endl;
        writer_.reset(new std::thread([this](){this->WriteFile(std::move(chunk_), record_time_);}));
        std::cout << "2 " << chunk_.get() << std::endl;
    }
    if (!chunk_.get()) {
        chunk_.reset(new Chunk());
        current_size_ = 0;
        record_time_ = record_time;
    }

    SingleMessage *new_message = chunk_->add_messages();
    new_message->set_sensor_name(sensor_name_);
    new_message->set_time(record_time);
    new_message->set_content(content);
    current_size_ += content.size();
    message_count_ ++;
    return true;
}



bool PBWriter::WriteFile(const std::unique_ptr<Chunk> chunk, const uint64_t record_time) {
    std::cout << "1 " <<   chunk.get() << std::endl;
    int fd;
    std::string path = output_dir_ + "/" + module_name_ + "_" + sensor_name_ + "_" + std::to_string(record_time); 
    fd = open(path.data(), O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (fd < 0) {
        ERROR_MSG(path + " file open faild");
        return false;
    }
    google::protobuf::io::FileOutputStream raw_output(fd);
    chunk->SerializeToZeroCopyStream(&raw_output);
    if (close(fd) < 0) {
        ERROR_MSG(path + " file close faild");
        return false;
    }
    INFO_MSG("flush file " + sensor_name_ + " finish size: " + std::to_string(chunk->messages_size()));

    return true;
};
