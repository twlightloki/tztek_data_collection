#pragma once
#include "common.h"
#include "message.pb.h"

class PBWriter {
    public:
        PBWriter(const std::string &module_name, const std::string &sensor_name, 
                const std::string &output_dir = "./", 
                const uint64_t file_size =  1024 * 1024 * 1024);
        ~PBWriter();


        bool Close();
        bool PushMessage(const std::string &content, const uint64_t record_time);
    private:
        bool WriteFile(const std::unique_ptr<common::Chunk> chunk, const uint64_t record_time);

        std::unique_ptr<common::Chunk> chunk_;
        std::unique_ptr<std::thread> writer_;
        std::mutex mutex_;
        std::string module_name_;
        std::string sensor_name_;
        std::string output_dir_;
        uint64_t file_size_;


        uint64_t current_size_{0};
        uint64_t record_time_{0};
        uint64_t message_count_{0};
};
