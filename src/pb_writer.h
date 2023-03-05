#pragma once
#include "common.h"
#include "message.pb.h"

class PBWriter {
    public:
        PBWriter(const std::string &module_name, 
                const std::string &output_dir = "./", 
                const uint64_t file_size =  1024 * kMBSize);
        ~PBWriter();


        bool Open();
        bool Close();
        bool PushMessage(const std::string &content, const std::string &sensor_name, const double record_time_sec);
        std::string& ModuleName() {return module_name_;};
    private:
        bool Consume();
        std::string MessageCount(double elapsed_time_sec);

        std::queue<std::shared_ptr<common::Chunk>> chunks_;
        std::queue<uint64_t> record_times_;
        std::shared_ptr<common::Chunk> chunk_;
        std::unique_ptr<std::thread> consumer_;
        std::mutex mutex_;
        std::string module_name_;
        std::string output_dir_;
        uint64_t file_size_;
        std::atomic<bool> stopped_{false};


        uint64_t current_size_{0};
        uint64_t record_time_{0};
        std::map<std::string, uint64_t> message_count_;
};
