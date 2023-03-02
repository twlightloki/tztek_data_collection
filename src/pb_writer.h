#pragma once
#include "common.h"
#include "message.pb.h"
#include "zmq.hpp"

class PBWriter {
    public:
        PBWriter(const std::string &module_name, 
                 const std::string &output_dir = "./", 
                 const uint64_t file_size =  1024 * kMBSize,
                 const std::string &visual_addr = "");
        ~PBWriter();


        bool Open();
        bool Close();

        enum PushType {
            ONLY_LOCAL,
            BOTH,
            ONLY_VISUAL};

        bool PushMessage(const std::string &content, const std::string &sensor_name, const double record_time_sec,
                         const PushType push_type = BOTH);
        std::string& ModuleName() {return module_name_;};
        bool AvailVisual() const { return zmq_socket_.get() != nullptr; };
        bool AvailDump() const { return file_size_ > 0; };
        //XXX adhoc
        int VisualHeight() const { return 768; };
        int VisualWidth() const {return 1024; };
        int VisualQuality() const {return 80; };
    private:
        bool Consume();
        std::string MessageCount(double elapsed_time_sec) const;

        std::queue<std::shared_ptr<common::Chunk>> chunks_;
        std::queue<uint64_t> record_times_;
        std::shared_ptr<common::Chunk> chunk_;
        std::unique_ptr<std::thread> consumer_;
        std::mutex mutex_;
        std::string module_name_;
        std::string output_dir_;
        uint64_t file_size_;
        std::string visual_port_;
        std::atomic<bool> stopped_{false};


        uint64_t current_size_{0};
        uint64_t record_time_{0};
        std::map<std::string, uint64_t> message_count_;
        std::map<std::string, uint64_t> flow_count_;

        std::shared_ptr<zmq::context_t> zmq_context_;
        std::shared_ptr<zmq::socket_t> zmq_socket_;
};
