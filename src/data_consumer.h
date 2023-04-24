#pragma once
#include "common.h"
#include "message.pb.h"
#include "zmq.hpp"

class NetworkController {
};

class DataWriter {
    public:
        DataWriter(const std::string &module_name);
        ~DataWriter();


        bool OpenDump(const std::string &output_dir = "./",  const uint64_t file_size =  1024 * kMBSize);
        bool CloseDump();

        bool OpenVisualize(const std::string &visual_port);
        bool CloseVisualize();


        enum PushType {
            ONLY_LOCAL,
            BOTH,
            ONLY_VISUAL};

        bool PushMessage(const std::string &content, const std::string &sensor_name, const double record_time_sec,
                         const PushType push_type = BOTH);
        std::string& ModuleName() {return module_name_;};
        bool AvailVisual() const { return visualize_opened_; };
        bool AvailDump() const { return dump_opened_; };
        //XXX adhoc
        int VisualWidth() const {return 960; };
        int VisualQuality() const {return 80; };
        int VisualStep() const {return 2; };
    private:
        bool DumpConsume();
        std::string MessageCount(double elapsed_time_sec) const;

        std::queue<std::shared_ptr<common::Chunk>> chunks_;
        std::queue<uint64_t> record_times_;
        std::shared_ptr<common::Chunk> chunk_;
        std::unique_ptr<std::thread> consumer_;
        std::mutex dump_mutex_;
        std::mutex visual_mutex_;
        std::shared_mutex consumer_state_mutex_;
        std::string module_name_;
        std::string output_dir_;
        uint64_t file_size_{0};


        uint64_t current_size_{0};
        uint64_t record_time_{0};
        std::map<std::string, uint64_t> message_count_;
        std::map<std::string, uint64_t> flow_count_;

        std::shared_ptr<zmq::context_t> zmq_context_;
        std::shared_ptr<zmq::socket_t> zmq_socket_;
        std::atomic<bool> dump_opened_{false};
        std::atomic<bool> visualize_opened_{false};
};
