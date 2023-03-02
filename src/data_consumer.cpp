#include <fcntl.h>
#include <unistd.h>
#include "google/protobuf/text_format.h"
#include "data_consumer.h"
#include <fstream>

using namespace common;
using namespace std::chrono;
NetworkController::NetworkController(const std::string &port, 
        const std::shared_ptr<DataWriter> &writer):
    port_(port), writer_(writer) {
    };

void NetworkController::Spin() {
    zmq::context_t context(1);
    INFO_MSG("listening port: " << port_);

    while (true) {
       try {
            zmq::message_t request;
            zmq::recv_result_t rtn;
            zmq::socket_t socket(context, zmq::socket_type::rep);
            socket.bind(std::string("tcp://*:") + port_);
            rtn = socket.recv(request);
            auto reply = [&](const std::string &msg) {
                zmq::const_buffer buf(msg.data(), msg.size());
                socket.send(buf);
            };
            if (*rtn > 0) {
                std::string msg = request.to_string();
                INFO_MSG("RECV MSG:"  << msg);
                bool writer_rtn = false;
                if (msg.substr(0, 4) == "exit") {
                    reply("exit succ");
                    break;
                } else if (msg.substr(0, 10) == "start dump") {
                    writer_rtn = writer_->OpenDump(msg.substr(11, msg.size() - 11));
                } else if (msg.substr(0, 9) == "stop dump") {
                    writer_rtn = writer_->CloseDump();
                } else if (msg.substr(0, 12) == "start visual") {
                    writer_rtn = writer_->OpenVisualize();
                } else if (msg.substr(0, 11) == "stop visual") {
                    writer_rtn = writer_->CloseVisualize();
                };
                if (writer_rtn) {
                    reply("exec succ");
                } else {
                    reply("exec fail");
                }
            }
        } catch (const std::exception &e){
//            ERROR_MSG("zmq listen fail: " << e.what());
        }
        usleep(100000);
    }
}


DataWriter::DataWriter(const std::string &module_name, const uint64_t file_size, const std::string &visual_port):
    module_name_(module_name), file_size_(file_size), visual_port_(visual_port) {
    };
DataWriter::~DataWriter() {};

bool DataWriter::OpenVisualize() {
    std::lock_guard<std::shared_mutex> _(consumer_state_mutex_);
    CHECK(!visualize_opened_);
    INFO_MSG("start visual");
    if (!zmq_context_.get()) {
        try {
            INFO_MSG("visual port: " << visual_port_);
            //more io threads?
            zmq_context_.reset(new zmq::context_t(1));
            zmq_socket_.reset(new zmq::socket_t(*zmq_context_, zmq::socket_type::pub));
            zmq_socket_->bind("tcp://*:" + visual_port_);
        } catch (const std::exception &e){
            ERROR_MSG("zmq init fail: " << e.what());
            return false;
        }
    }
    visualize_opened_ = true;
    return true;
};

bool DataWriter::OpenDump(const std::string &output_dir) {
    std::lock_guard<std::shared_mutex> _(consumer_state_mutex_);
    CHECK(!dump_opened_);
    output_dir_ = output_dir;
    INFO_MSG("dump output dir: " << output_dir);
    chunk_.reset();
    consumer_.reset(new std::thread([this](){DumpConsume();}));
    dump_opened_ = true;
    return true;
};


bool DataWriter::CloseVisualize() {
    std::lock_guard<std::shared_mutex> _(consumer_state_mutex_);
    CHECK(visualize_opened_);
    INFO_MSG("close visual");
    visualize_opened_ = false;
    return true;
};

bool DataWriter::CloseDump() {
    std::lock_guard<std::shared_mutex> _(consumer_state_mutex_);
    CHECK(dump_opened_);
    INFO_MSG("close dump");
    dump_opened_ = false;
    {
        std::lock_guard<std::mutex> lock(dump_mutex_);
        if (chunk_.get() && chunk_->messages_size() > 0) {
            chunks_.push(chunk_);
            record_times_.push(record_time_);
        }
    }
    if (consumer_.get()) {
        consumer_->join();
    }
    return true;
};


std::string DataWriter::MessageCount(double elapsed_time_sec) const {
    std::ostringstream buf;
    for (const auto& entry : message_count_) {
        double count_per_second = double(entry.second) / elapsed_time_sec;
        double flow_per_second = flow_count_.at(entry.first) / elapsed_time_sec / kMBSize;
        buf << std::setiosflags(std::ios::fixed) << std::setiosflags(std::ios::right) << std::setprecision(3) << "[" << entry.first << "]: " 
            << count_per_second << "/s " << flow_per_second << "mb/s" << std::endl;
    }
    return buf.str();
}

std::string DateStr(const double time_sec) {
    auto m_time = std::chrono::milliseconds(int64_t(time_sec * 1e3));
    auto tp = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>(m_time);
    auto tt = std::chrono::system_clock::to_time_t(tp);
    std::string rtn = std::asctime(std::localtime(&tt));
    return rtn.substr(0, rtn.size() - 1);
}

bool DataWriter::PushMessage(const std::string &content, const std::string &sensor_name, const double record_time_sec,
        const PushType push_type) {
    std::shared_lock<std::shared_mutex> _(consumer_state_mutex_);
    bool need_dump = push_type != ONLY_VISUAL && AvailDump();
    bool need_visual = push_type != ONLY_LOCAL && AvailVisual();
    if (need_dump || need_visual) {
        std::string message_content;
        SingleMessage *new_message = nullptr;
        if (need_dump) {
            std::lock_guard<std::mutex> lock(dump_mutex_);
            if (chunk_.get() && current_size_ > file_size_) {
                float size_mb = current_size_ / kMBSize;
                double elapsed_time_sec = record_time_sec - record_time_;
                INFO_MSG(std::endl << "flush_data from  [" << DateStr(record_time_) << "] -> [" << DateStr(record_time_sec) <<"], details:"  << 
                        std::endl << MessageCount(elapsed_time_sec) << 
                        "avg push data flow(mb/s):     " << size_mb / elapsed_time_sec << std::endl);
                chunks_.push(chunk_);
                record_times_.push(record_time_);
                chunk_.reset();
                message_count_.clear();
                flow_count_.clear();
                current_size_ = 0;
            }
            if (!chunk_.get()) {
                chunk_.reset(new Chunk());
                record_time_ = record_time_sec;
                INFO_MSG("new file start from: " << sensor_name << " " << record_time_sec);
            }
            new_message = chunk_->add_messages();
            current_size_ += content.size();
            message_count_[sensor_name] ++;
            flow_count_[sensor_name] += content.size();
        } else {
            new_message = new SingleMessage();
        }

        new_message->set_sensor_name(sensor_name);
        new_message->set_time(record_time_sec * 1e3);
        new_message->set_content(content);

        if (need_visual) {
            new_message->SerializeToString(&message_content);
            message_content = "byd66 " + message_content;
            zmq::const_buffer buf(message_content.data(), message_content.size());
            try {
                std::lock_guard<std::mutex> lock(visual_mutex_);
                zmq_socket_->send(buf);
            } catch (const std::exception &e){
                ERROR_MSG("zmq send message of " << sensor_name << " fail: " << e.what());
            }
            if (!need_dump) {
                delete new_message;
            }
        }
    }
    return true;
}



bool DataWriter::DumpConsume() {
    time_point<system_clock> last = system_clock::now();
    while (true) {
        std::shared_ptr<Chunk> chunk;
        int waiting_chunks = 0;
        uint64_t record_time = 0;
        {
            std::lock_guard<std::mutex> lock(dump_mutex_);
            if (chunks_.size() > 0) {
                chunk = chunks_.front();
                record_time = record_times_.front();
                chunks_.pop();
                record_times_.pop();
                waiting_chunks = chunks_.size();
            }
        }
        std::string current_file_name = module_name_ + "_" + std::to_string(record_time);
        if (record_time > 0) {
            std::string path = output_dir_ + "/" + current_file_name; 
            std::ofstream ouf(path, std::ios::binary);
            time_point<system_clock> start = system_clock::now();
            chunk->SerializeToOstream(&ouf);
            time_point<system_clock> end = system_clock::now();
            duration<float> saving_elapsed = end - start;
            duration<float> last_chunk_elapsed = end - last;
            last = end;
            float disk_speed = (float)file_size_ / kMBSize / saving_elapsed.count();
            float disk_flow = (float)file_size_ / kMBSize / last_chunk_elapsed.count();
            ouf.close();
            INFO_MSG(std::endl << "flush file " << current_file_name << "; writing disk speed(mb/s): " << 
                    disk_speed << ", avg consume data flow(mb/s): " << disk_flow << std::endl <<
                    "waiting chunks: " << waiting_chunks << std::endl);
        } else {
            usleep(1000);
        }


        {
            std::lock_guard<std::mutex> lock(dump_mutex_);
            if (!dump_opened_ && chunks_.size() == 0) {
                break;
            }
        }
    }
    return true;
};
