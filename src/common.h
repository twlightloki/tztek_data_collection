#include <string>
#include <mutex>
#include <thread>
#include <queue>
#include <atomic>
#include <memory>
#include "NvLogging.h"

#define CHECK(S) \
    for (bool status = S; status != true;) { \
        ERROR_MSG(__LINE__ << "  check faild"); \
        return false; }


