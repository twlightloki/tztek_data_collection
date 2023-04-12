#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <assert.h>
#include <iostream>
#include "NvJpegEncoder.h"
#include "NvLogging.h"
#include <fstream>
#include <memory>
#include "kernels.h"
#include "cuda_runtime.h"


int main(int argc, char** argv) {

    int width = std::atoi(argv[2]);
    int height = std::atoi(argv[3]);
    NvBuffer *buf;
    std::ifstream inf(argv[1], std::ios::binary);

    std::unique_ptr<NvJPEGEncoder> jpegenc(NvJPEGEncoder::createJPEGEncoder("test"));
    buf = new NvBuffer(V4L2_PIX_FMT_YUV420M, width, height, 0);
    buf->allocateMemory();

    char* src;
    cudaMallocHost((void**)&src, width * height * 2);
    inf.read(src, width * height * 2);
    YUYV2To420WithYCExtend((unsigned char*)src, buf->planes[0].data, buf->planes[1].data, buf->planes[2].data, width, height, 0);
    int rtn = cudaStreamSynchronize(0);
    assert(rtn == 0);
    rtn = cudaGetLastError();
    assert(rtn == 0);

    size_t jpeg_size = 10 * 1024 * 1024;
    unsigned char* jpeg_buf = new unsigned char[jpeg_size];
    rtn = jpegenc->encodeFromBuffer(*buf, JCS_YCbCr, &jpeg_buf, jpeg_size, 95);
    assert(rtn == 0);
    std::string jpeg_name = std::string(argv[1]) + ".jpeg";
    std::ofstream ouf(jpeg_name, std::ios::binary);
    ouf.write(reinterpret_cast<char*>(jpeg_buf), jpeg_size);
    cudaFree(src);
    delete jpeg_buf;
    delete buf;
    inf.close();
    ouf.close();
    return 0;
}

