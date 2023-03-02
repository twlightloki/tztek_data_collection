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

    if (argc != 4 && argc != 6) {
        printf("./yuyv_jpeg path input_width, input_height, [output_width, output_height]\n");
        return 1;
    }
    int width = std::atoi(argv[2]);
    int height = std::atoi(argv[3]);
    int output_width = argc == 4 ? width : std::atoi(argv[4]);
    int output_height = argc == 4 ? height : std::atoi(argv[5]);
    NvBuffer *buf;
    std::ifstream inf(argv[1], std::ios::binary);

    std::unique_ptr<NvJPEGEncoder> jpegenc(NvJPEGEncoder::createJPEGEncoder("test"));
    buf = new NvBuffer(V4L2_PIX_FMT_YUV420M, output_width, output_height, 0);
    buf->allocateMemory();

    char* src;
    cudaMallocHost((void**)&src, width * height * 2);
    inf.read(src, width * height * 2);
    inf.close();
    YUYV2To420WithYCExtend((unsigned char*)src, buf->planes[0].data, buf->planes[1].data, buf->planes[2].data, 
                           output_width, output_height, width, height, 0);
    int rtn = cudaStreamSynchronize(0);
    assert(rtn == 0);
    rtn = cudaGetLastError();
    assert(rtn == 0);
    
    std::string yuv420_name = std::string(argv[1]) + ".420.yuv";
    std::ofstream yuv420f(yuv420_name, std::ios::binary);
    yuv420f.write(reinterpret_cast<char*>(buf->planes[0].data), output_width * output_height);
    yuv420f.write(reinterpret_cast<char*>(buf->planes[1].data), output_width * output_height / 4);
    yuv420f.write(reinterpret_cast<char*>(buf->planes[2].data), output_width * output_height / 4);
    yuv420f.close();


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
    ouf.close();
    return 0;
}

