//#include "cuda_runtime_api.h"
//#include "opencv2/opencv.hpp"
//#include <fstream>
#include <iostream>
//#include <chrono>
//#include "nvjpeg.h"
//#include "kernels.h"
#include "camera_collect_worker.h"
#include <map>
#include <stdio.h>
#include <string.h>
#include "cfg.h"

void syncv4l2EventCallBack(int nEvent, void *pHint1, void *pHint2, void *pUserData)
{
    switch (nEvent)
    {
    case SYNCV4L2_TIMESTAMP_ERROR:
    case SYNCV4L2_FRAME_ERROR:
        /*为了多通道同步，这里需要软重启*/
        SYNCV4L2_Stop();
        SYNCV4L2_Start();
        break;
    case SYNCV4L2_FRAME_LOSE:
    {
        SYNCV4L2_TEventFrameHint *pFrameEvent = (SYNCV4L2_TEventFrameHint *)pHint1;
        printf("frame lose,chan[%d],timestame[%ld.%09ld]\n", pFrameEvent->nChan, pFrameEvent->stTime.tv_sec, pFrameEvent->stTime.tv_nsec);
    }
    break;
    default:
        break;
    }
}

int main(int argc, char** argv) {
	/*
    float a = 1.f;
    float b = 0.f;
    float *d_a;
    float *d_b;
    std::cout << cudaMalloc((void**)&d_a, sizeof(float)) << std::endl;
    std::cout << cudaMalloc((void**)&d_b, sizeof(float)) << std::endl;
    std::cout << cudaMemcpy(d_a, &a, sizeof(float), cudaMemcpyHostToDevice) << std::endl;
    std::cout << cudaMemcpy(d_b, d_a, sizeof(float), cudaMemcpyDeviceToDevice) << std::endl;
    std::cout << cudaMemcpy(&b, d_b, sizeof(float), cudaMemcpyDeviceToHost) << std::endl;
    std::cout << cudaFree(d_a) << std::endl;
    std::cout << cudaFree(d_b) << std::endl;
    std::cout << b << std::endl;
    int width = 3840;
    int height = 2160;
    int origin_len = width * height * 3;
    std::vector<unsigned char> src(origin_len);
    for (int i1 = 0; i1 < origin_len; i1 ++) {
        int x = i1 / 3 / width;
	int y = i1 / 3 % width;
	src[i1] = int(sqrt(x * x + y * y)) % 255;
    }
    cv::Mat rgb(height, width, CV_8UC3, src.data());
    std::vector<unsigned char> data_encode;
    std::vector<int> params = {cv::ImwriteFlags::IMWRITE_JPEG_QUALITY, 80};
    time_point<system_clock> start = system_clock::now();
    bool err = cv::imencode(".jpg", rgb, data_encode, params);
    time_point<system_clock> end = system_clock::now();
    duration<float> elapsed = end - start;
    std::cout <<  data_encode.size() << std::endl;
    std::ofstream ouf("test.jpeg", std::ios::out | std::ios::binary);
    ouf.write(reinterpret_cast<const char*>(data_encode.data()), data_encode.size());
    std::cout <<  elapsed.count() << std::endl;
    ouf.close();

    nvjpegHandle_t nvjpeg_handle;
    nvjpegEncoderState_t encoder_state;
    nvjpegEncoderParams_t encode_params;
#define CHECK_NVJPEG(S) do { \
	nvjpegStatus_t status; \
	status = S; \
	if (status != NVJPEG_STATUS_SUCCESS) std::cout << __LINE__ << " CHECK_NVJPEG - state = " << status << std::endl; \
        } while (false);

    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);

    nvjpegImage_t input;
    nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_444;
    //nvjpegInputFormat_t input_format = NVJPEG_INPUT_BGRI;
    std::ifstream inf("../camera_6_3840_2160.yuv", std::ios::binary);
    inf.seekg(0, std::ios::end);
    size_t size = inf.tellg();
    inf.seekg(0, std::ios::beg);
    std::cout << "yuv size: " << size << std::endl;
    std::vector<unsigned char> yuv_src(size);
    inf.read(reinterpret_cast<char*>(yuv_src.data()), size);
    inf.close();
    unsigned char* d_src;
    std::cout << cudaMalloc((void**)&d_src, yuv_src.size()) << std::endl;
    cudaMemcpy(d_src, yuv_src.data(), yuv_src.size(), cudaMemcpyHostToDevice);

    unsigned char* fff_src;
    std::cout << cudaMalloc((void**)&fff_src, width * height * 3) << std::endl;
    yuv2to444(fff_src, d_src, width * height);
    input.channel[0] = fff_src;
    input.channel[1] = fff_src + width * height;
    input.channel[2] = fff_src + width * height * 2;
    input.pitch[0] = width;
    input.pitch[1] = width;
    input.pitch[2] = width;
    nvjpegBackend_t backend = NVJPEG_BACKEND_DEFAULT;
    CHECK_NVJPEG(nvjpegCreate(backend, nullptr, &nvjpeg_handle));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(nvjpeg_handle, &encode_params, NULL));
    CHECK_NVJPEG(nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, NULL));

    CHECK_NVJPEG(nvjpegEncoderParamsSetEncoding(encode_params, nvjpegJpegEncoding_t::NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN, NULL));
    CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(encode_params, 1, NULL));
    CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(encode_params, 80, NULL));
    CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(encode_params, nvjpegChromaSubsampling_t::NVJPEG_CSS_444, NULL));

    cudaEventRecord(ev_start);
    CHECK_NVJPEG(nvjpegEncodeYUV(nvjpeg_handle,
                    encoder_state,
                    encode_params,
                    &input,
                    subsampling,
                    width,
                    height,
                    NULL));

//    CHECK_NVJPEG(nvjpegEncodeImage(nvjpeg_handle, encoder_state, encode_params, &input, input_format, width, height, NULL));
    cudaEventRecord(ev_end);

    std::vector<unsigned char> cuda_data_encode;
    size_t length = 0;
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, NULL, &length, NULL));
    std::cout << "cuda len: " << length << std::endl;

    cuda_data_encode.resize(length);
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nvjpeg_handle, encoder_state, cuda_data_encode.data(), &length, NULL));
    cudaEventSynchronize(ev_end);
    nvjpegEncoderParamsDestroy(encode_params);
    nvjpegEncoderStateDestroy(encoder_state);
    nvjpegDestroy(nvjpeg_handle);

    float ms;
    cudaEventElapsedTime(&ms, ev_start, ev_end);
    std::cout << "cuda: " << ms << std::endl;
    std::ofstream ouf2("test_cuda.jpeg", std::ios::out | std::ios::binary);
    ouf2.write(reinterpret_cast<const char*>(cuda_data_encode.data()), cuda_data_encode.size());
    ouf2.close();
	*/
    if (argc != 2)
    {
        printf("sudo %s  <cfgname>\n", argv[0]);
        return -1;
    }

    //init
	std::string strCfg = argv[1];
    CFG_init(strCfg.c_str());

    SYNCV4L2_TSyncPara stuSyncPara;
    memset(&stuSyncPara, 0, sizeof(stuSyncPara));
    stuSyncPara.nMode = CFG_get_section_value_int(strCfg.c_str(), "common", "gps_mode", 0);
    stuSyncPara.nInitTriggerSignal = CFG_get_section_value_int(strCfg.c_str(), "common", "init_trigger_signal", 0);
    stuSyncPara.nSlave = CFG_get_section_value_int(strCfg.c_str(), "common", "slave", 0);
    stuSyncPara.nVersion = CFG_get_section_value_int(strCfg.c_str(), "common", "version", 1);
    CFG_get_section_value(strCfg.c_str(), "common", "trigger_dev_name", stuSyncPara.szDevname, sizeof(stuSyncPara.szDevname));
    stuSyncPara.nReset = CFG_get_section_value_int(strCfg.c_str(), "common", "reset", 0);
    SYNCV4L2_Init(&stuSyncPara);
    SYNCV4L2_SetEventCallback(syncv4l2EventCallBack, nullptr);

    //init camera chan
    std::map<int, CameraCollectWorker *> mapJpeg;
	const int MAX_CAMER_NUM = 8;
    for (int i = 0; i < MAX_CAMER_NUM; i++)
    {
        int chan = i;
		std::string tag = "video" + std::to_string(chan);

        if (CFG_get_section_value_int(strCfg.c_str(), tag.c_str(), "enable", 0) == 0)
        {
            continue;
        }

        CameraCollectWorker *pJpeg = new (std::nothrow) CameraCollectWorker(chan, strCfg);
        if (pJpeg == nullptr)
        {
            printf("new (std::nothrow)CameraCollectWorker failed,chan=%d\n", chan);
            continue;
        }

        pJpeg->Init();
        pJpeg->Open();
        mapJpeg[chan] = pJpeg;
    }

    //start
    SYNCV4L2_Start();
	
	getchar();

    //stop
    SYNCV4L2_Stop();
    for (auto it : mapJpeg)
    {
        it.second->Release();
    }
    SYNCV4L2_Release();
    CFG_free(strCfg.c_str());
 
    return 0;
}
