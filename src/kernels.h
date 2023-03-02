#pragma once
#include "cuda_runtime_api.h"
#define CUDA_THREAD_NUM 512

void YUYV2To420WithYCExtend(const unsigned char *data, unsigned char *plane_0, unsigned char *plane_1, unsigned char *plane_2,
                            const int  width, const int height, cudaStream_t stream);
