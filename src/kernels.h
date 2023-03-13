#pragma once
#include "cuda_runtime_api.h"
#define CUDA_THREAD_NUM 512

void yuv2to444(unsigned char* dst, const unsigned char* src, const int count, cudaStream_t stream = 0);
