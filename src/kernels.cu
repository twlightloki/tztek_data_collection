#include "kernels.h"

__global__ void yuv2to444kernel(unsigned char* dst, const unsigned char* src, const int count) {
	for(int i1 = threadIdx.x; i1 < count / 2; i1 += CUDA_THREAD_NUM) {
		unsigned char* dst_y0 = dst + i1 * 2;
		unsigned char* dst_u0 = dst + count + i1 * 2;
		unsigned char* dst_v0 = dst + count * 2 + i1 * 2;
		unsigned char* dst_y1 = dst_y0 + 1;
		unsigned char* dst_u1 = dst_u0 + 1;
		unsigned char* dst_v1 = dst_v0 + 1;
		const unsigned char* src_y0 = src + i1 * 4;
		const unsigned char* src_u = src + i1 * 4 + 1;
		const unsigned char* src_y1 = src + i1 * 4 + 2;
		const unsigned char* src_v = src + i1 * 4 + 3;
		*dst_y0 = *src_y0;
		*dst_u0 = *src_u;
		*dst_v0 = *src_v;
		*dst_y1 = *src_y1;
		*dst_u1 = *src_u;
		*dst_v1 = *src_v;
	}
};
void yuv2to444(unsigned char* dst, const unsigned char* src, const int count, cudaStream_t stream) {
	dim3 block(CUDA_THREAD_NUM);
	dim3 grid(1);
	yuv2to444kernel<<<grid, block, 0, stream>>>(dst, src, count);
}
