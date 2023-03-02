#include "kernels.h"
#include "stdio.h"

__global__ void yuyv_to_420_with_extend_kernel(const unsigned char *src, unsigned char *plane_y, unsigned char *plane_u, unsigned char *plane_v, 
        const int width, const float width_scale, const float height_scale, const int count) {
    for(int i1 = threadIdx.x; i1 < count / 2; i1 += CUDA_THREAD_NUM) {
        int pos_x = i1 * 2 / width;
        int pos_y = (i1 * 2) % width;
        int input_x = pos_x * height_scale;
        int input_y = pos_y * width_scale;


        int src_offset = input_x * width * width_scale + input_y;
        const unsigned char* src_y0 = src + src_offset * 2;
        const unsigned char* src_y1 = src_y0 + 2;

        unsigned char* dst_y0 = plane_y + i1 * 2;
        unsigned char* dst_y1 = plane_y + i1 * 2 + 1;
        *dst_y0 = min(max(int(*src_y0) - 16, 0) * 255 / 219, 255);
        *dst_y1 = min(max(int(*src_y1) - 16, 0) * 255 / 219, 255);
        if (pos_x % 2  == 0) {
            int uv_offset = (pos_x / 2) * (width / 2) + pos_y / 2;
            unsigned char* dst_u = plane_u + uv_offset;
            unsigned char* dst_v = plane_v + uv_offset;
            const unsigned char* src_u = src_y0 + 1;
            const unsigned char* src_v = src_y0 + 3;
            unsigned char u = min(max(int(*src_u) - 16, 0) * 255 / 224, 255);
            unsigned char v = min(max(int(*src_v) - 16, 0) * 255 / 224, 255);

            *dst_u = u;
            *dst_v = v;
        }
    }
};
void YUYV2To420WithYCExtend(const unsigned char *data, unsigned char *plane_0, unsigned char *plane_1, unsigned char *plane_2,
        const int  width, const int height, const int input_width, const int input_height, cudaStream_t stream) {
    dim3 block(CUDA_THREAD_NUM);
    dim3 grid(1);
    int count = width * height;
    float width_scale = float(input_width) / width;
    float height_scale = float(input_height) / height;
    yuyv_to_420_with_extend_kernel<<<grid, block, 0, stream>>>(data, plane_0, plane_1, plane_2, width, width_scale, height_scale, count);
}
