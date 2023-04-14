#include "kernels.h"
#include "stdio.h"

__device__ unsigned char bilinear_functor(const unsigned char *src, const int width, const int height, const int stride, const int step,
        const float x, const float y) {
    int top_y = int(y);
    int bottom_y = min(top_y + 1, height - 1);
    int left_x = int(x);
    int right_x = min(left_x + 1, width - 1);

    double u = x - left_x; 
    double v = y - top_y; 
    return 
        (1.0 - u) * (1.0 - v) * (src + top_y * width + left_x / step * stride)[0] +
        u * (1.0 - v) * (src + top_y * width + right_x / step * stride)[0] +
        (1.0 - u) * v * (src + bottom_y * width + left_x / step * stride)[0] +
        u * v * (src + bottom_y * width + right_x / step * stride)[0];
}

__device__ unsigned char ycbcr(const int in, const int low, const int high) {
    return min(max(in - low, 0) * 255 / high, 255);
}


__global__ void yuyv_to_420_with_extend_and_resize_kernel(const unsigned char *src, unsigned char *plane_y, unsigned char *plane_u, unsigned char *plane_v, 
        const int width, const int height, const int in_width, const int in_height, const int count) {
    float width_scale = float(in_width) / width;
    float height_scale = float(in_height) / height;
    for(int i1 = threadIdx.x; i1 < count / 2; i1 += CUDA_THREAD_NUM) {
        int pos_y = i1 * 2 / width;
        int pos_x = (i1 * 2) % width;
        float y_y = (pos_y + 0.5) * height_scale - 0.5;
        float y_x0 = (pos_x + 0.5 ) * width_scale - 0.5;
        float y_x1 = (pos_x + 1 + 0.5 ) * width_scale - 0.5;


        unsigned char* dst_y0 = plane_y + i1 * 2;
        unsigned char* dst_y1 = plane_y + i1 * 2 + 1;
        *dst_y0 = ycbcr(bilinear_functor(src, in_width * 2, in_height, 2, 1, y_x0, y_y), 16, 219);
        *dst_y1 = ycbcr(bilinear_functor(src, in_width * 2, in_height, 2, 1, y_x1, y_y), 16, 219);
        if (pos_y % 2  == 0) {
            float uv_y = (pos_y + 1.0) * height_scale - 0.5;
            float uv_x = (pos_x + 1.0) * width_scale - 0.5;
            int uv_offset = (pos_y / 2) * (width / 2) + pos_x / 2;
            unsigned char* dst_u = plane_u + uv_offset;
            unsigned char* dst_v = plane_v + uv_offset;
            *dst_u = ycbcr(bilinear_functor(src + 1, in_width * 2, in_height, 4, 2, uv_x, uv_y), 16, 224);
            *dst_v = ycbcr(bilinear_functor(src + 3, in_width * 2, in_height, 4, 2, uv_x, uv_y), 16, 224);
        }
    }
};

__global__ void yuyv_to_420_with_extend_kernel(const unsigned char *src, unsigned char *plane_y, unsigned char *plane_u, unsigned char *plane_v, 
        const int width, const int height, const int count) {
    for(int i1 = threadIdx.x; i1 < count / 2; i1 += CUDA_THREAD_NUM) {
        int pos_y = i1 * 2 / width;
        int pos_x = (i1 * 2) % width;

        unsigned char* dst_y0 = plane_y + i1 * 2;
        unsigned char* dst_y1 = plane_y + i1 * 2 + 1;
        *dst_y0 = ycbcr((src + i1 * 4)[0], 16, 219);
        *dst_y1 = ycbcr((src + i1 * 4 + 2)[0], 16, 219);
        if (pos_y % 2  == 0) {
            float uv_y = (pos_y + 0.5);
            float uv_x = (pos_x + 0.5);
            int uv_offset = (pos_y / 2) * (width / 2) + pos_x / 2;
            unsigned char* dst_u = plane_u + uv_offset;
            unsigned char* dst_v = plane_v + uv_offset;
            *dst_u = ycbcr(bilinear_functor(src + 1, width * 2, height, 4, 2, uv_x, uv_y), 16, 224);
            *dst_v = ycbcr(bilinear_functor(src + 3, width * 2, height, 4, 2, uv_x, uv_y), 16, 224);
        }
    }
};


void YUYV2To420WithYCExtend(const unsigned char *data, unsigned char *plane_0, unsigned char *plane_1, unsigned char *plane_2,
        const int  width, const int height, const int input_width, const int input_height, cudaStream_t stream) {
    dim3 block(CUDA_THREAD_NUM);
    dim3 grid(1);
    int count = width * height;
    if (input_width == width && input_height == height) {
        yuyv_to_420_with_extend_kernel<<<grid, block, 0, stream>>>(data, plane_0, plane_1, plane_2, width, height, count);
    } else {
        yuyv_to_420_with_extend_and_resize_kernel<<<grid, block, 0, stream>>>(data, plane_0, plane_1, plane_2, width, height, input_width, input_height, count);
    }
}
