/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include <torch/serialize/tensor.h>
// #include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "iou3d_nms.h"

#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

#define CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;

void nmsLauncher(const float *boxes, unsigned long long * mask, int boxes_num, float nms_overlap_thresh);

int nms_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh){
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params keep: (N)
    CHECK_INPUT(boxes);
    CHECK_CONTIGUOUS(keep);

    int boxes_num = boxes.size(0);
    if(boxes_num == 0)
    {
      return 0;
    }
    const float * boxes_data = boxes.data<float>();
    long * keep_data = keep.data<long>();

    const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

    unsigned long long *mask_data = NULL;
    CHECK_ERROR(cudaMalloc((void**)&mask_data, boxes_num * col_blocks * sizeof(unsigned long long)));
    nmsLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

    // unsigned long long mask_cpu[boxes_num * col_blocks];
    // unsigned long long *mask_cpu = new unsigned long long [boxes_num * col_blocks];
    std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);

//    printf("boxes_num=%d, col_blocks=%d\n", boxes_num, col_blocks);
    CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data, boxes_num * col_blocks * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost));

    cudaFree(mask_data);

    unsigned long long remv_cpu[col_blocks];
    memset(remv_cpu, 0, col_blocks * sizeof(unsigned long long));

    int num_to_keep = 0;

    for (int i = 0; i < boxes_num; i++){
        int nblock = i / THREADS_PER_BLOCK_NMS;
        int inblock = i % THREADS_PER_BLOCK_NMS;

        if (!(remv_cpu[nblock] & (1ULL << inblock))){
            keep_data[num_to_keep++] = i;
            unsigned long long *p = &mask_cpu[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++){
                remv_cpu[j] |= p[j];
            }
        }
    }
    cudaError_t err_cuda = cudaGetLastError();
    if ( cudaSuccess != err_cuda )
    {
        printf("%s:%s\n", __FUNCTION__,cudaGetErrorString(err_cuda));
    }

    return num_to_keep;
}



