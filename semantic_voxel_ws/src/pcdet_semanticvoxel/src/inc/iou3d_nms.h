#ifndef IOU3D_NMS_H
#define IOU3D_NMS_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int nms_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh);

#endif
