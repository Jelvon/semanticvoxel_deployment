/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
* @file scatter_cuda.h
* @brief CUDA code for scatter operation
* @author Kosuke Murakami
* @date 2019/02/26
*/

#ifndef SCATTERCUDA_H
#define SCATTERCUDA_H
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include <iostream>
#include <torch/all.h>
class ScatterCuda
{
private:
  const int VOXELS_ROW_;
  const int VOXELS_COL_;
  const int MAX_INDICIES_;
  cudaStream_t stream;
public:
  /**
  * @brief Constructor
  * @param[in] VOXELS_COL The number of threads to launch cuda kernel
  * @param[in] VOXELS_ROW The number of threads to launch cuda block
  * @details Captital variables never change after the compile
  */
  ScatterCuda(const int VOXELS_ROW, const int VOXELS_COL);
  ~ScatterCuda();
  /**
  * @brief Call scatter cuda kernel
  * @param[in] pillar_count The valid number of pillars
  * @param[in] x_coors X-coordinate indexes for corresponding pillars
  * @param[in] y_coors Y-coordinate indexes for corresponding pillars
  * @param[in] pfe_output Output from Pillar Feature Extractor
  * @param[out] spatial_feature Gridmap representation for pillars' feature
  * @details Allocate pillars in gridmap based on index(coordinates) information
  */
  void doScatterCuda(const float* voxel_features, const long* indices, const int indices_size, float* spatial_feature);
};

class PaintVoxel
{
public:
    PaintVoxel();
    void paint_voxel_reverse(int voxel_num, torch::Tensor& voxels, torch::Tensor& num_points, torch::Tensor& coordinates,
                                    std::vector<float>& voxel_size, std::vector<float>& coors_range, at::Tensor &bev_map);

private:
    cudaStream_t stream;
};
#endif  // SCATTERCUDA_H
