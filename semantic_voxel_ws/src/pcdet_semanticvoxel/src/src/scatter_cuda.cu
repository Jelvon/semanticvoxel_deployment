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

//headers in local files
#include "scatter_cuda.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
__global__ void scatter_kernel( const float* voxel, const long* indices, const int voxel_row, const int voxel_col, const int indices_size, float* spatial_feature)
{
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  
  if(x >=indices_size)
  {
    return;
  }
  
  int _ind = indices[x];
  for(int y=0; y<voxel_row; y++)
  {
    float feature = voxel[y*indices_size + x];
    spatial_feature[y*voxel_col + _ind] = feature;
  }
  
}

ScatterCuda::ScatterCuda(const int VOXELS_ROW, const int VOXELS_COL):
VOXELS_ROW_(VOXELS_ROW),
VOXELS_COL_(VOXELS_COL),
MAX_INDICIES_(12000) // refer to autoware point pillars
{
  
  cudaStreamCreate(&stream);
}

ScatterCuda::~ScatterCuda()
{
  cudaError_t err_cuda = cudaStreamDestroy(stream);
}

void ScatterCuda::doScatterCuda(const float* voxel_features, const long* indices, const int indices_size, float* spatial_feature)
{
  dim3 block(1024);
  // int indices_ = std::min(indices_size, MAX_INDICIES_);
  int grid_x = (indices_size -1) / 1024 +1;
  dim3 grid(grid_x,
    1);
  scatter_kernel<<<block, grid, 0, stream>>>(voxel_features, indices, VOXELS_ROW_, VOXELS_COL_, indices_size, spatial_feature);
  cudaError_t err_cuda = cudaGetLastError();
	if(err_cuda!=cudaSuccess)
		printf("%s:%s\n", __FUNCTION__,cudaGetErrorString(err_cuda));
}


__global__ void paint_voxel_reverse_kernel(const int voxel_num, float* voxels, int* num_points, int* coordinates,
                                      float* voxel_size, float* coors_range, float* bev_map)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    const int channels_per_voxel = 3;
    const float fusion_voxel_split = 8.;
    if(i >=voxel_num)
    {
      return;
    }
    float voxel_scores[] = {0,0,0};
    int num_points_this_pillar = num_points[i];
//    auto points_in_voxel = voxels.select(0, i).slice(0, 0, num_points.data_ptr<int>()[i]);
    float* points_in_voxel = voxels + i*32*5; // points_in_voxel: shape: [num_points_this_pillar, 5]
    int z_idx[32] = {0}; // 32 = max points per pillar
    
    
    for(int j=0; j<num_points_this_pillar; j++)
    {
      z_idx[j] = floor( (points_in_voxel[j*5 + 2] - coors_range[2]) / (voxel_size[2] / fusion_voxel_split));
//        z_idx[j] = floor( (points_in_voxel[j*5 + 2] +3.f ) / (4.f / fusion_voxel_split));
    }

    for(int j=0; j<8; j++)
    {
      bool voxel_mask[32] = {false};
      int MaskNum = 0;
      float mean = 0.;
      for(int k=0; k<num_points_this_pillar; k++)
      {
        if(z_idx[k] == j)
        {
          voxel_mask[k] = true;
          MaskNum += 1;
          mean += points_in_voxel[k*5 + 4];
        }
      }
      if(MaskNum == 0)
      {
        continue;
      }
      if(coordinates[i * 3 + 1] >= 432 )
      {
        continue;
      }
        mean = mean / MaskNum;
        bev_map[(channels_per_voxel*j+0)*496*432 + coordinates[i*3 +1]*432 + coordinates[i*3 +2]] = mean;
        bev_map[(channels_per_voxel*j+1)*496*432 + coordinates[i*3 +1]*432 + coordinates[i*3 +2]] = mean;
        bev_map[(channels_per_voxel*j+2)*496*432 + coordinates[i*3 +1]*432 + coordinates[i*3 +2]] = mean;
    }
}

PaintVoxel::PaintVoxel(){
    cudaStreamCreate(&stream);
}

void PaintVoxel::paint_voxel_reverse(int voxel_num, torch::Tensor &voxels, torch::Tensor &num_points, torch::Tensor &coordinates,
                                std::vector<float>& voxel_size, std::vector<float>& coors_range, torch::Tensor &bev_map){
    auto start = std::chrono::steady_clock::now();
    dim3 block(1024);
    int grid_x = (voxel_num -1) / 1024 +1;
    dim3 grid(grid_x,
      1);
//     dim3 block(2);
//     int grid_x = (voxel_num -1) / 1024 +1;
//     dim3 grid(1,
//       1);
//    thrust::host_vector<float> h_voxel_size;
//    for(int i=0; i<voxel_size.size(); i++)
//    {
//        h_voxel_size.push_back(voxel_size[i]);
//    }

//    thrust::host_vector<float> h_coors_range;
//    for(int i=0; i<coors_range.size(); i++)
//    {
//        h_coors_range.push_back(coors_range[i]);
//    }
////    thrust::device_vector<float> d_voxel_size = h_voxel_size;
////    thrust::device_vector<float> d_coors_range = h_coors_range;

    float* d_voxel_size = nullptr;
    float* d_coors_range = nullptr;
    cudaMalloc(&d_voxel_size, voxel_size.size()*sizeof(float));
    cudaMalloc(&d_coors_range, coors_range.size()*sizeof(float));
    cudaMemcpy(d_voxel_size, voxel_size.data(), voxel_size.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coors_range, coors_range.data(), coors_range.size()*sizeof(float), cudaMemcpyHostToDevice);

    paint_voxel_reverse_kernel<<<block, grid, 0, stream>>>(voxel_num, voxels.data_ptr<float>(), num_points.data_ptr<int>(), coordinates.data_ptr<int>(),
                                          d_voxel_size, d_coors_range, bev_map.data_ptr<float>());
    cudaError_t err_cuda = cudaGetLastError();
    if(err_cuda!=cudaSuccess)
      printf("%s:%s\n", __FUNCTION__,cudaGetErrorString(err_cuda));
    auto end = std::chrono::steady_clock::now();
    std::cout << "paint_voxel_reverse time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000 << "ms" << std::endl;
    cudaFree(d_voxel_size);
    cudaFree(d_coors_range);
}
