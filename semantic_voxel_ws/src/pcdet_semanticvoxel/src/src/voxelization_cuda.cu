#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);       \
    CHECK_CONTIGUOUS(x)

namespace {
int const threadsPerBlock = sizeof(unsigned long long) * 8;
}

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
    i += blockDim.x * gridDim.x)



template <typename T_int>
__global__ void get_xyz(const T_int* coor,T_int* x, T_int* y, T_int* z,T_int* locx, const int num_points,const int NDim) {

    	CUDA_1D_KERNEL_LOOP(i, num_points) {
        	auto prev_coor = coor + i * NDim;

		x[i] = prev_coor[0];
        	y[i] = prev_coor[1];
        	z[i] = prev_coor[2];
		locx[i] = i;
	}
}
template <typename T, typename T_int>
__global__ void dynamic_voxelize_kernel(
        const T* points, T_int* coors, const float voxel_x, const float voxel_y,
        const float voxel_z, const float coors_x_min, const float coors_y_min,
        const float coors_z_min, const float coors_x_max, const float coors_y_max,
        const float coors_z_max, const int grid_x, const int grid_y,
        const int grid_z, const int num_points, const int num_features,
        const int NDim) {
    //   const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    CUDA_1D_KERNEL_LOOP(index, num_points) {
        // To save some computation
        auto points_offset = points + index * num_features;
        auto coors_offset = coors + index * NDim;
        int c_x = floor((points_offset[0] - coors_x_min) / voxel_x);
        if (c_x < 0 || c_x >= grid_x) {
            coors_offset[0] = -1;
            return;
        }

        int c_y = floor((points_offset[1] - coors_y_min) / voxel_y);
        if (c_y < 0 || c_y >= grid_y) {
            coors_offset[0] = -1;
            coors_offset[1] = -1;
            return;
        }

        int c_z = floor((points_offset[2] - coors_z_min) / voxel_z);
        if (c_z < 0 || c_z >= grid_z) {
            coors_offset[0] = -1;
            coors_offset[1] = -1;
            coors_offset[2] = -1;
        } else {
            coors_offset[0] = c_z;
            coors_offset[1] = c_y;
            coors_offset[2] = c_x;
        }
    }
}

template <typename T, typename T_int>
__global__ void assign_point_to_voxel(const int nthreads, const T* points,
                                      T_int* point_to_voxelidx,
                                      T_int* coor_to_voxelidx, T* voxels,
                                      const int max_points,
                                      const int num_features,
                                      const int num_points, const int NDim) {
    CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
        // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
        int index = thread_idx / num_features;

        int num = point_to_voxelidx[index];
        int voxelidx = coor_to_voxelidx[index];
        if (num > -1 && voxelidx > -1) {
            auto voxels_offset =
                    voxels + voxelidx * max_points * num_features + num * num_features;

            int k = thread_idx % num_features;
            voxels_offset[k] = points[thread_idx];
        }
    }
}

template <typename T, typename T_int>
__global__ void assign_voxel_coors(const int nthreads, T_int* coor,
                                   T_int* point_to_voxelidx,
                                   T_int* coor_to_voxelidx, T_int* voxel_coors,
                                   const int num_points, const int NDim) {
    CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
        // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
        // if (index >= num_points) return;
        int index = thread_idx / NDim;
        int num = point_to_voxelidx[index];
        int voxelidx = coor_to_voxelidx[index];
        if (num == 0 && voxelidx > -1) {
            auto coors_offset = voxel_coors + voxelidx * NDim;
            int k = thread_idx % NDim;
            coors_offset[k] = coor[thread_idx];
        }
    }
}

template <typename T_int>
__global__ void point_to_voxelidx_kernel(const T_int* coor,
                                         T_int* point_to_voxelidx,
                                         T_int* point_to_pointidx,

					 T_int* loc,
					 T_int* is_no_friend,
                                         const int max_points,
                                         const int max_voxels,
                                         const int num_points, const int NDim) {

    CUDA_1D_KERNEL_LOOP(i, num_points) {

	int index = loc[i];
	is_no_friend[index] = 0;
	auto coor_offset = coor + index * NDim;
        // skip invalid points
        assert(index < num_points);

	if (coor_offset[0] == -1) {
		is_no_friend[index] = 0;
		continue;
	}
        int num = 0;
        int coor_x = coor_offset[0];
        int coor_y = coor_offset[1];
        int coor_z = coor_offset[2];
	int min_same_index = num_points;
        for (int j=i-1;j >= 0;j --) {
	    int index1 = loc[j];
            auto coor_offset1 = coor + index1 * NDim;
            if (coor_offset1[0] == -1) break;
	    

            if ((coor_offset1[0] == coor_x) && (coor_offset1[1] == coor_y) && (coor_offset1[2] == coor_z)) {
			if (index1 > index) continue;
                	num++;
                	if (index1 < min_same_index) {

				min_same_index = index1;
                	} 
			if(num >= max_points)
			{
				is_no_friend[index] = 0;
				return;
			}

	    }
	    else {
		break;
	    }
        }

        for (int j=i+1;j < num_points;j ++) {
	    int index1 = loc[j];
            auto coor_offset1 = coor + index1 * NDim;
            if (coor_offset1[0] == -1) break;

            if ((coor_offset1[0] == coor_x) && (coor_offset1[1] == coor_y) && (coor_offset1[2] == coor_z)) {
                	if (index1 > index) continue;
			num++;
                	if (index1 < min_same_index) {

				min_same_index = index1;
                	} 
			if(num >= max_points)
			{
				is_no_friend[index] = 0;
				return;
			}
	    }
	    else {
		break;
	    }
        }
        if (num == 0) {
            point_to_pointidx[index] = index;
	    is_no_friend[index] = 1;
        }
        else {
            point_to_pointidx[index] = min_same_index;
	    is_no_friend[index] = 0;
        }
        if (num < max_points) {
            point_to_voxelidx[index] = num;
        }
    }
    /*
    CUDA_1D_KERNEL_LOOP(index, num_points) {
        auto coor_offset = coor + index * NDim;
        // skip invalid points
        if ((index >= num_points) || (coor_offset[0] == -1)) return;

        int num = 0;
        int coor_x = coor_offset[0];
        int coor_y = coor_offset[1];
        int coor_z = coor_offset[2];
        // only calculate the coors before this coor[index]
        for (int i = 0; i < index; ++i) {
            auto prev_coor = coor + i * NDim;
            if (prev_coor[0] == -1) continue;

            // Find all previous points that have the same coors
            // if find the same coor, record it
            if ((prev_coor[0] == coor_x) && (prev_coor[1] == coor_y) &&
                    (prev_coor[2] == coor_z)) {
                num++;
                if (num == 1) {
                    // point to the same coor that first show up
                    point_to_pointidx[index] = i;
                } else if (num >= max_points) {
                    // out of boundary
                    return;
                }
            }
        }
        if (num == 0) {
            point_to_pointidx[index] = index;
        }
        if (num < max_points) {
            point_to_voxelidx[index] = num;
        }
    }  
    */
                                                                     
}

template <typename T_int>
__global__ void determin_voxel_num(
        // const T_int* coor,
        T_int* num_points_per_voxel, T_int* point_to_voxelidx,
        T_int* point_to_pointidx, T_int* coor_to_voxelidx, T_int* voxel_num,T_int* point_to_voxelidx_before_nonzero,T_int* loc, const int max_points, const int max_voxels, const int num_points) {
    // only calculate the coors before this coor[index]
    CUDA_1D_KERNEL_LOOP(idx, num_points) {
    //for (int i = 0; i < num_points; ++i) {
        // if (coor[i][0] == -1)
        //    continue;
	int i = loc[idx];
        int point_pos_in_voxel = point_to_voxelidx[i];
        // record voxel
        if (point_pos_in_voxel == -1) {
            // out of max_points or invalid point
            continue;
        } else if (point_pos_in_voxel == 0) {
            // record new voxel
            int voxelidx = point_to_voxelidx_before_nonzero[i];
            if (point_to_voxelidx_before_nonzero[i] >= max_voxels) break;
            coor_to_voxelidx[i] = voxelidx;
	    int count = 1;
            for(int j=idx+1;j<num_points;j++) {
		int ii = loc[j];
		int point_pos_in_voxelii = point_to_voxelidx[ii];
		if(point_pos_in_voxelii == -1) continue;
		else if(point_pos_in_voxelii > 0) count++;
		else break;
	    }
            num_points_per_voxel[voxelidx] = count;
        } else {
            int point_idx = point_to_pointidx[i];
            int voxelidx = point_to_voxelidx_before_nonzero[point_idx];
            if (voxelidx != -1) {
                coor_to_voxelidx[i] = voxelidx;
            }
        }
    }

}
/*
template <typename T_int>
__global__ void pointcloud2Tensor(T_int *points,const pcl::PointCloud<pcl::PointXYZI> &cloud, const int num_points) {
	CUDA_1D_KERNEL_LOOP(i, num_points) {
		points[i][0] = cloud.points[i].x;
		points[i][1] = cloud.points[i].y;
		points[i][2] = cloud.points[i].z;
		points[i][3] = cloud.points[i].intensity;
	}

}
*/
struct ZipComparator
{
    __host__ __device__
    inline bool operator() (const thrust::tuple<int, int, int> &a, const thrust::tuple<int, int, int> &b)
    {
        if(a.get<0>() < b.get<0>()) {
	   return true;
	}
	else if(a.get<0>() > b.get<0>()) {
	   return false;
	}
	else {
	   if(a.get<1>() < b.get<1>()) {
		return true;
	   }
	   else if (a.get<1>() > b.get<1>()) {
		return false;
	   }
	   else {
		return a.get<2>() < b.get<2>();
	   }
        }
		
    }
};

namespace voxelization {
/*
void pointcloud2Tensor(at::Tensor& points,const pcl::PointCloud<pcl::PointXYZI> &cloud, const int num_points)
{

    dim3 grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
    dim3 block(512);
	pointcloud2Tensor_kernal<int><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(points.contiguous().data_ptr<int>(),&cloud, num_points);

}
*/
int hard_voxelize_gpu(const at::Tensor& points, at::Tensor& voxels,
                      at::Tensor& coors, at::Tensor& num_points_per_voxel,
                      const std::vector<float> voxel_size,
                      const std::vector<float> coors_range,
                      const int max_points, const int max_voxels,
                      const int NDim = 3) {
    // current version tooks about 0.04s for one frame on cpu
    // check device
    CHECK_INPUT(points);

    at::cuda::CUDAGuard device_guard(points.device());

    const int num_points = points.size(0);
    const int num_features = points.size(1);

    const float voxel_x = voxel_size[0];
    const float voxel_y = voxel_size[1];
    const float voxel_z = voxel_size[2];
    const float coors_x_min = coors_range[0];
    const float coors_y_min = coors_range[1];
    const float coors_z_min = coors_range[2];
    const float coors_x_max = coors_range[3];
    const float coors_y_max = coors_range[4];
    const float coors_z_max = coors_range[5];

    const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
    const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
    const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

    // map points to voxel coors
    at::Tensor temp_coors =
            at::zeros({num_points, NDim}, points.options().dtype(at::kInt));

    dim3 grid(std::min(at::cuda::ATenCeilDiv(num_points, 1024), 4096));
    dim3 block(1024);

    // 1. link point to corresponding voxel coors
    float *points_data = points.contiguous().data_ptr<float>();
    dynamic_voxelize_kernel<float, int>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                                                                     points_data,
                                                                     temp_coors.contiguous().data_ptr<int>(), voxel_x, voxel_y,
                                                                     voxel_z, coors_x_min, coors_y_min, coors_z_min, coors_x_max,
                                                                     coors_y_max, coors_z_max, grid_x, grid_y, grid_z, num_points,
                                                                     num_features, NDim);
    cudaDeviceSynchronize();
    AT_CUDA_CHECK(cudaGetLastError());

    // 2. map point to the idx of the corresponding voxel, find duplicate coor
    // create some temporary variables
    auto point_to_pointidx = -at::ones(
    {
                    num_points,
                },
                points.options().dtype(at::kInt));
    auto point_to_voxelidx = -at::ones(
    {
                    num_points,
                },
                points.options().dtype(at::kInt));

    dim3 map_grid(std::min(at::cuda::ATenCeilDiv(num_points, 1024), 4096));
    dim3 map_block(1024);

    int *x, *y, *z, *locx, *is_no_friend;
     cudaMalloc((void**)&x,sizeof(int)*num_points);
     cudaMalloc((void**)&y,sizeof(int)*num_points);
     cudaMalloc((void**)&z,sizeof(int)*num_points);
     cudaMalloc((void**)&locx,sizeof(int)*num_points);
     cudaMalloc((void**)&is_no_friend,sizeof(int)*num_points);

     get_xyz<int><<<map_grid, map_block, 0,at::cuda::getCurrentCUDAStream()>>>(temp_coors.contiguous().data_ptr<int>(),x,y,z,locx,num_points,NDim);
                          
     thrust::device_ptr<int> dev_x(x);
     thrust::device_ptr<int> dev_y(y);
     thrust::device_ptr<int> dev_z(z);
     thrust::device_ptr<int> dev_locx(locx); 
     thrust::device_ptr<int> dev_is_no_friend(is_no_friend);
     auto keytup_begin = thrust::make_tuple(dev_x,dev_y,dev_z);
     auto first =thrust::make_zip_iterator(keytup_begin);
     auto keytup_end = thrust::make_tuple(dev_x+num_points,dev_y+num_points,dev_z+num_points);
     auto end =thrust::make_zip_iterator(keytup_end);
     thrust::sort_by_key(thrust::device,first, end, dev_locx, ZipComparator());
     cudaFree(x);
     cudaFree(y);
     cudaFree(z);
     //cudaFree(first);

     point_to_voxelidx_kernel<int>
            <<<map_grid, map_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                                                                             temp_coors.contiguous().data_ptr<int>(),
                                                                             point_to_voxelidx.contiguous().data_ptr<int>(),
                                                                             point_to_pointidx.contiguous().data_ptr<int>(),
									     locx,
                                                                             is_no_friend,
                                                                             max_points,
                                                                             max_voxels, num_points, NDim);
 
    cudaDeviceSynchronize();
    AT_CUDA_CHECK(cudaGetLastError());

    auto voxel_num = at::zeros(
    {
                    1,
                },
                points.options().dtype(at::kInt));  // must be zero from the begining
   int offset = 0;
   if(dev_is_no_friend[num_points-1] == 0) {
	offset = 0;
   } else {
	offset = 1;
   } thrust::exclusive_scan(thrust::device,dev_is_no_friend,dev_is_no_friend+num_points,dev_is_no_friend);
    //std::cout<<dev_is_no_friend[num_points-1]<<std::endl;
    voxel_num[0] = offset+dev_is_no_friend[num_points-1];
    dim3 map_grid1(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
    dim3 map_block1(512);
    // 3. determin voxel num and voxel's coor index
    // make the logic in the CUDA device could accelerate about 10 times
    auto coor_to_voxelidx = -at::ones(
    {
                    num_points,
                },
                points.options().dtype(at::kInt));
    

    determin_voxel_num<int><<<map_grid1, map_block1, 0, at::cuda::getCurrentCUDAStream()>>>(
                                                                             num_points_per_voxel.contiguous().data_ptr<int>(),
                                                                             point_to_voxelidx.contiguous().data_ptr<int>(),
                                                                             point_to_pointidx.contiguous().data_ptr<int>(),
                                                                             coor_to_voxelidx.contiguous().data_ptr<int>(),
                                                                             voxel_num.contiguous().data_ptr<int>(), is_no_friend, locx, 
max_points, max_voxels, num_points);
    cudaDeviceSynchronize();
    AT_CUDA_CHECK(cudaGetLastError());
    //std::cout<<voxel_num[0]<<std::endl;
    cudaFree(is_no_friend);
    cudaFree(locx);
    // 4. copy point features to voxels
    // Step 4 & 5 could be parallel
    auto pts_output_size = num_points * num_features;
    dim3 cp_grid(std::min(at::cuda::ATenCeilDiv(pts_output_size, 1024), 4096));
    dim3 cp_block(1024);

    assign_point_to_voxel<float, int>
            <<<cp_grid, cp_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                                                                           pts_output_size, points.contiguous().data_ptr<float>(),
                                                                           point_to_voxelidx.contiguous().data_ptr<int>(),
                                                                           coor_to_voxelidx.contiguous().data_ptr<int>(),
                                                                           voxels.contiguous().data_ptr<float>(), max_points, num_features,
                                                                           num_points, NDim);
    //cudaDeviceSynchronize();
    //AT_CUDA_CHECK(cudaGetLastError());

    // 5. copy coors of each voxels

    auto coors_output_size = num_points * NDim;
    dim3 coors_cp_grid(
                std::min(at::cuda::ATenCeilDiv(coors_output_size, 1024), 4096));
    dim3 coors_cp_block(1024);
    assign_voxel_coors<float, int><<<coors_cp_grid, coors_cp_block, 0,
            at::cuda::getCurrentCUDAStream()>>>(
                                                  coors_output_size, temp_coors.contiguous().data_ptr<int>(),
                                                  point_to_voxelidx.contiguous().data_ptr<int>(),
                                                  coor_to_voxelidx.contiguous().data_ptr<int>(),
                                                  coors.contiguous().data_ptr<int>(), num_points, NDim);

    cudaDeviceSynchronize();
    AT_CUDA_CHECK(cudaGetLastError());

    auto voxel_num_cpu = voxel_num.to(at::kCPU);
    int voxel_num_int = voxel_num_cpu.data_ptr<int>()[0];

    return voxel_num_int;
}

void dynamic_voxelize_gpu(const at::Tensor& points, at::Tensor& coors,
                          const std::vector<float> voxel_size,
                          const std::vector<float> coors_range,
                          const int NDim = 3) {
    // current version tooks about 0.04s for one frame on cpu
    // check device
    CHECK_INPUT(points);

    at::cuda::CUDAGuard device_guard(points.device());

    const int num_points = points.size(0);
    const int num_features = points.size(1);

    const float voxel_x = voxel_size[0];
    const float voxel_y = voxel_size[1];
    const float voxel_z = voxel_size[2];
    const float coors_x_min = coors_range[0];
    const float coors_y_min = coors_range[1];
    const float coors_z_min = coors_range[2];
    const float coors_x_max = coors_range[3];
    const float coors_y_max = coors_range[4];
    const float coors_z_max = coors_range[5];

    const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
    const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
    const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

    const int col_blocks = at::cuda::ATenCeilDiv(num_points, threadsPerBlock);
    dim3 blocks(col_blocks);
    dim3 threads(threadsPerBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dynamic_voxelize_kernel<float, int><<<blocks, threads, 0, stream>>>(
                                                                             points.contiguous().data_ptr<float>(),
                                                                             coors.contiguous().data_ptr<int>(), voxel_x, voxel_y, voxel_z,
                                                                             coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max,
                                                                             coors_z_max, grid_x, grid_y, grid_z, num_points, num_features, NDim);
    cudaDeviceSynchronize();
    AT_CUDA_CHECK(cudaGetLastError());

    return;
}

}  // namespace voxelization
