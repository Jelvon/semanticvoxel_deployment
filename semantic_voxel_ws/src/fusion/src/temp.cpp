#include <iostream>
#include "inc/voxelization.h"
#include "inc/engine.h"
#include <time.h>
#include "torch/script.h"
#include "torch/torch.h"
#include <ATen/ATen.h>
#include "dlfcn.h"
#include <sys/stat.h>
#include "iou3d_nms.h"
//#include "scatter_cuda.h"
#include <dirent.h>
#include <opencv2/opencv.hpp>
// *********************************


/**开发流程
 * 1、理解inference python代码
 * 2、找到tensorrt或jit.trace不支持的op
 * 3、对模型做适当的分割，转换成onnx或jit.trace
 * 4、补全中间代码（c++，cuda）
 * 5、测试
 * */
bool get_filelist_from_dir(std::string _path, std::vector<std::string>& _files)
{
    DIR* dir;
    dir = opendir(_path.c_str());
    struct dirent* ptr;
    std::vector<std::string> file;
    while((ptr = readdir(dir)) != NULL)
    {
        if(ptr->d_name[0] == '.') {continue;}
        file.push_back(ptr->d_name);
    }
    closedir(dir);
    sort(file.begin(), file.end());
    _files = file;
}

size_t GetFileSize(const std::string& file_name){
    struct stat stat_buf;;
    stat(file_name.c_str(), &stat_buf);
    size_t size = stat_buf.st_size;
    return size; //单位是：byte
}

torch::Tensor fusion_points(int col, int row, std::string pointsPath, std::string maskImg)
{
    const double f_rect[] = {9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03, 0.,
            -9.869795000000e-03, 9.999421000000e-01, -4.278459000000e-03, 0.,
            7.402527000000e-03, 4.351614000000e-03, 9.999631000000e-01, 0.,
            0., 0., 0., 1.};

    const double f_P2[] = {7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01,
                    0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01,
                    0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03,
                    0., 0., 0., 1.};

    const double f_Trv2c[] = {7.533745000000e-03, -9.999714000000e-01, -6.166020000000e-04, -4.069766000000e-03,
                       1.480249000000e-02, 7.280733000000e-04, -9.998902000000e-01, -7.631618000000e-02,
                       9.998621000000e-01, 7.523790000000e-03, 1.480755000000e-02, -2.717806000000e-01,
                       0., 0., 0., 1.};

    const double f_z_points[] = {1.e-03, 1.e-03, 1.e-03, 1.e-03, 1.e+02, 1.e+02, 1.e+02, 1.e+02};

    const double near_clip = 0.001;
    const double far_clip = 100;

    /**
     * @brief read bin file add pic
     */
    int size_1 = GetFileSize(pointsPath) / sizeof(float);
    torch::Tensor points = torch::from_file(pointsPath, c10::nullopt, size_1, torch::requires_grad(false).dtype(at::kFloat));
    points.resize_({size_1/4, 4});
    cv::Mat rgb_image_label = cv::imread(maskImg);

    auto start0 = std::chrono::steady_clock::now();
    /**
     * projection_matrix_to_CRT_kitti
    */
    torch::Tensor rect = torch::rand({4, 4}, at::requires_grad(false).dtype(at::kDouble));
    torch::Tensor P2 = torch::rand({4, 4}, at::requires_grad(false).dtype(at::kDouble));
    torch::Tensor Trv2c = torch::rand({4, 4}, at::requires_grad(false).dtype(at::kDouble));
    std::memcpy(rect.data_ptr(), f_rect, sizeof(double)*16);
    std::memcpy(P2.data_ptr(), f_P2, sizeof(double)*16);
    std::memcpy(Trv2c.data_ptr(), f_Trv2c, sizeof(double)*16);

    auto CR = P2.slice(0, 0, 3).slice(1, 0, 3);
    auto CT = P2.slice(0, 0, 3).slice(1,3,4).squeeze(1);
    auto RinvCinv = CR.inverse();
    auto qr = RinvCinv.qr();
    torch::Tensor Rinv = std::get<0>(qr);
    torch::Tensor Cinv = std::get<1>(qr);
    torch::Tensor C = Cinv.inverse();
    torch::Tensor R = Rinv.inverse();
    torch::Tensor T = Cinv.matmul(CT);

    /**
     * get_frustum
     */
    auto fku = C[0][0];
    auto fkv = -C[1][1];
    auto u0v0 = C.slice(0,0,2).slice(1, 2, 3).squeeze(1);
    torch::Tensor box_corners = torch::zeros({4,2}, at::requires_grad(false).dtype(at::kInt)).contiguous();
    box_corners[1][1] = row;
    box_corners[2][0] = col;
    box_corners[2][1] = row;
    box_corners[3][0] = col;
    torch::Tensor z_points = torch::rand({8, 1}, at::requires_grad(false).dtype(at::kDouble)).contiguous();
    std::memcpy(z_points.data_ptr(), f_z_points, sizeof(double)*8);
    torch::Tensor temp = torch::rand({2}, at::requires_grad(false).dtype(at::kDouble)).contiguous();
    temp[0] = fku /near_clip;
    temp[1] = -fkv / near_clip;
    torch::Tensor near_box_corners = (box_corners - u0v0) / temp;
    temp[0] = fku / far_clip;
    temp[1] = -fkv /far_clip;
    torch::Tensor far_box_corners = (box_corners - u0v0) / temp;
    torch::Tensor ret_xy = torch::cat({near_box_corners, far_box_corners}, 0);
    torch::Tensor ret_xyz = torch::cat({ret_xy, z_points}, 1);
    torch::Tensor frustum = ret_xyz - T;
    frustum = R.inverse().matmul(frustum.t());
    torch::Tensor temp_points = frustum.t();
    /**
      * camera_to_lidar
      * */
    if(temp_points.size(1) == 3)
    {
        temp_points = torch::cat({temp_points, torch::ones({temp_points.size(0), 1}, at::requires_grad(false).dtype(at::kDouble)).contiguous()}, 1);
    }
    torch::Tensor lidar_points = temp_points.matmul( rect.matmul(Trv2c).t().inverse() );
    frustum = lidar_points.slice(1, 0, 3).clone();
    /**
     * corner_to_surfaces_3d_jit
    **/
    int num_boxes = 1;
    torch::Tensor surfaces = torch::zeros({num_boxes, 6, 4, 3}, at::requires_grad(false).dtype(at::kDouble)).contiguous();
    int corner_idxes[] = {0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6, 7};
    auto p_surfaces = surfaces.data_ptr<double>();
    auto p_frustum = frustum.data_ptr<double>();
//    std::cout<<frustum<<std::endl;
    for(int i=0; i<num_boxes; i++)
    {
        for(int j=0; j<6; j++)
        {
            for(int k=0; k<4; k++)
            {
//                surfaces[i][j][k] = frustum[corner_idxes[j*4+k]];
                p_surfaces[i*6*4*3 + j*4*3 + k*3 +0] = p_frustum[corner_idxes[j*4+k]*3 + 0];
                p_surfaces[i*6*4*3 + j*4*3 + k*3 +1] = p_frustum[corner_idxes[j*4+k]*3 + 1];
                p_surfaces[i*6*4*3 + j*4*3 + k*3 +2] = p_frustum[corner_idxes[j*4+k]*3 + 2];
            }
        }
    }
//    std::cout<<surfaces<<std::endl;
    /**
     * points_in_convex_polygon_3d_jit
    **/
    int max_num_surfaces =6, max_num_points_of_surface=4;
    int num_points = size_1/4;
    int num_polygons = surfaces.size(0);
    /**
     * surface_equ_3d_jit
    **/
    torch::Tensor polygon_surfaces = surfaces.slice(2,0,3);
    auto surface_vec = polygon_surfaces.slice(2, 0, 2) - polygon_surfaces.slice(2, 1, 3);
    auto normal_vec = torch::cross(surface_vec.select(2, 0), surface_vec.select(2, 1));
    torch::Tensor d = torch::einsum("aij, aij->ai", {normal_vec, polygon_surfaces.select(2, 0)}) * -1;
    torch::Tensor indices = torch::ones({num_points, num_polygons}, at::requires_grad(false).dtype(at::kBool)).contiguous();
    double sign = 0.0;

    auto p_points = points.data_ptr<float>();
    auto p_normal_vec = normal_vec.data_ptr<double>();
    auto p_d = d.data_ptr<double>();
    auto p_i = indices.data_ptr<bool>();
    for(int i=0; i<num_points; i++)
    {
        for(int j=0; j<num_polygons; j++)
        {

            for(int k=0; k<max_num_surfaces; k++)
            {
                if(k > 9999999)
                {
                    break;
                }
                sign = p_points[i*4 + 0] * p_normal_vec[j*6*3 + k*3 + 0] \
                        + p_points[i*4 + 1] * p_normal_vec[j*6*3 + k*3 + 1] \
                        + p_points[i*4 + 2] * p_normal_vec[j*6*3 + k*3 + 2] + p_d[j*6 +k];
                if(sign >= 0)
                {
                    p_i[i*num_polygons +j] = false;
                    break;
                }
            }
        }
    }
    auto points_v = points.index(indices.view(-1));

    /**
     * _get_semantic_segmentation_result
     **/

    torch::Tensor image_label = torch::rand({rgb_image_label.rows, rgb_image_label.cols}, at::requires_grad(false).dtype(at::kDouble)).contiguous();
    auto p_image_label = image_label.data_ptr<double>();
    for(int i=0; i<rgb_image_label.rows; i++)
    {
        for(int j=0; j<rgb_image_label.cols; j++)
        {
            unsigned char blue = rgb_image_label.at<cv::Vec3b>(i,j)[0];
            if(blue == 24)
            {
                p_image_label[i*rgb_image_label.cols + j] = 80;
            }
            else if(blue == 25)
            {
                p_image_label[i*rgb_image_label.cols + j] = 120;
            }
            else if(blue == 33)
            {
                p_image_label[i*rgb_image_label.cols + j] = 160;
            }
            else if(blue == 26)
            {
                p_image_label[i*rgb_image_label.cols + j] = 40;
            }
            else
            {
                p_image_label[i*rgb_image_label.cols + j] = blue;
            }
        }
    }
    for(int i=0; i<rgb_image_label.rows; i++)
    {
        for(int j=0; j<rgb_image_label.cols; j++)
        {
            unsigned char blue = rgb_image_label.at<cv::Vec3b>(i,j)[0];
            if(blue <= 34)
            {
                p_image_label[i*rgb_image_label.cols + j] = 0;
            }
        }
    }
    image_label = image_label / 40.;

    /**
     *   _add_class_score
     */
    torch::Tensor points_xyz = points_v.slice(1, 0, 3);
    torch::Tensor reflectance = points_v.slice(1, 3, 4);
    /**
     * lidar_to_camera
     */
    if(points_xyz.size(1) == 3)
    {
        temp_points = torch::cat({points_xyz, torch::ones({points_xyz.size(0), 1}, at::requires_grad(false).dtype(at::kDouble)).contiguous()}, 1);
    }
    torch::Tensor camera_points = temp_points.matmul( rect.matmul(Trv2c).t());
    torch::Tensor points_v_to_c = camera_points.slice(1, 0, 3);
    /**
     * project_to_image
     */
    temp_points = torch::cat({points_v_to_c, torch::zeros({points_v_to_c.size(0), 1}, at::requires_grad(false).dtype(at::kDouble)).contiguous()}, -1);
    torch::Tensor point_2d = temp_points.matmul(P2.t());
    torch::Tensor points_v_to_image = point_2d.slice(1, 0, 2) / point_2d.slice(1, 2, 3);

    num_points = points_v_to_image.size(0);
    torch::Tensor class_score = torch::zeros({num_points, 1}, at::requires_grad(false).dtype(at::kFloat)).contiguous();
    auto p_class_score = class_score.data_ptr<float>();
    p_image_label = image_label.data_ptr<double>();
    auto p_points_v_to_image = points_v_to_image.data_ptr<double>();
    for(int num=0; num<num_points; num++)
    {
        int u = int( std::round(p_points_v_to_image[num*2+1]) );
        int v = int( std::round(p_points_v_to_image[num*2+0]) );
        if(u<image_label.size(0) && v<image_label.size(1))
        {
//            auto watch = p_image_label[u*rgb_image_label.cols + v];
            p_class_score[num] = p_image_label[u*rgb_image_label.cols + v];
        }
    }
    torch::Tensor points_cat = torch::cat({points_xyz, reflectance, class_score}, 1);
    auto p_point_cat = points_cat.data_ptr<float>();
    std::vector<int> rider;
    std::vector<int> bicycle;
    for(int num=0; num<points_cat.size(0); num++)
    {
        if(p_point_cat[num*5+4] == 4)
            bicycle.push_back(num);
        else if(p_point_cat[num*5+4] == 3)
            rider.push_back(num);
    }
    p_point_cat = points_cat.data_ptr<float>();
    for(int i=0; i<bicycle.size(); i++)
    {
        double xb, yb, zb;
        xb = p_point_cat[i*5 + 0];
        yb = p_point_cat[i*5 + 1];
        zb = p_point_cat[i*5 + 2];
        for(int j=0; j<rider.size(); j++)
        {
            double xr, yr, zr;
            xr = p_point_cat[i*5 + 0];
            yr = p_point_cat[i*5 + 1];
            zr = p_point_cat[i*5 + 2];
            if( (xb-xr)*(xb-xr) + (yb-yr)*(yb-yr) + (zb-zr)*(zb-zr) <= 1)
            {
                p_point_cat[i*5 + 4] = 3;
            }
            else
            {
                p_point_cat[i*5 + 4]=0;
            }
        }
    }

    auto end0 = std::chrono::steady_clock::now();
    std::cout << "fuse time=" << std::chrono::duration_cast<std::chrono::microseconds>(end0 - start0).count() / 1000.f << "ms" << std::endl;
    return points_cat.toType(at::kFloat);
}

torch::Tensor paint_voxel_reverse_kernel(int voxel_num, torch::Tensor& voxels, torch::Tensor& num_points, torch::Tensor& coordinates,
                                         std::vector<float>& voxel_size, std::vector<float>& coors_range)
{
    auto start = std::chrono::steady_clock::now();
    auto cp_num_points = num_points.cpu();
    auto cpu_coordinates = coordinates.cpu();
    const float fusion_voxel_split = 8.f;
    int channels_per_voxel = 3;
    auto voxel_scores = torch::zeros({channels_per_voxel}, at::requires_grad(false).dtype(at::kFloat)).cuda().contiguous();
    auto p_coordinates = cpu_coordinates.data_ptr<int>();
    auto bev_map = torch::zeros({1, 24, 496, 432}, at::requires_grad(false).dtype(at::kFloat)).cuda().contiguous();
    for(int i=0; i<voxel_num; i++)
    {

        auto points_in_voxel = voxels.select(0, i).slice(0, 0, cp_num_points.data_ptr<int>()[i]);
        auto z_idx = torch::floor((points_in_voxel.select(1,2) -  coors_range[2]) / (voxel_size[2] / fusion_voxel_split) );
        for(int j=0; j<8; j++)
        {

            auto voxel_mask = (z_idx == j);
            voxel_mask = voxel_mask.repeat({1, 5}).view({5, -1}).transpose(0,1);
            auto points = points_in_voxel.masked_select(voxel_mask).view({-1, 5});
            if(points.size(0) == 0)
            {
                continue;
            }
            if(p_coordinates[i*3 + 2] > 432)
            {
                continue;
            }

            for(int chan=0; chan<channels_per_voxel; chan++)
            {
                voxel_scores[chan] = torch::mean(points.select(1, 4));
            }
            bev_map[0][channels_per_voxel*j+0][0][p_coordinates[i*3 + 1]] = voxel_scores[0];
            bev_map[0][channels_per_voxel*j+1][0][p_coordinates[i*3 + 1]] = voxel_scores[1];
            bev_map[0][channels_per_voxel*j+2][0][p_coordinates[i*3 + 1]] = voxel_scores[2];

        }

    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000 << "ms" << std::endl;
    return bev_map;
}

int main()
{
    //std::string bin_root = "/home/xavier/Desktop/demo_cpp/velodyne/";
    std::string bin_root = "../velodyne/";
    std::string result_root = "./output/";
    std::vector<std::string> bin_files;
    get_filelist_from_dir(bin_root, bin_files);
    // init start
    Engine backbone;
    torch::jit::script::Module module;
    torch::jit::script::Module module_before_nms;
    try
    {
        module = torch::jit::load("./model/pp_fusion_pfn.pt");
        module.to(torch::kCUDA);

        module_before_nms = torch::jit::load("./model/before_nms_script.pt");
        module_before_nms.to(torch::kCUDA);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model " << e.what() << std::endl;
        return -1;
    }
    std::unique_ptr<ScatterCuda> scatter_cuda_ptr_;
    std::unique_ptr<PaintVoxel> paintvoxel_reverse_ptr_;

    ALGErrCode ret = backbone.initEngine("../model/fusion_backbone.trt");
   if (ret != ALGORITHM_OPERATION_SUCCESS)
   {
       return -1;
   }
    bool redo = false;
    const int max_voxels = 16000;
    const int max_points = 32;
    const float dir_offset = 0.78539;
    const float dir_limit_offset = 0.;
    const float period = 3.141592653589793;
    const int in_channels = 64;
    const int nx = 432;
    const int ny = 496;
    scatter_cuda_ptr_.reset(new ScatterCuda(in_channels, nx*ny));
    paintvoxel_reverse_ptr_.reset(new PaintVoxel());
    std::vector<float> voxel_size;
    voxel_size.push_back(0.16);
    voxel_size.push_back(0.16);
    voxel_size.push_back(4);

    std::vector<float> coors_range;
    coors_range.push_back(0);
    coors_range.push_back(-39.68);
    coors_range.push_back(-3);
    coors_range.push_back(69.12);
    coors_range.push_back(39.68);
    coors_range.push_back(1);

    const float nms_overlap_thresh = 0.01;
    const int NMS_POST_MAXSIZE = 500;

    torch::Tensor batch_cls_preds = torch::zeros({1, 321408, 3}, at::requires_grad(false).dtype(at::kFloat)).cuda().contiguous();
    torch::Tensor batch_box_preds = torch::zeros({1, 321408, 7}, at::requires_grad(false).dtype(at::kFloat)).cuda().contiguous();
    torch::Tensor dir_labels = torch::zeros({1, 321408}, at::requires_grad(false).dtype(at::kInt)).cuda().contiguous();
    for (int i = 0; i < bin_files.size(); i++)
    {
        auto spatial_feature = torch::zeros({in_channels, nx * ny}, torch::requires_grad(false).dtype(at::kFloat)).cuda().contiguous();

        std::string binfile = bin_root + bin_files[i];
        std::cout<<binfile<<std::endl;
        int size_1 = GetFileSize(binfile) / sizeof(float);
        torch::Tensor points_1 = torch::from_file(binfile, c10::nullopt, size_1, torch::requires_grad(false).dtype(at::kFloat));
        points_1.resize_({size_1/5, 5}).cuda();
        torch::Tensor points = points_1.to(at::kFloat).cuda();
        auto voxels = torch::zeros({max_voxels, max_points, points.size(1)}, torch::requires_grad(false).dtype(torch::kFloat)).cuda().contiguous();
        auto coors = torch::zeros({max_voxels, 3}, torch::requires_grad(false).dtype(torch::kInt)).cuda().contiguous();
        auto num_points_per_voxel = torch::zeros(max_voxels, torch::requires_grad(false).dtype(torch::kInt)).cuda().contiguous();
        auto start_total = std::chrono::steady_clock::now();
        auto start = std::chrono::steady_clock::now();
        int voxel_num = voxelization::hard_voxelize(points, voxels, coors, num_points_per_voxel, voxel_size, coors_range, max_points, max_voxels, 3);
        auto end = std::chrono::steady_clock::now();
        std::cout << "hard_voxelize time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000 << "ms" << std::endl;
        std::cout<<"pillar num = "<<voxel_num<<std::endl;
        auto bev_map = torch::zeros({1, 24, 496, 432}, at::requires_grad(false).dtype(at::kFloat)).cuda().contiguous();
        paintvoxel_reverse_ptr_->paint_voxel_reverse(voxel_num, voxels, num_points_per_voxel, coors, voxel_size, coors_range, bev_map);
//        std::cout<<bev_map[0][21][0][248]<<std::endl;
        torch::Tensor zero_tensor = torch::zeros({max_voxels, 1}, torch::requires_grad(false).dtype(at::kInt)).cuda();
        coors = torch::cat({zero_tensor, coors}, 1); //
        start = std::chrono::steady_clock::now();
        std::vector<torch::jit::IValue> inputs = {voxels, num_points_per_voxel, coors};
        /**
         * vfe选用trace部署：因为trt对vfe中的激活函数不友好，加速之后变得很慢
         * */
        auto ans = module.forward(inputs).toTensor();
        end = std::chrono::steady_clock::now();
        std::cout << "vfe_pfn time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000 << "ms" << std::endl;
        auto voxel_features = ans.slice(0, 0, voxel_num).squeeze(1); //.squeeze(0).squeeze(1)
        auto _coors = coors.slice(0, 0, voxel_num);

        // middle encoder start
        // Create the canvas for this sample
        start = std::chrono::steady_clock::now();
        auto this_coors = _coors;
        torch::Tensor t_indices = this_coors.slice(1, 1, 2, 1) + this_coors.slice(1, 2, 3, 1) * nx + this_coors.slice(1, 3, 4, 1); // [xxx, 1]
        auto indices = t_indices.toType(at::kLong).view(-1).contiguous();
        auto temp_voxels = voxel_features.t().contiguous(); // temp_voxels: [64, xxxx]
        /**
         * cuda函数实现index_put操作，因为当前版本的torch，index_put函数有内存溢出
         * */
        scatter_cuda_ptr_->doScatterCuda(temp_voxels.data_ptr<float>(), indices.data_ptr<long>(), indices.size(0), spatial_feature.data_ptr<float>());
        end = std::chrono::steady_clock::now();
        std::cout << "cuda scatter time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000 << "ms" << std::endl;
        // Undo the column stacking to final 4-dim tensor
        torch::Tensor spatial_features = spatial_feature.view({1, in_channels, ny, nx}).contiguous();

        // backbone start
        start = std::chrono::steady_clock::now();
        cudaMemset(batch_cls_preds.data_ptr(), 0, 321408 * 3 * sizeof(float));
        cudaMemset(batch_box_preds.data_ptr(), 0, 321408 * 7 * sizeof(float));
        cudaMemset(dir_labels.data_ptr(), 0, 321408 * sizeof(float));
        end = std::chrono::steady_clock::now();
        std::cout << "tensor alcocate time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000 << "ms" << std::endl;



        vector<void *> buffers = {spatial_features.data_ptr(),
                                    bev_map.data_ptr(),
                                  batch_cls_preds.data_ptr(),
                                  batch_box_preds.data_ptr(),
                                  dir_labels.data_ptr()};
        start = std::chrono::steady_clock::now();
        /**
         * tensorrt backend
         * */
        ret = backbone.infer(buffers, 1);
        end = std::chrono::steady_clock::now();
        std::cout << "backbone time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000 << "ms" << std::endl;
        if (ret != ALGORITHM_OPERATION_SUCCESS)
        {
            return -1;
        }

        // limit period
        auto val = batch_box_preds.slice(2, 6, 7, 1).squeeze(2) - dir_offset;
        auto dir_rot = val - torch::floor(val / period) * period;
        batch_box_preds = torch::cat({batch_box_preds.slice(2, 0, 6), (dir_rot + dir_offset + period * dir_labels).unsqueeze_(2)}, 2).contiguous();

        start = std::chrono::steady_clock::now();
        /**
         * trace：好处少写代码
         * */
        std::vector<torch::jit::IValue> inputs_1 = {batch_box_preds, batch_cls_preds};
        auto before_nms = module_before_nms.forward(inputs_1).toTuple();
        auto cls_preds = before_nms->elements()[0].toTensor();
        auto label_preds = before_nms->elements()[1].toTensor();
        auto scores_mask = before_nms->elements()[2].toTensor();
        auto boxes_for_nms = before_nms->elements()[3].toTensor().contiguous();
        auto order = before_nms->elements()[4].toTensor();
        auto indices_topk = before_nms->elements()[5].toTensor();
        torch::Tensor keep = torch::zeros({boxes_for_nms.size(0)}, at::requires_grad(false).dtype(at::kLong)).cpu().contiguous();
        int64_t index = 0;
        int num_out = 0;
        if(redo)
        {
            // auto cpu_indices = indices.cpu();
            // int id_0 = cpu_indices.data_ptr<long>()[0];
            // int id_1 = cpu_indices.data_ptr<long>()[1];
            // std::cout<<spatial_feature[0][id_0]<<std::endl;
            // std::cout<<spatial_feature[0][id_1]<<std::endl;

            std::cout<<batch_box_preds[0][0]<<std::endl;
            std::cout<<batch_box_preds[0][1]<<std::endl;
            std::cout<<batch_cls_preds[0][0]<<std::endl;
            std::cout<<batch_cls_preds[0][1]<<std::endl;
            redo = false;
        }
        if(boxes_for_nms.size(0) > 0)
        {
            num_out = nms_gpu(boxes_for_nms, keep, 0.01);
        }
        else
        {
            std::cout<<"zero predict"<<std::endl;
            redo=true;
            // auto cpu_indices = indices.cpu();
            // int id_0 = cpu_indices.data_ptr<long>()[0];
            // int id_1 = cpu_indices.data_ptr<long>()[1];
            // std::cout<<spatial_feature[0][id_0]<<std::endl;
            // std::cout<<spatial_feature[0][id_1]<<std::endl;
            std::cout<<batch_box_preds[0][0]<<std::endl;
            std::cout<<batch_box_preds[0][1]<<std::endl;
            std::cout<<batch_cls_preds[0][0]<<std::endl;
            std::cout<<batch_cls_preds[0][1]<<std::endl;
            i--;
        }
        keep = keep.slice(0,0,num_out).cuda();
        auto keep_idx = order.index_select(0, keep);
        if(keep_idx.size(0) > NMS_POST_MAXSIZE)
            keep_idx = keep_idx.slice(0, 0 , NMS_POST_MAXSIZE);
        auto selected = indices_topk.index_select(0,keep_idx);
        auto original_idxs = scores_mask.nonzero().view(-1);
        selected = original_idxs.index_select(0, selected);
        auto final_scores = cls_preds.index_select(0, selected);
        auto final_labels = label_preds.index_select(0, selected);
        auto final_boxes = batch_box_preds.index_select(1, selected);
        end = std::chrono::steady_clock::now();
        std::cout << "nms time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000 << "ms" << std::endl;
        std::cout << "total time=" << std::chrono::duration_cast<std::chrono::microseconds>(end - start_total).count() / 1000 << "ms" << std::endl;
        ofstream out;
        std::string txt = bin_files[i].substr(0, bin_files[i].find_last_of("."))+".txt";
        std::cout<<"save "<<txt<<std::endl;
        out.open(result_root + txt, ios::trunc);
        out<<final_scores<<std::endl;
        out<<final_labels<<std::endl;
        out<<final_boxes<<std::endl;
        out.close();
    }
    return 0;
}
