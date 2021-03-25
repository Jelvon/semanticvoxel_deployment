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
#include "scatter_cuda.h"
#include <dirent.h>
#include <opencv2/opencv.hpp>
// ROS include

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <fusion/matrix2D_msg.h>
ros::Publisher point_pub;


torch::Tensor fusion_points(int col, int row, const torch::Tensor points, const cv::Mat rgb_image_label)
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
     /*
    int size_1 = GetFileSize(pointsPath) / sizeof(float);
    torch::Tensor points = torch::from_file(pointsPath, c10::nullopt, size_1, torch::requires_grad(false).dtype(at::kFloat));
    points.resize_({size_1/4, 4});
    cv::Mat rgb_image_label = cv::imread(maskImg);
	*/
	
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
    **/
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
    **/
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
    int num_points = points.size(0);
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
            if(blue == 11)
            {
                p_image_label[i*rgb_image_label.cols + j] = 80;
            }
            else if(blue == 12)
            {
                p_image_label[i*rgb_image_label.cols + j] = 120;
            }
            else if(blue == 33)            {
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
     * add_class_score
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
//          auto watch = p_image_label[u*rgb_image_label.cols + v];
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

void callback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::PointCloud2ConstPtr& pCloud)
{
	// pointcloud to tensor
	int num_points = pCloud->width;
	int dims  = 4;
	int point_array[dims*num_points];
	auto options = torch::TensorOptions().dtype(torch::kFloat);
	memcpy(&point_array,&(pCloud->data[0]),num_points*dims*sizeof(int));
	torch::Tensor points = torch::from_blob(point_array,{num_points,dims},options);
	points = points.cuda();
	// image_msg to rgb_image_label
	cv::Mat rgb_image_label = cv_bridge::toCvShare(image_msg, "bgr8")->image;

	torch::Tensor point_Matrix = fusion_points(rgb_image_label.cols, rgb_image_label.rows, points, rgb_image_label);
	point_Matrix = point_Matrix.contiguous();
	fusion::matrix2D_msg pArray_pub;

    pArray_pub.x_label = "point_index";
    pArray_pub.y_label = "point_attr";
    pArray_pub.x_size = point_Matrix.size(0);
    pArray_pub.y_size = point_Matrix.size(1);
    pArray_pub.x_stride = point_Matrix.size(0)*point_Matrix.size(1);
    pArray_pub.y_stride = point_Matrix.size(0);
	std::vector<float> v(point_Matrix.data_ptr<float>(),point_Matrix.data_ptr<float>()+point_Matrix.numel());
	pArray_pub.data = v;
	point_pub.publish(pArray_pub);
}
int main(int argc, char **argv)
{
  ros::init(argc, argv, "deeplabv3_mobilnet");
  ros::NodeHandle nh;

  std::string pointcloud_topic;
  ros::param::get("~pointcloud_topic",pointcloud_topic);
  std::string image_topic;
  ros::param::get("~image_topic",image_topic);

  point_pub = nh.advertise<fusion::matrix2D_msg>("fusion_points", 1);

  message_filters::Subscriber<sensor_msgs::Image> image_sub(nh,image_topic, 1);
  message_filters::Subscriber<sensor_msgs::PointCloud2> pointcloud_sub(nh,pointcloud_topic, 1);
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;

  typedef message_filters::Synchronizer<MySyncPolicy> sync;
  boost::shared_ptr<sync> sync_;
  //sync_.reset(new sync(MySyncPolicy(10), image_sub, pointcloud_sub));
  //sync_->registerCallback(boost::bind(&callback, _1, _2));
	
  ros::spin();
  return 0;
}
