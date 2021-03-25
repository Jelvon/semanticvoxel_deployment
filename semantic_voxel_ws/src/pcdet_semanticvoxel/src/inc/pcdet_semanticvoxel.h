#ifndef pcdet_semanticvoxel_h
#define pcdet_semanticvoxel_h
// Original packages refer to /usr/Desktop/pcdet_semanticvoxel_cpp
#include <iostream>
#include "voxelization.h"
#include "engine.h"
#include <time.h>
#include "torch/script.h"
#include "torch/torch.h"
#include <ATen/ATen.h>
#include "dlfcn.h"
#include <sys/stat.h>
#include "iou3d_nms.h"
#include "scatter_cuda.h"
#include <dirent.h>
// *********************************
// ROS dependency packages
#include <ros/ros.h>
#include <vision_msgs/Detection3DArray.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <pcdet_semanticvoxel/matrix2D_msg.h>


/**开发流程
 * 1、理解inference python代码
 * 2、找到tensorrt或jit.trace不支持的op
 * 3、对模型做适当的分割，转换成onnx或jit.trace
 * 4、补全中间代码（c++，cuda）
 * 5、测试
 * */
/* publish bounding boxes
para detectionsMessages
*/
class Pcdet_semanticvoxel
{
	public:
		Pcdet_semanticvoxel(const char *input_topic,const float score_thd_input, const float nms_overlap_thresh_input,
		   const float voxel_size_x, const float voxel_size_y, const float voxel_size_z, 
		   const float coors_range_xmin, const float coors_range_xmax,
		   const float coors_range_ymin, const float coors_range_ymax, 
		   const float coors_range_zmin, const float coors_range_zmax,const char *module_loc,const char *module_before_nms_loc,const char *backbone_loc);
		void doEveryThing();
	private:
		ros::Time timestart;
		std::unique_ptr<PaintVoxel> paintvoxel_reverse_ptr_;
		const char *topic;
		const int max_voxels = 16000;
    	const int max_points = 32;
    	const float dir_offset = 0.78539;
    	const float dir_limit_offset = 0.;
    	const float period = 3.141592653589793;
    	const int in_channels = 64;
    	const int nx = 432;
    	const int ny = 496;
        float score_thd = 0.8;
        float nms_overlap_thresh = 0.01;
    	const int NMS_POST_MAXSIZE = 500;

		size_t GetFileSize(const std::string& file_name);
		bool get_filelist_from_dir(std::string _path, std::vector<std::string>& _files);
	
		bool test;
		torch::Tensor points;
		// set ros handle
		ros::NodeHandle nh;
		
		ros::Publisher detectionsPublisher;

		ros::Subscriber pointCloudSubscriber;

	    std::string bin_root;
        std::string result_root;
        std::vector<std::string> bin_files;
        

        Engine backbone;
        torch::jit::script::Module module;
        torch::jit::script::Module module_before_nms;
    
        std::unique_ptr<ScatterCuda> scatter_cuda_ptr_;
    
        ALGErrCode ret;
    
        bool redo;


        std::vector<float> voxel_size;


        std::vector<float> coors_range;

		
		void pointsCloudSubCallback(const pcdet_semanticvoxel::matrix2D_msgConstPtr& pCloud);
        void publishDetectionArray(const at::Tensor final_boxes, const at::Tensor final_labels, const at::Tensor final_scores, const pcdet_semanticvoxel::matrix2D_msgConstPtr& pCloud);
		 
};


#endif
