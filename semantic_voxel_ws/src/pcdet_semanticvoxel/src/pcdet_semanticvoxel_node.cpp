#include "pcdet_semanticvoxel_node.h"

int main(int argc, char **argv)
{
	
  // initial ros
  //ros::Rate rate(10);
  ros::init(argc, argv, "pcdet_semanticvoxel");
  ros::NodeHandle n;
  std::string input_topic1;
  ros::param::get("~input_topic",input_topic1);
  const char *input_topic = input_topic1.c_str();
  std::string module_loc1;
  ros::param::get("~module_loc",module_loc1);
  const char *module_loc = module_loc1.c_str();
  std::string module_before_nms_loc1;
  ros::param::get("~module_before_nms_loc",module_before_nms_loc1);
  const char *module_before_nms_loc = module_before_nms_loc1.c_str();
  std::string backbone_loc1;
  ros::param::get("~backbone_loc",backbone_loc1);
  const char *backbone_loc = backbone_loc1.c_str();  
  float score_thd_input,nms_overlap_thresh_input,voxel_size_x,voxel_size_y,voxel_size_z,coors_range_xmin,coors_range_xmax,coors_range_ymin,coors_range_ymax,coors_range_zmin,coors_range_zmax;
  ros::param::get("~score_thd_input",score_thd_input);
  ros::param::get("~nms_overlap_thresh_input",nms_overlap_thresh_input);
  ros::param::get("~voxel_size_x",voxel_size_x);
  ros::param::get("~voxel_size_y",voxel_size_y);
  ros::param::get("~voxel_size_z",voxel_size_z);
  ros::param::get("~coors_range_xmin",coors_range_xmin);
  ros::param::get("~coors_range_xmax",coors_range_xmax);
  ros::param::get("~coors_range_ymin",coors_range_ymin);
  ros::param::get("~coors_range_ymax",coors_range_ymax);
  ros::param::get("~coors_range_zmin",coors_range_zmin);
  ros::param::get("~coors_range_zmax",coors_range_zmax);
  std::cout<<input_topic1<<std::endl;
  std::cout<<score_thd_input<<std::endl;
  Pcdet_semanticvoxel Pcdet_semanticvoxel(input_topic,score_thd_input,nms_overlap_thresh_input,voxel_size_x,voxel_size_y,voxel_size_z,coors_range_xmin,coors_range_xmax,coors_range_ymin,coors_range_ymax,coors_range_zmin,coors_range_zmax,module_loc,module_before_nms_loc,backbone_loc);
  Pcdet_semanticvoxel.doEveryThing();
  ros::spin();
  return 0;
}
