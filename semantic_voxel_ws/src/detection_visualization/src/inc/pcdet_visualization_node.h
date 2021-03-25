#ifndef PCDET_VISUALIZATION_NODE
#define PCDET_VISUALIZATION_NODE

#include <ros/ros.h>
#include <vision_msgs/Detection3DArray.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <cmath>
void getDetectionMarkerCubeArray(const vision_msgs::Detection3DArrayConstPtr& detectionsMessages, visualization_msgs::MarkerArray *msg);
void getDetectionMarkerLabelArray(const vision_msgs::Detection3DArrayConstPtr& detectionsMessages, visualization_msgs::MarkerArray *msg);
void publish_all(const vision_msgs::Detection3DArrayConstPtr& detectionsMessages);

#endif
