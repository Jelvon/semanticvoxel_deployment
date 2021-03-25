#include "pcdet_visualization_node.h"
ros::Publisher markersPublisher;

void getDetectionMarkerCubeArray(const vision_msgs::Detection3DArrayConstPtr& detectionsMessages, visualization_msgs::MarkerArray *msg)
{
	int i;
	for(auto detection : detectionsMessages->detections) 
	{
		visualization_msgs::Marker bbx_marker; 
		bbx_marker.header.stamp = detectionsMessages->header.stamp;
		bbx_marker.header.frame_id = detectionsMessages->header.frame_id;
		bbx_marker.ns = "detection_cubes"; 
		bbx_marker.type = visualization_msgs::Marker::LINE_STRIP;
		bbx_marker.action = visualization_msgs::Marker::ADD;
		bbx_marker.frame_locked = false;

		bbx_marker.pose.orientation.w = 1.0;
		bbx_marker.scale.x = 0.15;

		bbx_marker.color.g = 0;
		bbx_marker.color.r = 1;
		bbx_marker.color.a = 1.0; 
		bbx_marker.points.clear();

		bbx_marker.color.b = 0;
		//bbx_marker.color.b = ceil(final_scores[i].item<float>() * 255.0);
		bbx_marker.id = i;

		geometry_msgs::Point Vs[8];
		int ct = 0;

		//float centerx = final_boxe   s[0][i][0].item<float>()-final_boxes[0][i][3].item<float>();
		//float centery = final_boxes[0][i][1].item<float>()+final_boxes[0][i][4].item<float>();
		//float centerz = final_boxes[0][i][2].item<float>()+final_boxes[0][i][5].item<float>();

		float w = detection.bbox.center.orientation.w;

		float centerx = detection.bbox.center.position.x;
		float centery = detection.bbox.center.position.y;
		float centerz = detection.bbox.center.position.z;
		for(int m=0;m<2;m++)
		{
			for(int n=0;n<2;n++)
			{
				for(int k=0;k<2;k++)
				{
					geometry_msgs::Point V;

					V.x = -(n-0.5)*detection.bbox.size.y*sin(w)+(m-0.5)*detection.bbox.size.x*cos(w) + centerx;
					V.y = (m-0.5)*detection.bbox.size.x*sin(w)+(n-0.5)*detection.bbox.size.y*cos(w) + centery;
					V.z = (k-0.5)*detection.bbox.size.z + centerz;
					Vs[ct] = V;
					ct += 1;
				}
			}
		}
		int drawOrder[] = {0,1,3,2,0,4,5,7,6,4,0,1,5,7,3,2,6};
		for(auto index:drawOrder)
		{
			bbx_marker.points.push_back(Vs[index]);
		}
		bbx_marker.lifetime = ros::Duration(0.1);
		//bbx_marker.text = (int) final_labels[i].item<long>();

		msg->markers.push_back(bbx_marker);
		i++;
	
  }
}

void getDetectionMarkerLabelArray(const vision_msgs::Detection3DArrayConstPtr& detectionsMessages, visualization_msgs::MarkerArray *msg)
{

  std::string types[] = {"car","cyclist", "pedestrian"};
  //msg2->markers.clear();
  int i=0;
  for(auto detection : detectionsMessages->detections)
  {
	visualization_msgs::Marker bbx_marker;
    bbx_marker.header = detectionsMessages->header;

    bbx_marker.ns = "detection_labels";
    
    bbx_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    bbx_marker.action = visualization_msgs::Marker::ADD;
    bbx_marker.frame_locked = false;
    bbx_marker.pose.orientation.x = 0;
    bbx_marker.pose.orientation.y = 0;
    bbx_marker.pose.orientation.z = 0;
    bbx_marker.color.b = 1;
    bbx_marker.color.g = 1;
    bbx_marker.color.r = 1;
    bbx_marker.color.a = 0.8;

	bbx_marker.header.stamp = detectionsMessages->header.stamp;
	bbx_marker.header.frame_id = detectionsMessages->header.frame_id; 
	bbx_marker.id = i;   
    bbx_marker.pose.position.x = detection.bbox.center.position.x;
    bbx_marker.pose.position.y = detection.bbox.center.position.y;
    bbx_marker.pose.position.z = detection.bbox.center.position.z;

    bbx_marker.pose.orientation.w = detection.bbox.center.orientation.w;
    bbx_marker.scale.x = detection.bbox.size.x;
    bbx_marker.scale.y = detection.bbox.size.y;
    bbx_marker.scale.z = detection.bbox.size.z;

    bbx_marker.lifetime = ros::Duration(0.1);


	int idx = (int) detection.results[0].id -1;
	float dist = round(1000*sqrt(pow(detection.bbox.size.x,2)+pow(detection.bbox.size.y,2)+pow(detection.bbox.size.z,2)))/float(1000);
	using boost::lexical_cast;
    bbx_marker.text =  "ID: " + lexical_cast<std::string>(i) + "\n" + types[idx] + "@" + lexical_cast<std::string>(round(detection.results[0].score*100)) + "\n" + lexical_cast<std::string>(dist) + "M";
    msg->markers.push_back(bbx_marker);
	i++;
  }

}
void publish_all(const vision_msgs::Detection3DArrayConstPtr& detectionsMessages)
{

	visualization_msgs::MarkerArray markerCubeMsg;
	visualization_msgs::MarkerArray markerLabelMsg;
	getDetectionMarkerLabelArray(detectionsMessages, &markerLabelMsg);
	getDetectionMarkerCubeArray(detectionsMessages, &markerCubeMsg);

	markersPublisher.publish(markerCubeMsg);
	markersPublisher.publish(markerLabelMsg);


}
int main(int argc, char **argv)
{
	
  // initial ros
  //ros::Rate rate(10);
  ros::init(argc, argv, "pcdet_visualization_node");
  ros::NodeHandle n;

  markersPublisher = n.advertise<visualization_msgs::MarkerArray>("marker_array",1);
  ros::Subscriber sub = n.subscribe<vision_msgs::Detection3DArray>("detection_publisher",1,publish_all);
  ros::spin();
  return 0; 
}
