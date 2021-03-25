#include "algorithm_sdk.h"
#include "opencv2/opencv.hpp"
#include "thread"
#include "mutex"
#include "vector"
#include "dirent.h"
#include <sys/time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fstream>

// ROS include
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
using namespace std;
using namespace cv;

ATHENA_algorithm::Segment_SDK m_smoke;
MAX_HANDLE mImage;
image_transport::Publisher result_publisher;
image_transport::Publisher result_color_publisher;


void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{

//    std::vector<cv::Mat> input_imgs;
    //double process_total = 0.0;


    Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;

    if (frame.empty()) {
        return;
    }
    // 图像预处理
    ATHENA_algorithm::ALGErrCode ret = m_smoke.image_uniformization(frame, mImage);
    if (ret != ATHENA_algorithm::ALGORITHM_OPERATION_SUCCESS) {
        std::cout << "error" << std::endl;
        return;
    }

    cv::Mat result = cv::Mat(frame.size(), CV_8UC1);
    cv::Mat result_color = cv::Mat(frame.size(), CV_8UC3);
    std::chrono::steady_clock::time_point process_start = std::chrono::steady_clock::now();

    // 模型推理
    ret = m_smoke.forward(mImage, result,result_color, true);
    std::chrono::steady_clock::time_point process_end = std::chrono::steady_clock::now();
    //process_total += std::chrono::duration<double, std::milli>(process_end - process_start).count();

    if (ret != ATHENA_algorithm::ALGORITHM_OPERATION_SUCCESS) {
        std::cout << "error" << std::endl;
        return;
    }
    sensor_msgs::Image result_msg;
	result_msg.header.stamp = msg->header.stamp;
	result_msg.header.frame_id = msg->header.frame_id;
    cv_bridge::CvImage result_bidge = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::MONO8, result);

	result_bidge.toImageMsg(result_msg); // from cv_bridge to sensor_msgs::Image

    result_publisher.publish(result_msg);
//        input_imgs.push_back(image.clone());


    //std::cout<<"process:"<<image_total<<"pictures,avg time: "<<process_total/image_total<<" ms"<<std::endl;

    

}

int main(int argc, char **argv)
{

	  ros::init(argc, argv, "deeplabv3_mobilnet");
	  ros::NodeHandle nh;
	  image_transport::ImageTransport it(nh);
	  std::string trtpath1;
	  ros::param::get("~trtpath",trtpath1);
	  std::string input_topic;
	  ros::param::get("~input_topic",input_topic);
	  const char *trtpath = trtpath1.c_str();
	  result_publisher = it.advertise("mobilnet/result", 1);
	  result_color_publisher = it.advertise("mobilnet/result_color", 1);
	  ATHENA_algorithm::ALGErrCode  ret =m_smoke.init(
		  trtpath,
		  1,
		  0
	  );
	  std::cout<<"ALGErrCode:"<<ret<<std::endl;
	  auto lib_version = ATHENA_algorithm::GetLibraryVersion();
	  std::cout<<"Alogirthm SDK version:"<<lib_version<<std::endl;

	  std:: string version =  m_smoke.getVersion();
	  std::cout<<"Current Model version:"<<version<<std::endl;

	  ATHENA_algorithm::ALGErrCode  ret2 = m_smoke.image_uniformization_create(&mImage);
	  if(ret2 != ATHENA_algorithm::ALGORITHM_OPERATION_SUCCESS){
	      return -1;
	  }

	  image_transport::Subscriber sub = it.subscribe(input_topic, 1, imageCallback);
	  ros::spin();
 
   //m_smoke.image_uniformization_free(&mImage);
   //return 0;
}
