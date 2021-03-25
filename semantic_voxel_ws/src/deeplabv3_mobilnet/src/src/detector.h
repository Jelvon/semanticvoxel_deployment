#ifndef DETECTOR_SDK_H
#define DETECTOR_SDK_H

#include "algorithm_sdk_error_code.h"
#include "engine.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#ifndef WIN32
#include "syslog.h"
#endif
#include <common_export.h>


namespace ATHENA_algorithm {

#define IMAGE_UNIFORMIZATION_NAME "frame"
    struct Image_uniformization
    {
        std::string name = IMAGE_UNIFORMIZATION_NAME;
        int batch = 0;
        int width = 0;
        int height = 0;
        int channel= 0;
        std::vector<int> src_image_width;//原始图像宽

        std::vector<int> src_image_height;//原始图像长

        std::vector<void *> device_data;
        std::vector<void *> host_data;

        // 图片在显存上的指针
        // 由于提前申请内存和显存
        // Create device buffers
        void *image_data_d = nullptr;
        std::vector< std::vector <float>> trans_mat;
//        float data[] = { 1, 2, 3,
//                         4, 5, 6 };
//        torch::Tensor f = torch::from_blob(data, {2, 3});
    };


    std::vector<cv::Scalar> get_color_list();

class Detector_SDK {
public:
    Detector_SDK()
            :detector(), MAX_BATCHSIZE(32) ,inputCount(1),outputCount(1), isInitialized(false),
            FLAGS_Detect_size_W(224),FLAGS_Detect_size_H(224),FLAGS_Detect_size_C(3){
        version = "";
        color_list = get_color_list();
    }

    ~Detector_SDK(){

    }

    /**
     * SDK初始化函数，模型输入采用经加密软件加密过后的模型
     * @param modelPath 模型路径
     * @param maxBatchSize 设置输入的最大Batchsize大小
     * @param encrtyption 模型是否加密
     * @param deviceID 设置GPU ID
     * @return
     */
    virtual ALGErrCode init(
            const char * modelPath,
            const int maxBatchSize,
            int deviceID = 0,
            int max_threads = 1);


    /**
     * 图像预处理主要是做resize，同时为了保证线程安全，算法中间数据全部由该函数申请
     * @param input
     * @param outputs
     * @return
     */
    virtual ALGErrCode image_uniformization(std::vector<cv::Mat> input, Image_uniformization *malloc_image_ptr);

    /**
     * 图像预处理释放
     * @param malloc_image_ptr
     * @return
     */
    virtual ALGErrCode image_uniformization_free(Image_uniformization *malloc_image_ptr);
    /**
     *
     * @param malloc_image_ptr
     * @param result
     * @return
     */
     virtual ALGErrCode forward(Image_uniformization *malloc_image_ptr,
                         cv::Mat &result,cv::Mat &result_color
                        ,bool verbose = false);

    /**
     * 获取模型版本号
     * @return
     */
    const std::string getVersion() const;


    /**
     * image_uniformization中的图片预处理函数，默认操作仅仅只有resize
     * @param src
     * @param dst
     * @param size
     */
     virtual void image_preprocess(const cv::Mat &src,cv::Mat &dst,cv::Size size) ;

    /**
     *  结果还原函数，将模型输出坐标信息还原成原始图片
     * @param boxes
     * @param current_dim_w
     * @param current_dim_h
     * @param orig_h
     * @param orig_w
     */
    virtual void reduction_boxes(float *boxes,const int current_dim_w,const int current_dim_h,const int orig_h,const int orig_w) ;

    virtual ALGErrCode do_pre_forward();

    // 释放申请的显存
    void release_cuda_mem(void *detections_d,void *boxes_d,void *scores_d,void *classes_d);

    // 释放申请的内存
    void release_cpu_mem(int *detections,float *boxes,float *scores,float *classes);


    // Queue for the Msgs
    int FLAGS_Detect_size_H;
    int FLAGS_Detect_size_W;
    int FLAGS_Detect_size_C;
    // 网络输入图像尺寸 ，如果不清楚请咨询算法开发人员
    Engine detector;

    size_t MAX_BATCHSIZE;


    bool isInitialized;

    // 输入个数
    int inputCount;
    // 输出个数
    int outputCount;


    std::vector<std::vector<int64_t>> inputs_size;
    std::vector<int> inputs_numel;
    std::vector<int> inputs_type_size;
    std::vector<nvinfer1::DataType> inputs_type;

    std::vector<std::vector<int64_t>> outputs_size;
    std::vector<int> outputs_numel;
    std::vector<int> outputs_type_size;
    std::vector<nvinfer1::DataType> outputs_type;

    std::vector<int64_t> num_detections_size;
    std::vector<int64_t> nmsed_boxes_size;
    std::vector<int64_t> nmsed_scores_size;
    std::vector<int64_t> nmsed_classes_size;
    std::vector<cv::Scalar> color_list ;
    int num_detections;
    int num_boxes ;
    int num_scores ;
    int num_classes ;


    /***************** Mat转vector **********************/
    template<typename _Tp>
    std::vector<_Tp> convertMat2Vector(const cv::Mat &mat)
    {
        return (std::vector<_Tp>)(mat.reshape(1, 1));//通道数不变，按行转为一行
    }
public:
    std::string version;

};

}
#endif // DETECTOR_SDK_H
