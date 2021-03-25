#ifndef ALGORITHM_SDK_HPP
#define ALGORITHM_SDK_HPP

#include "algorithm_sdk_error_code.h"
#include "common_export.h"
#include <opencv2/core/core.hpp>
#include <vector>
#include <string>
#include <memory>


#ifdef WIN32
#ifdef DLL_EXPORTS
#define ALOGRITHM_EXPORT __declspec(dllexport)
#else
#define ALOGRITHM_EXPORT __declspec(dllimport)
#endif
#else
#define ALOGRITHM_EXPORT
#endif

namespace ATHENA_algorithm {
    /**
     * 获取当前库版本的版本号
     */
    ALOGRITHM_EXPORT const char *GetLibraryVersion();


    /**
     * @brief SMOKE
     * 支持动态多batch, 以下测试batch=1

     */
    class ALOGRITHM_EXPORT Segment_SDK {
    public:
        Segment_SDK();

        ~Segment_SDK();

        /**
        * SDK初始化函数，模型输入采用经加密软件加密过后的模型
        * @param modelPath 模型路径
        * @param maxBatchSize 模型输入最大Batch设置
        * @param encrtyption 模型是否加密
        * @return
        */
        ALGErrCode init(
                const char * trtModelPath,
                const int maxBatchSize = 1,
                const int deviceID = 0);


        /**
         * 图像预处理数据create，由于涉及到内存和显存的释放，所以将内存管理由算法类处理
         * 使用样例：
            MAX_HANDLE mImagePlateRecognition;
            m_detectorPlateRecognition.image_uniformization_create(&mImagePlateRecognition);
            记得调用释放函数image_uniformization_free以实现内存和显存的释放
         * @param malloc_image_ptr　图像数据句柄
         * @return
         */
        ALGErrCode image_uniformization_create(MAX_HANDLE *malloc_image_ptr);

        /**
         * 图像预处理主要是做resize之类的预处理，图像数据从内存到显存，改函数会得malloc_image_ptr指向的值改变，
         * 多线程操作时请勿对同一malloc_image_ptr指针操作
         * @param  input 单张图片
         * @param malloc_image_ptr 图像数据句柄
         * @return
         */
        ALGErrCode image_uniformization(cv::Mat input, MAX_HANDLE malloc_image_ptr);

        /**
         * 图像预处理主要是做resize之类的预处理，图像数据从内存到显存，改函数会得malloc_image_ptr指向的值改变，
         * 多线程操作时请勿对同一malloc_image_ptr指针操作
         * @param inputs 多张图片输入，大小请小于init函数设置的maxBatchSize数
         * @param malloc_image_ptr　图像数据句柄
         * @return
         */
        ALGErrCode image_uniformization(std::vector<cv::Mat> inputs, MAX_HANDLE malloc_image_ptr);

        /**
         * 图像预处理数据释放，由于涉及到内存和显存的释放，所以将内存管理由算法类处理
         * @param malloc_image_ptr　图像数据句柄
         * @return
         */
        ALGErrCode image_uniformization_free(MAX_HANDLE *malloc_image_ptr);

        /**
         * 前向预测，结果获取函数，该函数是线程安全的
         * @param malloc_image_ptr　图像数据句柄
         * @param obj_info　检测结果
         * @param confidence_thresh　置信阈值
         * @param verbose　是否打印相关调试信息
         * @return
         */
        ALGErrCode forward(MAX_HANDLE malloc_image_ptr,
                           cv::Mat result,cv::Mat &result_color,bool verbose);

        /**
         * 当算法库初始化成功后获取模型的输入宽 ,高，通道数
         * @param width 模型的输入宽
         * @param height 模型的输入高
         * @param channels 模型的输入通道数
         * @return 错误码
         */
        ALGErrCode getModelInputShape(int &width, int &height, int &channels);

        /**
         * 获取模型算法库是否初始化成功
         * @return false 未初始化， true 初始化成功
         */
        bool isInitialized() const;

        /**
         * 获取模型版本号
         * @return
         */
        const char *getVersion() const;


    private:
        class Impl;

    public:
        std::unique_ptr<Impl> impl_;


    };


}
#endif //ALGORITHM_SDK_HPP
