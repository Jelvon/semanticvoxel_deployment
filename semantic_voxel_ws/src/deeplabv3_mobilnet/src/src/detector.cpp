#include "detector.h"
#include "engine.h"
#include <string.h>
#include <fstream>




namespace ATHENA_algorithm {


ALGErrCode Detector_SDK::init(const char * modelPath,
                                   const int maxBatchSize,
                                   int deviceID, int max_threads)
{
    try {

        std::string model_path_str = modelPath;
        ATHENA_algorithm::ALGErrCode ret_code  = detector.initEngine(modelPath, deviceID,max_threads);
        if(ret_code != ALGORITHM_OPERATION_SUCCESS){
            return ret_code;
        }

        if(maxBatchSize > detector.getMaxBatchSize() || maxBatchSize==0)
        {
            std::cerr<<"Error maxBatchSize must be in 1~"<<detector.getMaxBatchSize()<<",if you want to use dynamic batch or multibatch,just set maxBatchSize to "<<detector.getMaxBatchSize()<<std::endl;
            return ALGORITHM_INIT_ERR;
        }
        MAX_BATCHSIZE = maxBatchSize;
        inputs_size = detector.getInputSize(inputCount);
        auto inputSize = inputs_size[0];
        // 根据模型获取输入参数
        FLAGS_Detect_size_W = inputSize[2];
        FLAGS_Detect_size_H = inputSize[1];
        FLAGS_Detect_size_C = inputSize[3];
        // Copy image to device
        // Run inference n times
        outputs_size =detector.getOutputSize(inputCount);
//        if(outputs_size.size() != 4){
//            return ALGORITHM_MODEL_ERR;
//        }
        if(outputs_size.size() != outputCount){
            return ALGORITHM_MODEL_ERR;
        }
        auto names = detector.getOutputNames(inputCount);
//        num_detections_size = outputs_size[0];
//        nmsed_boxes_size = outputs_size[1];
//        nmsed_scores_size = outputs_size[2];
//        nmsed_classes_size = outputs_size[3];
//        // Get back the bounding boxes
//        num_detections = totalCount(num_detections_size,names[0]);
//        num_boxes = totalCount(nmsed_boxes_size,names[1]);
//        num_scores = totalCount(nmsed_scores_size,names[2]);
//        num_classes = totalCount(nmsed_classes_size,names[3]);
        //获取 所有的输出 元素个数 以及 元素类型
        outputs_numel.clear();
        for(int i = 0; i < names.size(); ++i){
            int output_numel = totalCount(outputs_size[i], names[i]);
            outputs_numel.push_back(output_numel);
        }
        auto type_and_size = detector.getOutputTypeSize(inputCount);
        outputs_type_size = std::get<0>(type_and_size);
        outputs_type = std::get<1>(type_and_size);

        isInitialized = true;
        try {
            ALGErrCode iRet = do_pre_forward();
            return iRet;
        }
        catch (...) {
            return ALGORITHM_INIT_ERR;
        }
    }
    catch (...) {
        return ALGORITHM_INIT_ERR;
    }
}


/**
 *  图像预处理，主要是做resize
 * @param input
 * @param outputs
 * @return
 */
ALGErrCode Detector_SDK::image_uniformization(std::vector<cv::Mat> input, Image_uniformization *malloc_image_ptr) {

    try {

        if (true != isInitialized) {
            std::cerr << "Detector model uninitialized,please use init function first！！！！" << std::endl;
            return ALGORITHM_INIT_MODEL_ERR;
        }
        if(input.size() > MAX_BATCHSIZE){
            image_uniformization_free(malloc_image_ptr);
            return ALGORITHM_BATCHSIZE_ERR;
        } else if (input.size() == 0) {
            image_uniformization_free(malloc_image_ptr);
            return ALGORITHM_INPUT_FORMAT_ERR;
        }

        if(nullptr == malloc_image_ptr)
        {
            image_uniformization_free(malloc_image_ptr);
            return ALGORITHM_PARAM_ERR;
        }
        else
        {
            malloc_image_ptr->src_image_height.clear();
            malloc_image_ptr->src_image_width.clear();
            malloc_image_ptr->channel = FLAGS_Detect_size_C;
            malloc_image_ptr->width = FLAGS_Detect_size_W;
            malloc_image_ptr->height = FLAGS_Detect_size_H;
            if(malloc_image_ptr->image_data_d == nullptr){
                if(cudaSuccess != cudaMalloc(&malloc_image_ptr->image_data_d,  input.size() * FLAGS_Detect_size_W * FLAGS_Detect_size_H * FLAGS_Detect_size_C* sizeof(float)))
                {
//                    if(malloc_image_ptr->image_data_d)
//                    {
//                        cudaFree(malloc_image_ptr->image_data_d);
//                    }
                    image_uniformization_free(malloc_image_ptr);
                    return ALGORITHM_CUDA_MEMORY_ERR;
                }

            }else{
                if(input.size() != malloc_image_ptr->batch){
                    image_uniformization_free(malloc_image_ptr);
                    if(cudaSuccess != cudaMalloc(&malloc_image_ptr->image_data_d,  input.size() * FLAGS_Detect_size_W * FLAGS_Detect_size_H * FLAGS_Detect_size_C* sizeof(float)))
                    {
                        if(malloc_image_ptr->image_data_d)
                        {
                            cudaFree(malloc_image_ptr->image_data_d);
                            malloc_image_ptr->image_data_d = nullptr;
                        }
                        return ALGORITHM_CUDA_MEMORY_ERR;
                    }

                }
            }
            if( nullptr == malloc_image_ptr->image_data_d )
            {
                image_uniformization_free(malloc_image_ptr);
                return ALGORITHM_ERROR_FUNC_MEM_ENOUGH;
            }

        }
        vector<float> image_data;
        for (int i = 0; i < input.size(); ++i) {

            cv::Mat dst;
            cv::Mat dstColor;

            if(input[i].channels() != FLAGS_Detect_size_C){
                if(FLAGS_Detect_size_C ==1)
                {
                    cv::cvtColor(input[i], dstColor, cv::COLOR_BGR2GRAY);
                }
                else if(FLAGS_Detect_size_C ==3)
                {
                    cv::cvtColor(input[i], dstColor, cv::COLOR_GRAY2BGR);
                }
            }else{
                dstColor = input[i].clone();
            }
            image_preprocess(dstColor,dst,cv::Size( FLAGS_Detect_size_W, FLAGS_Detect_size_H ));
            malloc_image_ptr->src_image_height.push_back(input[i].rows);
            malloc_image_ptr->src_image_width.push_back(input[i].cols);

            if(FLAGS_Detect_size_C == 1)
            {
                dstColor.convertTo(dstColor, CV_32FC1, 1.0 , 0);
            }
            else if(FLAGS_Detect_size_C == 3)
            {
                dstColor.convertTo(dstColor, CV_32FC3, 1.0 , 0);
            }
            vector<float> data = convertMat2Vector<float>(dst);
            // Copy image to device
            image_data.insert(image_data.end(), data.begin(), data.end());
        }
        malloc_image_ptr->batch = input.size();
        size_t dataSize = image_data.size() * sizeof(float);
        if(cudaSuccess != cudaMemcpy(malloc_image_ptr->image_data_d, image_data.data(), dataSize, cudaMemcpyHostToDevice))
        {
            image_uniformization_free(malloc_image_ptr);
            return ALGORITHM_CUDA_MEMORY_ERR;
        }


//        }
        if(malloc_image_ptr->device_data.size() == 0){
            for(int i = 0; i < outputs_numel.size(); ++i){
                void *device_data = nullptr;
                void *host_data = nullptr;
                host_data = malloc(malloc_image_ptr->batch * outputs_numel[i] * outputs_type_size[i]);
                if(cudaSuccess != cudaMalloc(&device_data, malloc_image_ptr->batch * outputs_numel[i] * outputs_type_size[i]))
                {
                    if(device_data)
                    {
                        cudaFree(device_data);
                        device_data = nullptr;
                    }
                    image_uniformization_free(malloc_image_ptr);
                    return ALGORITHM_CUDA_MEMORY_ERR;
                }
                malloc_image_ptr->host_data.push_back(host_data);
                malloc_image_ptr->device_data.push_back(device_data);
            }
        }
        return ALGORITHM_OPERATION_SUCCESS;
    }
    catch (...) {
//        if(malloc_image_ptr->image_data_d)
//        {
//            cudaFree(malloc_image_ptr->image_data_d);
//        }
        image_uniformization_free(malloc_image_ptr);
        return ALGORITHM_IMAGE_UNIFORMAZATION_ERR;
    }
}

ALGErrCode Detector_SDK::image_uniformization_free(Image_uniformization *malloc_image_ptr){
    if(malloc_image_ptr)
    {
        if(malloc_image_ptr->image_data_d){
            cudaFree(malloc_image_ptr->image_data_d );
            malloc_image_ptr->image_data_d = nullptr;
        }
        for(int i = 0; i < malloc_image_ptr->device_data.size(); ++i)
        {
            if(malloc_image_ptr->device_data[i])
            {
                cudaFree(malloc_image_ptr->device_data[i]);
                malloc_image_ptr->device_data[i] = nullptr;
            }
            if(malloc_image_ptr->host_data[i])
            {
                free(malloc_image_ptr->host_data[i]);
                malloc_image_ptr->host_data[i] = nullptr;
            }
        }
    }
    return ALGORITHM_OPERATION_SUCCESS;
}




/**
 * 模型预测代码，支持多张图片输入
 * @param candidate_imgs
 * @param forward_infoes
 * @param confidence_thresh
 * @return
 */
ALGErrCode Detector_SDK::forward(Image_uniformization *malloc_image_ptr,
                                 cv::Mat &result,cv::Mat &result_color,bool verbose) {
    if (true != isInitialized) {
        std::cerr << "Detector model uninitialized,please use init function first！！！！" << std::endl;
        return ALGORITHM_INIT_MODEL_ERR;
    }
    if(malloc_image_ptr->batch > MAX_BATCHSIZE){
        return ALGORITHM_BATCHSIZE_ERR;
    }
    void *detections_d = nullptr;
    int *detections = nullptr;


    try {
        auto startTime = std::chrono::high_resolution_clock::now();

        detections_d = malloc_image_ptr->device_data[0];

        detections = (int *)malloc_image_ptr->host_data[0];

        auto start = chrono::steady_clock::now();
        // very important,需要将所有申请的内存放在buffer里面，顺序为input+output，且与getBindingDimensions需要的顺序一致
        vector<void *> buffers = {malloc_image_ptr->image_data_d, detections_d};

        ALGErrCode ret = detector.infer(buffers, malloc_image_ptr->batch);
        if(ret != ALGORITHM_OPERATION_SUCCESS)
        {
            image_uniformization_free(malloc_image_ptr);
            return ret;
        }
        if(verbose){
            auto stop = chrono::steady_clock::now();
            auto timing = chrono::duration_cast<chrono::duration<double>>(stop - start);
            std::cout << "Segmentor_SDK inference Took " << timing.count()  << " seconds per inference." << endl;
        }
        if(cudaSuccess != cudaMemcpy(detections, detections_d, malloc_image_ptr->batch * outputs_numel[0] * outputs_type_size[0], cudaMemcpyDeviceToHost))
        {

            image_uniformization_free(malloc_image_ptr);
            return ALGORITHM_CUDA_MEMORY_ERR;
        }

        // 取结果
        for (int j = 0; j < malloc_image_ptr->batch; ++j) {

            cv::Mat mask = cv::Mat::ones(cv::Size(FLAGS_Detect_size_W, FLAGS_Detect_size_H),
                                         CV_8UC1);
            cv::Mat mask_color = cv::Mat::ones(cv::Size(FLAGS_Detect_size_W, FLAGS_Detect_size_H),
                                               CV_8UC3);
            for (int row = 0; row < mask.rows; ++row){
                for (int col = 0; col < mask.cols; ++col){
                    unsigned char value = (unsigned char)detections[row*mask.cols + col];
                    if( value>= color_list.size()){
                        value = color_list.size() -1;
                    }
//                    mask.at<uchar>(row, col) = value;
                    unsigned char *b = mask.ptr<unsigned char>(row, col);
                    * b = value;
                    cv::Vec3b *c = mask_color.ptr<cv::Vec3b>(row, col);
                    c->val[0] = color_list[value][0];        //B
                    c->val[1] = color_list[value][1];        //G
                    c->val[2] = color_list[value][2];        //R

                }
            }

//            cv::Mat resized= cv::Mat::ones(cv::Size(malloc_image_ptr->src_image_width[j],malloc_image_ptr->src_image_height[j]),
//                                           CV_8UC1);
//            cv::Mat resized_color= cv::Mat::ones(cv::Size(malloc_image_ptr->src_image_width[j],malloc_image_ptr->src_image_height[j]),
//                                          CV_8UC3);;
            cv::resize( mask, result, cv::Size(malloc_image_ptr->src_image_width[j],malloc_image_ptr->src_image_height[j]), (0.0), (0.0), cv::INTER_LINEAR );
            cv::resize( mask_color, result_color, cv::Size(malloc_image_ptr->src_image_width[j],malloc_image_ptr->src_image_height[j]), (0.0), (0.0), cv::INTER_LINEAR );
//            result = resized.clone();
//            result_color = mask_color.clone();
        }

        return ALGORITHM_OPERATION_SUCCESS;
    }
    catch (std::exception e) {
        image_uniformization_free(malloc_image_ptr);
        std::cout << e.what() << std::endl;
        return ALGORITHM_POSTPROCESS_FAILE;
    }
}


void  Detector_SDK::image_preprocess(const cv::Mat &src,cv::Mat &dst,cv::Size size) {
    cv::resize( src, dst, size, (0.0), (0.0), cv::INTER_LINEAR );
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
}

void Detector_SDK::reduction_boxes(float *boxes,const int current_dim_w,const int current_dim_h,const int orig_h,const int orig_w){
    boxes[0] = boxes[0] *orig_w;
    boxes[1] = boxes[1] *orig_h;
    boxes[2] = boxes[2] *orig_w;
    boxes[3] = boxes[3] *orig_h;
}

/*
 *  初始化时，做一次完整的测试。虽然会增加耗时，但是可以增加程序的稳定性。
 */
ALGErrCode Detector_SDK::do_pre_forward() {
    std::vector<cv::Mat> inputs;
    cv::Mat img = cv::Mat::ones(cv::Size(FLAGS_Detect_size_W, FLAGS_Detect_size_H),
                                CV_8UC3);;
    Image_uniformization mImage;
    for (int i = 0; i < MAX_BATCHSIZE; ++i) {
        inputs.push_back(img);
    }
    ALGErrCode iRet = image_uniformization(inputs,&mImage);
    if(iRet != ALGORITHM_OPERATION_SUCCESS){
        isInitialized = false;
        return ALGORITHM_INIT_ERR;
    }
    cv::Mat result,result_color;
    iRet = forward(&mImage,result,result_color);
    if(iRet != ALGORITHM_OPERATION_SUCCESS){
        isInitialized = false;
        return ALGORITHM_INIT_ERR;
    }
    iRet = image_uniformization_free(&mImage);
    if(iRet != ALGORITHM_OPERATION_SUCCESS){
        isInitialized = false;
        return ALGORITHM_INIT_ERR;
    }
    return ALGORITHM_OPERATION_SUCCESS;
}



//release 下会出现未定义的引用
///***************** Mat转vector **********************/
//template<typename _Tp>
//std::vector<_Tp> Detector_SDK::convertMat2Vector(const cv::Mat &mat)
//{
//    return (std::vector<_Tp>)(mat.reshape(1, 1));//通道数不变，按行转为一行
//}


    std::vector<cv::Scalar> get_color_list(){
        std::vector<cv::Scalar> color_list = {
                cv::Scalar(128, 64, 128),
                cv::Scalar(244, 35, 232),
                cv::Scalar(70, 70, 70),
                cv::Scalar(102, 102, 156),
                cv::Scalar(190, 153, 153),
                cv::Scalar(153, 153, 153),
                cv::Scalar(250, 170, 30),
                cv::Scalar(220, 220, 0),
                cv::Scalar(107, 142, 35),
                cv::Scalar(152, 251, 152),
                cv::Scalar(70, 130, 180),
                cv::Scalar(220, 20, 60),
                cv::Scalar(255, 0, 0),
                cv::Scalar(0, 0, 142),
                cv::Scalar(0, 0, 70),
                cv::Scalar(0, 60, 100),
                cv::Scalar(0, 80, 100),
                cv::Scalar(0, 0, 230),
                cv::Scalar(119, 11, 32)
        };
        return color_list;
    }

}
