#include "algorithm_sdk.h"
#include "detector.h"
#include <chrono>
#include <vector>
#include <numeric>
#include <functional>
#include <math.h>



namespace ATHENA_algorithm {
    class Segment_SDK::Impl : public Detector_SDK {
    public:
        Impl() {
        }

    };


    Segment_SDK::Segment_SDK() : impl_(new Impl) {}

    Segment_SDK::~Segment_SDK() {}

    ALGErrCode Segment_SDK::init(const char * trtModelPath,
                                const int maxBatchSize,
                                const int deviceID
                                ) {
        return impl_->init(trtModelPath, maxBatchSize, deviceID);
    }



    ALGErrCode Segment_SDK::image_uniformization_create(MAX_HANDLE *malloc_image_ptr) {
        Image_uniformization *pImage = new Image_uniformization;

        *malloc_image_ptr = (MAX_HANDLE) pImage;

        return ALGORITHM_OPERATION_SUCCESS;
    }

    ALGErrCode Segment_SDK::image_uniformization(cv::Mat inputs, MAX_HANDLE malloc_image_ptr) {
        if (nullptr == malloc_image_ptr) {
            return ATHENA_algorithm::ALGORITHM_PARAM_ERR;
        }
        Image_uniformization *pImage = (Image_uniformization *) (malloc_image_ptr);

        if (pImage->name != IMAGE_UNIFORMIZATION_NAME) {
            return ATHENA_algorithm::ALGORITHM_WRONG_IMAGE_ERR;
        }
        return impl_->image_uniformization({inputs}, pImage);
    }

    ALGErrCode Segment_SDK::image_uniformization(std::vector<cv::Mat> inputs, MAX_HANDLE malloc_image_ptr) {
        if (nullptr == malloc_image_ptr) {
            return ATHENA_algorithm::ALGORITHM_PARAM_ERR;
        }
        Image_uniformization *pImage = (Image_uniformization *) (malloc_image_ptr);

        if (pImage->name != IMAGE_UNIFORMIZATION_NAME) {
            return ATHENA_algorithm::ALGORITHM_WRONG_IMAGE_ERR;
        }
        return impl_->image_uniformization(inputs, pImage);
    }

    ALGErrCode Segment_SDK::image_uniformization_free(MAX_HANDLE *malloc_image_ptr) {
        if (nullptr == malloc_image_ptr || nullptr == (*malloc_image_ptr)) {
            return ATHENA_algorithm::ALGORITHM_PARAM_ERR;
        }
        Image_uniformization *pImage = (Image_uniformization *) (*malloc_image_ptr);
        if (pImage->name != IMAGE_UNIFORMIZATION_NAME) {
            return ATHENA_algorithm::ALGORITHM_WRONG_IMAGE_ERR;
        }
        ALGErrCode ret_code = ALGORITHM_OPERATION_SUCCESS;
        if (pImage) {
            ret_code = impl_->image_uniformization_free(pImage);
            delete pImage;
        }
        *malloc_image_ptr = nullptr;
        return ret_code;
    }

    ALGErrCode Segment_SDK::forward(MAX_HANDLE malloc_image_ptr,
                                    cv::Mat result,cv::Mat &result_color,bool verbose) {
        if (nullptr == malloc_image_ptr) {
            return ATHENA_algorithm::ALGORITHM_PARAM_ERR;
        }
        Image_uniformization *pImage = (Image_uniformization *) (malloc_image_ptr);

        if (pImage->name != IMAGE_UNIFORMIZATION_NAME) {
            return ATHENA_algorithm::ALGORITHM_WRONG_IMAGE_ERR;
        }

        ALGErrCode ret_code = impl_->forward(pImage,result,result_color,verbose);
        return ret_code;
    }

    const char *Segment_SDK::getVersion() const {
        return impl_->version.c_str();
    }

    ALGErrCode Segment_SDK::getModelInputShape(int &width, int &height, int &channels) {
        if (impl_->isInitialized) {
            width = impl_->FLAGS_Detect_size_W;
            height = impl_->FLAGS_Detect_size_H;
            channels = impl_->FLAGS_Detect_size_C;
        } else {
            return ALGORITHM_INIT_MODEL_ERR;
        }
        return ALGORITHM_OPERATION_SUCCESS;
    }



    bool Segment_SDK::isInitialized() const {
        return impl_->isInitialized;
    }
}
