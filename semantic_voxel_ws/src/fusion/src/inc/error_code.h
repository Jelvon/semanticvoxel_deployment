#ifndef ERROR_CODE_H
#define ERROR_CODE_H
typedef enum
    {
        ALGORITHM_OPERATION_SUCCESS                      =  0,
        /******************init*****************************/
        ALGORITHM_INIT_ERR                     = (  01 ),
        ALGORITHM_INIT_REPETITION_ERR          = (  02 ),   /*重复初始化模型*/
        ALGORITHM_INIT_MODEL_ERR               = (  03 ),   /*模型加载出错*/
        ALGORITHM_WRONG_IMAGE_ERR              = (  04 ),   /*not support this net*/
        ALGORITHM_MODEL_PATH_ERR               = (  05 ),   /*模型路径错误 */
        ALGORITHM_MODEL_ERR                    = (  06 ),   /*模型错误*/
        ALGORITHM_ERROR_FUNC_MEM_ENOUGH        = (  07 ),
        ALGORITHM_POSTPROCESS_FAILE            = (  8  ),
        /***************** 图片数据准备 **********************/
        ALGORITHM_IMAGE_UNIFORMAZATION_ERR     = (  9  ),
        ALGORITHM_DELETE_ERR         = (  10 ),
        /***********************************************/

        ALGORITHM_CUDA_RUNTIME_ERR                      = (  39 ),      /*CUDA运行环境异常*/
        ALGORITHM_MODEL_INFERENCE_ERR                   = (  40 ),      /*模型推理异常*/
        ALGORITHM_BATCHSIZE_ERR                         = (  41 ),      /*错误的Batchsize*/
        ALGORITHM_GPU_DEVICE_ID_ERR                     = (  42 ),      /*错误的GPU 设备ID*/
        ALGORITHM_INPUT_FORMAT_ERR                      = (  43 ),      /*输入格式错误*/

        /***************** 算法类别错误码 **********************/

        ALGORITHM_ERR_OTHER                             =  999
    } ALGErrCode;
#endif // ERROR_CODE_H
