#ifndef ALGORITHM_SDK_ERROR_CODE_H
#define ALGORITHM_SDK_ERROR_CODE_H
namespace ATHENA_algorithm {
    typedef enum
    {
        ALGORITHM_LIB_LICENSE_UNLAWFULNESS    = 0,    /* 无效 */
        ALGORITHM_LIB_LICENSE_PROBATION       = 1,    /* 试用 */
        ALGORITHM_LIB_LICENSE_TARGET_DATE     = 2,    /* 一段时间内有效 */
        ALGORITHM_LIB_LICENSE_LIFETIME        = 3,    /* 终身许可 */
        ALGORITHM_LIB_LICENSE_END
    }Algorithm_Lib_License_mode;


    /*error code*/
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


        ALGORITHM_INIT_FIRMWARE_HEAD_ERR				= (  11 ),		/* 固件头错误 */
        ALGORITHM_FIRMWARE_VERSION_MAIN_ERR			    = (  12 ),		/* 加密固件软件主版本不一致 */
        ALGORITHM_FIRMWARE_VERSION_MID_ERR			    = (  13 ),		/* 加密固件软件中版本不一致 */
        ALGORITHM_FIRMWARE_VERSION_MAINMAIN_ERR		    = (  14 ),		/* 加密固件软件低版本不一致 */
        ALGORITHM_FIRMWARE_FILE_LENGTH_ERR			    = (  15 ),		/*  */
        ALGORITHM_FIRMWARE_FILE_ENC_CHECK_ERR           = (  16 ),		/*  */
        ALGORITHM_FIRMWARE_FILE_SRC_CHECK_ERR           = (  17 ),		/*  */
        ALGORITHM_FIRMWARE_FILE_TYPE_ERR				= (  18 ),		/*  */
        ALGORITHM_FIRMWARE_FILE_SAVE_ERR				= (  19 ),		/*  */
        ALGORITHM_ERROR_FUNC_PARAM_LAW                  = (  20 ),		/*  */
        ALGORITHM_NEW_ALGORITHM_ENCRYPTION_FAIL		    = (  21 ),		/*  */
        ALGORITHM_UNAUTHORZED				            = (  22 ),		/* 未授权 */
        ALGORITHM_KEY_OVER_PROABTION					= (  23 ),		/* 试用时间过期 */
        ALGORITHM_KEY_OVERDUE                           = (  24 ),		/* 秘钥过期 */
        ALGORITHM_LICENSE_FRMAT_ILL					    = (  30 ),		/* 秘钥格式非法 */
        ALGORITHM_LICENSE_CPU_CHECK					    = (  31 ),		/* 芯片ID验证失败 */
        ALGORITHM_FIRMWARE_AL_VERSION_HEAD	            = (  32 ),		/*  无法获取算法模型的版本号*/
        ALGORITHM_INIT_FIRMWARE_NOT_EXIT				= (  33 ),		/*  */
        ALGORITHM_FIRMWARE_FILE_COUNT_ERR				= (  34 ),		/*  */
        ALGORITHM_KEY_VERIFY_FAIL						= (  35 ),		/*  */
        ALGORITHM_ERROR_FUNC_FILE_READ                  = (  36 ),		/* 无法读取固件 */
        ALGORITHM_OPEN_CHIP_ID_ERR                      = (  37 ),		/* 获取芯片ID失败 */
        ALGORITHM_PARAM_ERR                             = (  38 ),      /* 算法参数错误 */
        ALGORITHM_CUDA_RUNTIME_ERR                      = (  39 ),      /*CUDA运行环境异常*/
        ALGORITHM_MODEL_INFERENCE_ERR                   = (  40 ),      /*模型推理异常*/
        ALGORITHM_BATCHSIZE_ERR                         = (  41 ),      /*错误的Batchsize*/
        ALGORITHM_GPU_DEVICE_ID_ERR                     = (  42 ),      /*错误的GPU 设备ID*/
        ALGORITHM_INPUT_FORMAT_ERR                      = (  43 ),      /*输入格式错误*/
        ALGORITHM_CUDA_MEMORY_ERR                       = (  44 ),       /*显存错误*/
        ALGORITHM_FORWARD_OVERFIT_MAXTHREADS_ERR       = (  45 ),       /*Forward 推理设置的线程编号超过了初始化设置的最大线程数*/
        ALGORITHM_ONNXRUNTIME_REE                       = ( 100 ),  /*onnxruntime错误*/
        /***************** 算法类别错误码 **********************/
        ALGORITHM_JIEBA_INIT_ERR                        = (  200 ),      /*切词模型初始化错误*/
        ALGORITHM_CORRECTOR_INIT_ERR                    = (  201 ),      /*纠错模型初始化错误*/
        ALGORITHM_TASKBOT_INIT_ERR                      = (  202 ),      /*任务模型初始化错误*/
        ALGORITHM_FAQBOT_INIT_ERR                       = (  203 ),      /*问答模型初始化错误*/
        ALGORITHM_ERR_OTHER                             =  999
    } ALGErrCode;

}
#endif // ALGORITHM_SDK_ERROR_CODE_H
