/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "engine.h"

#include <iostream>
#include <fstream>
#include <cuda_runtime_api.h>
#include <NvOnnxConfig.h>
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include "mutex"
//#define USING_CUSTOM_TENSORRT
#ifdef USING_CUSTOM_TENSORRT
    #include "NvInferCustomPlugin.h"
#endif

using namespace nvonnxparser;



namespace ATHENA_algorithm {

    static bool isPluginCreatorRegistration = false;
    std::mutex mtx;
    class Logger : public nvinfer1::ILogger {
    public:
        Logger(bool verbose)
                : _verbose(verbose) {
        }

        void log(Severity severity, const char *msg) override {
            if (_verbose || (severity != Severity::kINFO))
                cout << msg << endl;
        }

    private:
        bool _verbose{false};
    };

    ATHENA_algorithm::ALGErrCode Engine::_load(const string &path) {
        ifstream file(path, ios::in | ios::binary);
        if (!file)
        {
            return ATHENA_algorithm::ALGORITHM_MODEL_PATH_ERR;
        }
        file.seekg (0, file.end);
        size_t size = file.tellg();
        file.seekg (0, file.beg);

        char *buffer = new char[size];
        file.read(buffer, size);
        if (!file)
        {
            return ATHENA_algorithm::ALGORITHM_MODEL_ERR;
        }
        file.close();
        try {
//            _engine = _runtime->deserializeCudaEngine(buffer, size, nullptr);
//            if(_engine == nullptr){
//                return ATHENA_algorithm::ALGORITHM_MODEL_ERR;
//            }
//            delete[] buffer;
//            if(_engine == nullptr){
//                return ATHENA_algorithm::ALGORITHM_MODEL_ERR;
//            }
            for(int i = 0; i < _max_threads; ++i){
                nvinfer1::ICudaEngine *_engine = _runtime->deserializeCudaEngine(buffer, size, nullptr);
                if(_engine == nullptr){
                    return ATHENA_algorithm::ALGORITHM_MODEL_ERR;
                }
                _engines.push_back(_engine);
                //获取第一个输入的尺寸，如果batch 为 -1 则为动态batch
                auto input0_dims = _engine->getBindingDimensions(0);
                if(input0_dims.d[0] == -1)
                {
                    isdynamic_batch = true;
                }
                else
                {
                    //固定batch的情况下只需要反序列化一次 engine即可
                    isdynamic_batch = false;
                    break;
                }
            }
            delete[] buffer;
        }
        catch(...){
            delete[] buffer;
            return ATHENA_algorithm::ALGORITHM_MODEL_ERR;
        }

        return ATHENA_algorithm::ALGORITHM_OPERATION_SUCCESS;
    }

    ATHENA_algorithm::ALGErrCode Engine::_load_buffer(const std::string buffer) {
        const char * buf=buffer.c_str();
        size_t size = buffer.size();
        if(size == 0){
            return ATHENA_algorithm::ALGORITHM_MODEL_ERR;
        }
        try {
//            _engine = _runtime->deserializeCudaEngine(buf, size, nullptr);
//            if(_engine == nullptr){
//                return ATHENA_algorithm::ALGORITHM_MODEL_ERR;
//            }
            for(int i = 0; i < _max_threads; ++i){
                nvinfer1::ICudaEngine *_engine = _runtime->deserializeCudaEngine(buf, size, nullptr);
                if(_engine == nullptr){
                    return ATHENA_algorithm::ALGORITHM_MODEL_ERR;
                }
                _engines.push_back(_engine);
                //获取第一个输入的尺寸，如果batch 为 -1 则为动态batch
                auto input0_dims = _engine->getBindingDimensions(0);
                if(input0_dims.d[0] == -1)
                {
                    isdynamic_batch = true;
                }
                else
                {
                    //固定batch的情况下只需要反序列化一次 engine即可
                    isdynamic_batch = false;
                    break;
                }
            }
        }
        catch(...){
            return ATHENA_algorithm::ALGORITHM_MODEL_ERR;
        }
        return ATHENA_algorithm::ALGORITHM_OPERATION_SUCCESS;
    }


    ATHENA_algorithm::ALGErrCode Engine::initEngine(const std::string path, int deviceID , int max_threads, bool verbose) {
        int countGpus=0;
        cudaGetDeviceCount(&countGpus);
        if(deviceID > countGpus -1){
            return ALGORITHM_GPU_DEVICE_ID_ERR;
        }
        cudaSetDevice(deviceID);
        Logger logger(verbose);
        while (mtx.try_lock()==false);
        if(isPluginCreatorRegistration == false){
            initLibNvInferPlugins(&logger, "");
            isPluginCreatorRegistration = true;
        }

        _runtime = nvinfer1::createInferRuntime(logger);
        if(_runtime == nullptr){
            mtx.unlock();
            return ATHENA_algorithm::ALGORITHM_CUDA_RUNTIME_ERR;
        }
        _max_threads = max_threads;
        ATHENA_algorithm::ALGErrCode ret = _load(path);
        if(ret != ATHENA_algorithm::ALGORITHM_OPERATION_SUCCESS){
            mtx.unlock();
            return ret;
        }
        ret = _prepare();
        mtx.unlock();
        return ret;
    }

    ATHENA_algorithm::ALGErrCode Engine::initEngineWithBuffer(const string buffer, int deviceID , int max_threads, bool verbose) {
        int countGpus=0;
        cudaGetDeviceCount(&countGpus);
        if(deviceID > countGpus -1){
            return ALGORITHM_GPU_DEVICE_ID_ERR;
        }
        cudaSetDevice(deviceID);
        Logger logger(verbose);
        while (mtx.try_lock()==false);
        if(isPluginCreatorRegistration == false){
            initLibNvInferPlugins(&logger, "");

            isPluginCreatorRegistration = true;
        }
        _runtime = nvinfer1::createInferRuntime(logger);
        if(_runtime == nullptr){
            mtx.unlock();
            return ATHENA_algorithm::ALGORITHM_CUDA_RUNTIME_ERR;
        }
        _max_threads = max_threads;
        ATHENA_algorithm::ALGErrCode ret = _load_buffer(buffer);
        if(ret != ATHENA_algorithm::ALGORITHM_OPERATION_SUCCESS){
            mtx.unlock();
            return ret;
        }
        ret = _prepare();
        mtx.unlock();
        return ret;
    }

    ATHENA_algorithm::ALGErrCode Engine::_prepare() {
//        _context = _engine->createExecutionContext();
//        if (!_context)
//        {
//            return ATHENA_algorithm::ALGORITHM_CUDA_RUNTIME_ERR;
//        }
        for(int i = 0; i < _max_threads; ++i){
            if(isdynamic_batch){
                _context = _engines[i]->createExecutionContext();
                if (!_context)
                {
                    return ATHENA_algorithm::ALGORITHM_CUDA_RUNTIME_ERR;
                }
                _contexts_map[_context] = true;
            }
            else{
                _context = _engines[0]->createExecutionContext();
                if (!_context)
                {
                    return ATHENA_algorithm::ALGORITHM_CUDA_RUNTIME_ERR;
                }
                _contexts_map[_context] = true;
            }
        }
//        cudaError_t ret = cudaStreamCreate(&_stream);
//        if(ret != cudaSuccess){
//            return ATHENA_algorithm::ALGORITHM_CUDA_RUNTIME_ERR;
//        }
        return ATHENA_algorithm::ALGORITHM_OPERATION_SUCCESS;
    }

    Engine::~Engine() {
//        if (_stream) cudaStreamDestroy(_stream);
        while(mtx.try_lock()==false);
        // 必须注释掉这两个,不然多线程程序推出时会有bug
//         if (_context) _context->destroy();
//         if (_engine) _engine->destroy();

//         for(int i = 0; i < _engines.size(); ++i){
//             if (_engines[i])
//                 _engines[i]->destroy();
//         }
         std::map<nvinfer1::IExecutionContext *,bool>::iterator it;
         for(it = _contexts_map.begin(); it != _contexts_map.end(); it++){
             if(it->first){
                 it->first->destroy();
             }
         }
         for(int i = 0; i < _engines.size(); ++i){
             if(_engines[i]){
                  _engines[i]->destroy();
             }
         }
         mtx.unlock();
        if (_runtime) _runtime->destroy();
    }

    void Engine::save(const string &path) {
        cout << "Writing to " << path << "..." << endl;
//        auto serialized = _engine->serialize();
//        ofstream file(path, ios::out | ios::binary);
//        file.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
//        serialized->destroy();
        for(int i = 0; i < _engines.size(); ++i){
            auto serialized = _engines[i]->serialize();
            ofstream file(path, ios::out | ios::binary);
            file.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
            serialized->destroy();
        }
    }

    ATHENA_algorithm::ALGErrCode Engine::infer(vector<void *> &buffers, int batch) {
//        bool ok = _context->enqueue(batch, buffers.data(), _stream, nullptr); //error
//        bool ok = _context->execute(batch, buffers.data()); //error
//        bool ok = _context->enqueueV2(buffers.data(), _stream, nullptr); //success

//        std::chrono::steady_clock::time_point time_start = std::chrono::steady_clock::now();

//        auto input0 = _context->getBindingDimensions(0);
//        if(input0.d[0] != batch){
//            input0.d[0] = batch;
//            _context->setBindingDimensions(0,input0);
////            _context->setBindingDimensions(0,Dims4{batch,input0.d[1],input0.d[2],input0.d[3]}); //也可以使用Dims4来设置输入尺寸
//            if(!_context->allInputDimensionsSpecified())
//            {
//                std::cout<<"set input dims failed"<<std::endl;
//            }
//        }
////        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
////        std::cout<<"set batch time: "<<std::chrono::duration<double, std::milli>(time_end - time_start).count()<<" ms"<<std::endl;
//        bool ok = _context->executeV2(buffers.data());

        int threads = 0;
        std::map<nvinfer1::IExecutionContext *,bool>::iterator it;
        pthread_rwlock_rdlock(&flock);//上读锁
        for(it = _contexts_map.begin(); it != _contexts_map.end(); it++)
        {
            threads ++;
//            std::cout<<"i :" << threads << std::endl;
            if(it->second == true)
            {
                break;
            }
            if(threads == _max_threads)
            {
                return ALGORITHM_FORWARD_OVERFIT_MAXTHREADS_ERR;
            }
        }
        pthread_rwlock_unlock(&flock);//解读锁
//        free_thread_id_mutex.lock();
        pthread_rwlock_wrlock(&flock);//上写锁
        it->second = false;
        pthread_rwlock_unlock(&flock);//解写锁
//        free_thread_id_mutex.unlock();
        auto input0 = it->first->getBindingDimensions(0);
        if(input0.d[0] != batch){
            input0.d[0] = batch;
            it->first->setBindingDimensions(0,input0);
//            _context->setBindingDimensions(0,Dims4{batch,input0.d[1],input0.d[2],input0.d[3]}); //也可以使用Dims4来设置输入尺寸
            if(!it->first->allInputDimensionsSpecified())
            {
                std::cout<<"set input dims failed"<<std::endl;
            }
        }
        bool ok = it->first->executeV2(buffers.data());
//        bool ok = it->first->enqueueV2(buffers.data(), _stream, nullptr);
//        free_thread_id_mutex.lock();
        pthread_rwlock_wrlock(&flock);//上写锁
        it->second = true;
        pthread_rwlock_unlock(&flock);//解写锁
//        free_thread_id_mutex.unlock();

        if(ok == false){
            return ATHENA_algorithm::ALGORITHM_MODEL_INFERENCE_ERR;
        }

//        cudaError_t ret  = cudaStreamSynchronize(_stream);

//        if(ret != cudaSuccess){
//            return ATHENA_algorithm::ALGORITHM_CUDA_RUNTIME_ERR;
//        }
        return ATHENA_algorithm::ALGORITHM_OPERATION_SUCCESS;
    }



    vector<int> Engine::getInputSize() {
//        auto dims = _engine->getBindingDimensions(0);
        auto dims = _engines[0]->getBindingDimensions(0);
        return {dims.d[1], dims.d[2]};
    }

    //tensorrt7 获取最大batch
    int Engine::getMaxBatchSize() {
        //因为tensorrt7支持动态shape,因此在转换模型时会将shape设置到profile中,通过getProfileDimensions获取需要的shape
//        auto input_max = _engine->getProfileDimensions(0,0,nvinfer1::OptProfileSelector::kMAX);
        auto input_max = _engines[0]->getProfileDimensions(0,0,nvinfer1::OptProfileSelector::kMAX);
        return input_max.d[0];
    }

    //tensorrt7 获取最小batch
    int Engine::getMinBatchSize() {
//        auto input_min = _engine->getProfileDimensions(0,0,nvinfer1::OptProfileSelector::kMIN);
        auto input_min = _engines[0]->getProfileDimensions(0,0,nvinfer1::OptProfileSelector::kMIN);
        return input_min.d[0];
    }

    int Engine::getMaxDetections() {
//        return _engine->getBindingDimensions(1).d[0];
        return _engines[0]->getBindingDimensions(1).d[0];
    }

    int Engine::getStride() {
        return 1;
    }

    vector<vector<int64_t>>  Engine::getInputSize(int inputNum) {

        vector<vector<int64_t>> shapes;
        for (int i = 0; i < inputNum; ++i) {
//            auto dims = _engine->getBindingDimensions(i);
            auto dims = _engines[0]->getBindingDimensions(i);
            vector<int64_t> shape;
            for(int j = 0; j < dims.nbDims; ++j){
                shape.push_back(dims.d[j]);
            }
            //            vector<int> shape{dims.d[0], dims.d[1],dims.d[2], dims.d[3]};
            shapes.push_back(shape);
        }

        return shapes;
    }

    vector<vector<int64_t>> Engine::getOutputSize( int inputNum ) {
        vector<vector<int64_t>> shapes;
//        for (int b = inputNum; b < _engine->getNbBindings(); ++b) {

//            const char *name_ = _engine->getBindingName(b);
//            auto dims = _engine->getBindingDimensions(_engine->getBindingIndex(name_));
//            vector<int64_t> shape;
//            for(int j = 0; j < dims.nbDims; ++j){
//                shape.push_back(dims.d[j]);
//            }
////            shape{dims.d[0], dims.d[1],dims.d[2]};
//            shapes.push_back(shape);
//        }

        for (int b = inputNum; b < _engines[0]->getNbBindings(); ++b) {

            const char *name_ = _engines[0]->getBindingName(b);
            auto dims = _engines[0]->getBindingDimensions(_engines[0]->getBindingIndex(name_));
            vector<int64_t> shape;
            for(int j = 0; j < dims.nbDims; ++j){
                shape.push_back(dims.d[j]);
            }
//            shape{dims.d[0], dims.d[1],dims.d[2]};
            shapes.push_back(shape);
        }
        return shapes;
    }

    vector<const char *> Engine::getOutputNames( int inputNum ) {
        vector<const char *> shapes;
//        for (int b = inputNum; b < _engine->getNbBindings(); ++b) {

//            const char *name_ = _engine->getBindingName(b);

////            shape{dims.d[0], dims.d[1],dims.d[2]};
//            shapes.push_back(name_);
//        }

        for (int b = inputNum; b < _engines[0]->getNbBindings(); ++b) {

            const char *name_ = _engines[0]->getBindingName(b);

//            shape{dims.d[0], dims.d[1],dims.d[2]};
            shapes.push_back(name_);
        }
        return shapes;
    }

    tuple<vector<int>,vector<nvinfer1::DataType>> Engine::getInputTypeSize(int inputNum){
        vector<int> sizes;
        vector<nvinfer1::DataType> datatype_vec;
        for (int b = 0; b < inputNum; ++b) {
            int type_size;
            nvinfer1::DataType datatype = _engines[0]->getBindingDataType(b);
            switch (datatype) {
                case nvinfer1::DataType::kINT32:
                {
                    type_size = 4;
                    break;
                }
                case nvinfer1::DataType::kFLOAT:
                {
                    type_size = 4;
                    break;
                }
                case nvinfer1::DataType::kHALF:
                {
                    type_size = 2;
                    break;
                }
                case nvinfer1::DataType::kINT8:
                {
                    type_size = 1;
                    break;
                }
            }
            sizes.push_back(type_size);
        }
        tuple<vector<int>,vector<nvinfer1::DataType>> type_and_size(sizes,datatype_vec);
        return type_and_size;
    }

    tuple<vector<int>,vector<nvinfer1::DataType>> Engine::getOutputTypeSize(int inputNum){
        vector<int> sizes;
        vector<nvinfer1::DataType> datatype_vec;
        for (int b = inputNum; b < _engines[0]->getNbBindings(); ++b) {
            int type_size;
            nvinfer1::DataType datatype = _engines[0]->getBindingDataType(b);
            switch (datatype) {
                case nvinfer1::DataType::kINT32:
                {
                    type_size = 4;
                    break;
                }
                case nvinfer1::DataType::kFLOAT:
                {
                    type_size = 4;
                    break;
                }
                case nvinfer1::DataType::kHALF:
                {
                    type_size = 2;
                    break;
                }
                case nvinfer1::DataType::kINT8:
                {
                    type_size = 1;
                    break;
                }
            }
            sizes.push_back(type_size);
            datatype_vec.push_back(datatype);
        }
        tuple<vector<int>,vector<nvinfer1::DataType>> type_and_size(sizes,datatype_vec);
        return type_and_size;
    }

//因为tensorrt7 增加了batch维度 ,所以从1维开始计算tensor的元素个数
    int totalCount(std::vector<int64_t> in, const char * name){
        int total = 1;
//            std::cout<<name<<":";
        //i = 0; ==> i = 1;
        for (int i = 1; i < in.size(); ++i) {
//                std::cout<<","<<in[i];
            if(in[i] != 0){
                total *=in[i];
            }
        }
//            std::cout<<std::endl;
        return total;
    }

}
