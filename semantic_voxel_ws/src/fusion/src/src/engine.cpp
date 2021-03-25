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
using namespace nvinfer1;
using namespace nvonnxparser;


static bool isPluginCreatorRegistration = false;

class Logger : public ILogger {
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

ALGErrCode Engine::_load(const string &path) {
    ifstream file(path, ios::in | ios::binary);
    if (!file)
    {
        return ALGORITHM_MODEL_PATH_ERR;
    }
    file.seekg (0, file.end);
    size_t size = file.tellg();
    file.seekg (0, file.beg);

    char *buffer = new char[size];
    file.read(buffer, size);
    if (!file)
    {
        return ALGORITHM_MODEL_ERR;
    }
    file.close();
    try {
        _engine = _runtime->deserializeCudaEngine(buffer, size, nullptr);
        delete[] buffer;
        if(_engine == nullptr){
            return ALGORITHM_MODEL_ERR;
        }
    }
    catch(...){
        delete[] buffer;
        return ALGORITHM_MODEL_ERR;
    }

    return ALGORITHM_OPERATION_SUCCESS;
}

ALGErrCode Engine::_load_buffer(const std::string buffer) {
    const char * buf=buffer.c_str();
    size_t size = buffer.size();
    if(size == 0){
        return ALGORITHM_MODEL_ERR;
    }
    try {
        _engine = _runtime->deserializeCudaEngine(buf, size, nullptr);
        if(_engine == nullptr){
            return ALGORITHM_MODEL_ERR;
        }
    }
    catch(...){
        return ALGORITHM_MODEL_ERR;
    }
    return ALGORITHM_OPERATION_SUCCESS;
}


ALGErrCode Engine::initEngine(const std::string path,int deviceID , bool verbose) {
    int countGpus=0;
    cudaGetDeviceCount(&countGpus);
    if(deviceID > countGpus -1){
        return ALGORITHM_GPU_DEVICE_ID_ERR;
    }
    cudaSetDevice(deviceID);
    Logger logger(verbose);
    if(isPluginCreatorRegistration == false){
        initLibNvInferPlugins(&logger, "");
#ifdef USING_CUSTOM_TENSORRT
        initLibNvInferCustomPlugins(&logger, "");
#endif
        isPluginCreatorRegistration = true;
    }

    _runtime = createInferRuntime(logger);
    if(_runtime == nullptr){
        return ALGORITHM_CUDA_RUNTIME_ERR;
    }
    ALGErrCode ret = _load(path);
    if(ret != ALGORITHM_OPERATION_SUCCESS){
        return ret;
    }
    ret = _prepare();
    return ret;
}

ALGErrCode Engine::initEngineWithBuffer(const string buffer,int deviceID , bool verbose) {
    int countGpus=0;
    cudaGetDeviceCount(&countGpus);
    if(deviceID > countGpus -1){
        return ALGORITHM_GPU_DEVICE_ID_ERR;
    }
    cudaSetDevice(deviceID);
    Logger logger(verbose);
    if(isPluginCreatorRegistration == false){
        initLibNvInferPlugins(&logger, "");
#ifdef USING_CUSTOM_TENSORRT
        initLibNvInferCustomPlugins(&logger, "");
#endif
        isPluginCreatorRegistration = true;
    }
    _runtime = createInferRuntime(logger);
    ALGErrCode ret = _load_buffer(buffer);
    if(ret != ALGORITHM_OPERATION_SUCCESS){
        return ret;
    }
    ret = _prepare();
    return ret;
}

ALGErrCode Engine::_prepare() {
    _context = _engine->createExecutionContext();
    if (!_context)
    {
        return ALGORITHM_CUDA_RUNTIME_ERR;
    }
    cudaError_t ret = cudaStreamCreate(&_stream);
    if(ret != cudaSuccess){
        return ALGORITHM_CUDA_RUNTIME_ERR;
    }
    return ALGORITHM_OPERATION_SUCCESS;
}

Engine::~Engine() {
    if (_stream) cudaStreamDestroy(_stream);
    // 必须注释掉这两个,不然多线程程序推出时会有bug
    // if (_context) _context->destroy();
    // if (_engine) _engine->destroy();
    if (_runtime) _runtime->destroy();
}

void Engine::save(const string &path) {
    cout << "Writing to " << path << "..." << endl;
    auto serialized = _engine->serialize();
    ofstream file(path, ios::out | ios::binary);
    file.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());

    serialized->destroy();
}

ALGErrCode Engine::infer(vector<void *> &buffers, int batch) {
    bool ok = _context->enqueue(batch, buffers.data(), _stream, nullptr);
    if(ok == false){
        return ALGORITHM_MODEL_INFERENCE_ERR;
    }
    cudaError_t ret  = cudaStreamSynchronize(_stream);

    if(ret != cudaSuccess){
        std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;
        return ALGORITHM_CUDA_RUNTIME_ERR;
    }
    return ALGORITHM_OPERATION_SUCCESS;
}

vector<int> Engine::getInputSize() {
    auto dims = _engine->getBindingDimensions(0);
    return {dims.d[1], dims.d[2]};
}

int Engine::getMaxBatchSize() {
    return _engine->getMaxBatchSize();
}

int Engine::getMaxDetections() {
    return _engine->getBindingDimensions(1).d[0];
}

int Engine::getStride() {
    return 1;
}

vector<vector<int64_t>>  Engine::getInputSize(int inputNum) {

    vector<vector<int64_t>> shapes;
    for (int i = 0; i < inputNum; ++i) {
        auto dims = _engine->getBindingDimensions(i);
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
    for (int b = inputNum; b < _engine->getNbBindings(); ++b) {

        const char *name_ = _engine->getBindingName(b);
        auto dims = _engine->getBindingDimensions(_engine->getBindingIndex(name_));
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
    for (int b = inputNum; b < _engine->getNbBindings(); ++b) {

        const char *name_ = _engine->getBindingName(b);

        //            shape{dims.d[0], dims.d[1],dims.d[2]};
        shapes.push_back(name_);
    }
    return shapes;
}




int totalCount(std::vector<int64_t> in, const char * name){
    int total = 1;
    //            std::cout<<name<<":";
    for (int i = 0; i < in.size(); ++i) {
        //                std::cout<<","<<in[i];
        if(in[i] != 0){
            total *=in[i];
        }
    }
    //            std::cout<<std::endl;
    return total;
}

