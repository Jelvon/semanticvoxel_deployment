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

#pragma once

#include <string>
#include <vector>
#include "error_code.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace nvinfer1;

// TRT model wrapper around TensorRT CUDA engine
class Engine {
public:

    // Create engine from engine path
    /**
     *
     * @param engine_path
     * @param deviceID 模型初始化选用设备ID，默认为0
     * @param verbose
     * @return
     */
    ALGErrCode initEngine(const string engine_path,int deviceID = 0, bool verbose=false);

    ALGErrCode initEngineWithBuffer(const string buffer,int deviceID = 0, bool verbose=false);

    Engine(){
    }
    ~Engine();

    // Save model to path
    void save(const string &path);

    // Infer using pre-allocated GPU buffers {data, scores, boxes, classes}
    ALGErrCode infer(vector<void *> &buffers, int batch=1);

    // Get (h, w) size of the fixed input
    vector<int> getInputSize();

    // 获取Input 的所有 Size
    vector<vector<int64_t>>  getInputSize(int inputNum);

    // 获取Output 的所有 Size
    vector<vector<int64_t>> getOutputSize( int inputNum );

    vector<const char *> getOutputNames( int inputNum );

    // Get max allowed batch size
    int getMaxBatchSize();

    // Get max number of detections
    int getMaxDetections();

    // Get stride
    int getStride();

private:
    nvinfer1::IRuntime *_runtime = nullptr;
    nvinfer1::ICudaEngine *_engine = nullptr;
    nvinfer1::IExecutionContext *_context = nullptr;
    cudaStream_t _stream = nullptr;

    ALGErrCode _load(const string &path);
    ALGErrCode _load_buffer(const string buffer);
    ALGErrCode _prepare();

};



int totalCount(std::vector<int64_t> in, const char * name);
