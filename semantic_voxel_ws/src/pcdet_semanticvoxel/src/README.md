## Deps

Xavier NX(JetPack 4.4.1 (L4T R32.4.4) pytorch1.6)
参考：https://forums.developer.nvidia.com/t/jetpack-4-4-l4t-r32-4-3-production-release/140870
主要是：
CUDA 10.2
cuDNN 8.0.0
TensorRT 7.1.3

## 模型

/model
第一个模型 trace: pp_vfe_pfn.pt
第二个模型 onnx backbone.trt
第三个模型 trace before_nms_script.pt

将onnx模型序列化为tensorrt模型,若当前显卡支持fp16运算则添加`--fp16`命令

```sh
cd $Tensorrt_Root/bin
./trtexec --onnx=backbone.onnx --fp16 --saveEngine=backbone.trt
```

trace.pt模型无需转换

## CMakeList修改

将CMakeList中依赖路径替换为你自己的路径

```sh
line:12
line:35
```

添加自己的gpu_arch，查看自己的gpu_arch请参考https://developer.nvidia.com/cuda-gpus#compute

```sh
line:29
```

## 代码修改

修改main.cpp中：`bin_root`（点云数据的根目录，应保证该目录下只有点云bin文件）

`result_root`结果保存目录

修改line:58, 61, 71的模型路径，也就是我提到的trace，onnx

## 编译&运行

```sh
mkdir debug && cd debug
cmake .. && make -j4
./demo
```

