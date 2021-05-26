1、T4服务器ip
root@10.168.101.28:/opt/workspace/xiaoying/tensorrt_inference/unet/
2、docker环境准备
docker pull nvcr.io/nvidia/tensorrt:21.02-py3
nvidia-docker run --gpus all -v /opt:/opt --network host -itd --name ying_tensorrt_21.02-py3 nvcr.io/nvidia/tensorrt:2b3c0461174a4094d18bd0c79d9763af52fe35ee94a29e406348d07aac25ab63e
docker exec -it ying_tensorrt_21.02-py3 /bin/bash
3、命令行推理过程
cd /workspace/tensorrt/bin
./trtexec --onnx=/opt/workspace/xiaoying/tensorrt_inference/unet.onnx --shapes=\'input.1\':1x3x256x256 --saveEngine=/opt/workspace/xiaoying/tensorrt_inference/unet-1_3_256_256_fp32_dynamic.engine
./trtexec --onnx=/opt/workspace/xiaoying/tensorrt_inference/unet.onnx --shapes=\'input.1\':1x3x256x256 --saveEngine=/opt/workspace/xiaoying/tensorrt_inference/unet-1_3_256_256_int8_dynamic.engine --int8
./trtexec --onnx=/opt/workspace/xiaoying/tensorrt_inference/unet.onnx --shapes=\'input.1\':1x3x256x256 --saveEngine=/opt/workspace/xiaoying/tensorrt_inference/unet-1_3_256_256_fp16_dynamic.engine --fp16
4、脚本推理过程
bash /opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/generate_engine.sh /opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/unet.onnx /opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/result/unet_result.txt 128
bash /opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/generate_log.sh
5、Description
trtexec脚本中的参数：
--onnx：导出的onnx模型文件位置
--minShapes：最小输入tensor，NxCxWxH方式，N代表batchsize, C表示channel，W表示tensor的weight，H表示tensor的height，具体表示方式根据onnx模型的input数据格式更改
--optShapes：TensorRT根据此输入进行最佳优化，格式同上
--maxShapes：允许输入最大tensor，格式同上，如果想改变模型允许的最大batchsize，可以调整N的值，但是由于gpu memory限制，过大的N值会导致outofmemery
--saveEngine：编译后的推理引擎名称
--fp16 or --int8：模型推理精度，只可以选择一个进行编译，如果是空，则为fp32
--exportProfile=<file>：Write the profile information per layer in a json file (default = disabled)
--exportTimes=<file>: Write the timing results in a json file (default = disabled)
--streams：如果要测试throughput，需要设置multi-streaming
  其他参数可以根据trtexec -h查看
  如果想测试某个batchsize的profile表现，设置minShapes optShapes和maxShapes为同一值，在test.sh脚本中，batch size设置为同一个值