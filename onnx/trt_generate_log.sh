#!/bin/bash

for bs in 1 2 4 8 16 32 64
do
  ./trtexec --loadEngine=/opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/engine/unet_256x256_b"$bs"_fp16_static.engine --shapes=input:"$bs"x3x256x256 --dumpProfile --exportProfile=/opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/result/unet_bs"$bs"_fp16_profile.json > /opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/result/unet_bs"$bs"_fp16_profile_log.txt
  ./trtexec --loadEngine=/opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/engine/unet_256x256_b"$bs"_int8_static.engine --shapes=input:"$bs"x3x256x256 --dumpProfile --exportProfile=/opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/result/unet_bs"$bs"_int8_profile.json > /opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/result/unet_bs"$bs"_int8_profile_log.txt
  ./trtexec --loadEngine=/opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/engine/unet_256x256_b"$bs"_fp32_static.engine --shapes=input:"$bs"x3x256x256 --dumpProfile --exportProfile=/opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/result/unet_bs"$bs"_fp32_profile.json > /opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/result/unet_bs"$bs"_fp32_profile_log.txt

  ./trtexec --loadEngine=/opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/engine/unet_256x256_b"$bs"_fp16_static.engine --shapes=input:"$bs"x3x256x256 --exportTimes=/opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/result/unet_bs"$bs"_fp16_trace.json > /opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/result/unet_bs"$bs"_fp16_trace_log.txt
  ./trtexec --loadEngine=/opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/engine/unet_256x256_b"$bs"_int8_static.engine --shapes=input:"$bs"x3x256x256 --exportTimes=/opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/result/unet_bs"$bs"_int8_trace.json > /opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/result/unet_bs"$bs"_int8_trace_log.txt
  ./trtexec --loadEngine=/opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/engine/unet_256x256_b"$bs"_fp32_static.engine --shapes=input:"$bs"x3x256x256 --exportTimes=/opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/result/unet_bs"$bs"_fp32_trace.json > /opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/result/unet_bs"$bs"_fp32_trace_log.txt
done
