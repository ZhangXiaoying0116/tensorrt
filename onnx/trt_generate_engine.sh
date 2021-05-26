for i in {0..100}
do
        if [[ $((2**i)) -gt $3 ]]
        then
            break
        else
            echo "BatchSize $((2**i))" >> $2
            ./trtexec --onnx=$1 --minShapes=\'input.1\':$((2**i))x3x256x256 --optShapes=\'input.1\':$((2**i))x3x256x256  --maxShapes=\'input.1\':$((2**i))x3x256x256  --saveEngine=/opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/engine/unet_256x256_b$((2**i))_fp32_static.engine >> $2
            ./trtexec --onnx=$1 --minShapes=\'input.1\':$((2**i))x3x256x256 --optShapes=\'input.1\':$((2**i))x3x256x256  --maxShapes=\'input.1\':$((2**i))x3x256x256  --saveEngine=/opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/engine/unet_256x256_b$((2**i))_fp16_static.engine --fp16 >> $2
            ./trtexec --onnx=$1 --minShapes=\'input.1\':$((2**i))x3x256x256 --optShapes=\'input.1\':$((2**i))x3x256x256  --maxShapes=\'input.1\':$((2**i))x3x256x256  --saveEngine=/opt/workspace/xiaoying/tensorrt_inference/pytorch-Unet/engine/unet_256x256_b$((2**i))_int8_static.engine --int8 >> $2
        fi
done
