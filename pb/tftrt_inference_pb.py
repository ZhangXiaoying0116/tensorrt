import numpy as np
import tensorflow as tf
import time
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.contrib import tensorrt as trt 
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.framework import graph_util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

graph_def = tf.GraphDef()
img = np.random.rand(1,3,512,512)

## open pb file for inference
with tf.gfile.GFile("./Joint_PER300_POST64.pb", 'rb') as f: 
    graph_def.ParseFromString(f.read())
    converted_graph_def = trt.create_inference_graph(
        input_graph_def = graph_def,
        max_batch_size =1,
        is_dynamic_op=False,
        outputs=['import/output/labels:0', 'import/output/boxes:0',"import/output/scores:0"])
    output_node = tf.import_graph_def(converted_graph_def,return_elements = ['import/output/labels:0', 'import/output/boxes:0',"import/output/scores:0"])

with tf.Session() as sess:
    ## print tensorflow-tensorrt graph node name
    tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]

    input_image_tensor = sess.graph.get_tensor_by_name('import/x_input:0')
    sess.run(tf.global_variables_initializer())
    sess.run(output_node,feed_dict={input_image_tensor: img})

    ## calculate times
    time_start =time.time() 
    for i in range(1000):
        sess.run(output_node,feed_dict={input_image_tensor: img})
    time_end =time.time()
    print(' Inference fps ',1/((time_end-time_start)/1000))