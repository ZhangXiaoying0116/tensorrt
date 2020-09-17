import numpy as np
import tensorflow as tf
import time
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.contrib import tensorrt as trt 
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.framework import graph_util
import os

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

graph_def = tf.GraphDef()
img = np.random.rand(1,3,512,512)
# print(" !!! xiaoying | img.shape:",img.shape)

# !!! xiaoying | open pb file for inference
with tf.gfile.GFile("./Joint_PER300_POST64.pb", 'rb') as f: 
    graph_def.ParseFromString(f.read())
    converted_graph_def = trt.create_inference_graph(
        input_graph_def = graph_def,
        max_batch_size =1,
        is_dynamic_op=False,
        outputs=['import/output/labels:0', 'import/output/boxes:0',"import/output/scores:0"])
    output_node = tf.import_graph_def(converted_graph_def,return_elements = ['import/output/labels:0', 'import/output/boxes:0',"import/output/scores:0"])

with tf.Session() as sess:
    # !!! xiaoying | save tensorflow-tensorrt graph
    # output_combined_trtf = graph_util.convert_variables_to_constants(sess, sess.graph_def,['import/import/output/labels', 'import/import/output/boxes',"import/import/output/scores"])
    # tf.train.write_graph(output_combined_trtf, '/lwj/xiaoying.zhang/', 'trtf_padnostack_transpose_new.pb', as_text=False)
    
    # !!! xiaoying | print tensorflow-tensorrt graph node name
    tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    # print("0:",tensor_name_list[0])
    # print("1:",tensor_name_list[1])

    input_image_tensor = sess.graph.get_tensor_by_name('import/x_input:0')
    print(" !!! xiaoying | ",input_image_tensor)
    sess.run(tf.global_variables_initializer())
    sess.run(output_node,feed_dict={input_image_tensor: img}) # !!! xiaoying | notice first step

    # !!! xiaoying | calculate times
    time_start =time.time() 
    for i in range(1000):
        sess.run(output_node,feed_dict={input_image_tensor: img})
    time_end =time.time()
    print(' !!! xiaoying | Inference fps ',1/((time_end-time_start)/1000))