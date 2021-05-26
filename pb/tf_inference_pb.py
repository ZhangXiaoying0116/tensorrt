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
with open("./Joint_PER300_POST64.pb", "rb") as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")
    with tf.Session() as sess:
        ## print tensorflow graph node name
        tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]

        sess.run(tf.global_variables_initializer())
        input_image_tensor = sess.graph.get_tensor_by_name('x_input:0')
        output_tensor_boxes = sess.graph.get_tensor_by_name('import/output/labels:0')
        output_tensor_scores = sess.graph.get_tensor_by_name('import/output/boxes:0')
        output_tensor_labels = sess.graph.get_tensor_by_name('import/output/scores:0')
        output= sess.run([output_tensor_boxes,output_tensor_scores,output_tensor_labels], feed_dict={input_image_tensor: img})

        ## calculate times
        time_start =time.time()
        for i in range(1000):
            output= sess.run([output_tensor_boxes,output_tensor_scores,output_tensor_labels], feed_dict={input_image_tensor: img})
        time_end =time.time()
        print('Inference fps:',1/((time_end-time_start)/1000))