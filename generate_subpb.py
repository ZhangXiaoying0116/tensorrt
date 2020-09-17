import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

graph_def = tf.GraphDef()
img = np.random.rand(1,3,512,512)
# print(" !!! xiaoying | img.shape:",img.shape)

with tf.gfile.GFile("./Joint_PER300_POST64.pb", 'rb') as f:
    graph_def.ParseFromString(f.read())
    x_input = tf.placeholder(tf.float32, [1, 3, 512, 512], name = 'x_input')
    tf.import_graph_def(graph_def,input_map={'transpose:0':x_input},return_elements = ['output/labels:0', 'output/boxes:0',"output/scores:0"])
with tf.Session() as sess:
    # !!! xiaoying | print tensorflow graph node name
    # tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    output_pb = graph_util.convert_variables_to_constants(sess, sess.graph_def,['import/output/labels', 'import/output/boxes',"import/output/scores"])
    tf.train.write_graph(output_pb, './', 'Joint_PER300_POST64_new.pb', as_text=False)