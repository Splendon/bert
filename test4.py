import numpy as np
import tensorflow as tf
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import utils
from tensorflow.python import ipu


class Dataset:
        def __init__(self,l):
                self.data = []
                for i in range(0,l):
                    self.data.append((np.random.random([50,200]),np.random.random([50,10])))
        def __getitem__(self,k):
                return self.data[k]
        def __iter__(self):
                return (self[j] for j in range(len(self.data)))
        def __len__(self):
                return len(self.data)

def graph(x,y):
        W = tf.get_variable(initializer=lambda:tf.zeros([200,10],dtype=tf.float16),name="W")
        b = tf.get_variable(initializer=lambda:tf.zeros([10],dtype=tf.float16),name="b")
        pred = tf.nn.softmax(tf.matmul(x,W,name="matmul") + b, name = "pred")

        cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices = 1), name = "cost") 
        optimizer = tf.train.GradientDescentOptimizer(0.01)

        return cost,W,b,optimizer.minimize(cost)



with ipu_scope("/device:IPU:0"):
    x = tf.placeholder(tf.float16,[50,200],name='x')
    y = tf.placeholder(tf.float16,[50,10],name='y')

    #cost,update = ipu.ipu_compiler.compile(graph,[x,y])
    batch = ipu.ipu_compiler.compile(graph,[x,y])



opts = utils.create_ipu_config()
cfg = utils.auto_select_ipus(opts,1)
ipu.utils.configure_ipu_system(cfg)


data = Dataset(100)
with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for train_x,train_y in data:
            c,w,b = sess.run(batch,feed_dict = {x: train_x,y : train_y})
        print (c)
        print (w)
        print (b)
