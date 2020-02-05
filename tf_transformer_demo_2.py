import tensorflow as tf
import numpy as np
import os

import tensorflow.contrib.slim as slim
from tensorflow.python.client import timeline
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu import utils
from tensorflow.python import ipu

import modeling


class Dataset:
    def __init__(self, l):
        self.data = []
        for i in range(0, l):
            self.data.append((np.random.randint(0, 1, [64, 128]), np.random.randint(0, 1, [64, 128])))
    def __getitem__(self, k):
        return self.data[k]
    def __iter__(self):
        return (self[j] for j in range(len(self.data)))
    def __len__(self):
        return len(self.data)

cfg = ipu.utils.create_ipu_config(profiling=True, use_poplar_text_report=True)
cfg = ipu.utils.auto_select_ipus(cfg, 1)
ipu.utils.configure_ipu_system(cfg)

pathname = "pretrained_model/cased_L-12_H-768_A-12/bert_model.ckpt"
bert_config = modeling.BertConfig.from_json_file("pretrained_model/cased_L-12_H-768_A-12/bert_config.json")
#configsession = tf.ConfigProto()
#configsession.gpu_options.allow_growth = True
#sess = tf.Session(config=configsession)
input_ids = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="input_ids")
input_mask = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="input_mask")
token_type_ids = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="token_type_ids")

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def transformer():
    model = modeling.BertModel(
        config=bert_config,
        is_training=True,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=token_type_ids,
        use_one_hot_embeddings=False)
    label_embeddings = tf.get_variable(name="word_embeddings", shape=[768, 12],
                                           initializer=tf.truncated_normal_initializer(0.02))
    pooled_output = model.get_pooled_output()
    logits = tf.matmul(pooled_output, label_embeddings)

def run_sess():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #print(sess.run(logits, feed_dict={input_ids: rand_array, input_mask: rand_array, token_type_ids: rand_array}))

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        data = Dataset(100)
        for rand_array in data:
            print(sess.run(transformer, feed_dict={input_ids: rand_array, input_mask: rand_array, token_type_ids: rand_array},
                           options=run_options, run_metadata=run_metadata))

#        rand_array = np.random.randint(0, 1, [64, 128])

        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)

with ipu_scope("/device:IPU:0"):
    transformer_out = ipu.ipu_compiler.compile(run_sess())

if __name__ == "__main__":
    print('tf-bert-transformer')
    print(run_sess())
    print(transformer_out)
    model_summary()

