import tensorflow as tf
import numpy as np
import os

import modeling

pathname = "pretrained_model/cased_L-12_H-768_A-12/bert_model.ckpt"
bert_config = modeling.BertConfig.from_json_file("pretrained_model/cased_L-12_H-768_A-12/bert_config.json")
#configsession = tf.ConfigProto()
#configsession.gpu_options.allow_growth = True
#sess = tf.Session(config=configsession)
input_ids = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="input_ids")
input_mask = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="input_mask")
token_type_ids = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="token_type_ids")

with tf.Session() as sess:
    model = modeling.BertModel(
        config=bert_config,
        is_training=True,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=token_type_ids,
        use_one_hot_embeddings=False)
    label_embeddings = tf.get_variable(name="word_embeddings", shape=[768, 12], initializer=tf.truncated_normal_initializer(0.02))
    pooled_output = model.get_pooled_output()
    logits = tf.matmul(pooled_output, label_embeddings)

    sess.run(tf.global_variables_initializer())
    print('tf-bert-transformer')
    rand_array = np.random.randint(0, 1, [64, 128])
    print(sess.run(logits, feed_dict = {input_ids:rand_array, input_mask:rand_array, token_type_ids: rand_array}))