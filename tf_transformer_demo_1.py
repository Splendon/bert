import tensorflow as tf
import numpy as np

import modeling

# Already been converted into WordPiece token ids
#input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
#input_ids = tf.constant(np.random.randint(1,128, [2, 3]))
input_ids = tf.placeholder(shape=[2, 3], dtype=tf.int32, name= 'input_ids')


#input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
#input_mask = tf.constant(np.random.randint(0,1, [2, 3]))
input_mask = tf.placeholder(shape=[2, 3], dtype=tf.int32, name= 'input_mask')

#token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])
#token_type_ids = tf.constant(np.random.randint(0,2, [2, 3]))
token_type_ids = tf.placeholder(shape=[2, 3], dtype=tf.int32, name= 'token_type_ids')


config = modeling.BertConfig(vocab_size=32000, hidden_size=768,
                             num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

model = modeling.BertModel(config=config, is_training=True,
                           input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

label_embeddings = tf.get_variable(name="word_embeddings", shape=[768, 12], initializer=tf.truncated_normal_initializer(0.02))
pooled_output = model.get_pooled_output()
logits = tf.matmul(pooled_output, label_embeddings)

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    rand_array = np.random.randint(0,1,[2, 3])
    print(sess.run(logits, feed_dict = {input_ids:rand_array, input_mask:rand_array, token_type_ids: rand_array}))
#    print(sess.run(logits))

#print(logits)
#sess= tf.Session()
#sess.run(tf.global_variables_initializer())
#print(sess.run(logits))