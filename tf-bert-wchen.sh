
BERT-Base:12-layer, 768-hidden, 12-heads, 110M parameters

scp -r wchen@199.187.246.204:/home/wchen/tf_bert /home/wchen/
scp wchen@199.187.246.182:/home/wchen/tf_bert/log/base_sample.log /Users/chenwenhui/Desktop/tf_bert
scp -r /home/wchen/tf_bert/log chenwenhui@192.168.2.203:/Users/chenwenhui/Desktop/tf_bert/log
scp ipu_estimator_cnn_README.md wchen@199.187.246.182:/home/wchen/tf_bert



find ./ -name 'bert_config*'


1.Pre-training with BERT
python3 create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=./tmp/tf_examples.tfrecord \
  --vocab_file=./pretrained_model/cased_L-12_H-768_A-12/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5

python3 run_pretraining.py \
  --input_file=./tmp/tf_examples.tfrecord \
  --output_dir=./tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=./pretrained_model/cased_L-12_H-768_A-12/bert_config.json \
  --train_batch_size=1 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=2000 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5  2>&1 | tee log/base_sample_2.log 
#  --init_checkpoint=./pretrained_model/cased_L-12_H-768_A-12/bert_model.ckpt \

Details   cat /proc/15831/status   ps -aux | grep 15831  pmap -d 15831
server memory: 75.461G
A.train_batch_size=32
%CPU: 3822
used Memory: 3.244G
B.train_batch_size=1
%CPU: 2490
used Memory: 0.6G


python3 run_pretraining_transformer_cwh.py \
  --input_file=./tmp/tf_examples.tfrecord \
  --output_dir=./tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=./pretrained_model/cased_L-12_H-768_A-12/bert_config.json \
  --train_batch_size=1 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=2000 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5  2>&1 | tee log/transformer_error.log 



