#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'
layers=1
enc_dec_type=transformer
python ~/ProgrammingAlpha/OpenNMT-py/train.py -data /home/LAB/zhangzy/ProjectData/openNMT/knowledgeData \
                   -save_model /home/LAB/zhangzy/ProjectModels/${enc_dec_type}-${layers} \
                   -layers $layers \
                   -rnn_size 768 \
                   -word_vec_size 768 \
                   -max_grad_norm 0 \
                   -optim adam \
                   -encoder_type transformer \
                   -decoder_type transformer \
                   -position_encoding \
                   -dropout 0\.2 \
                   -param_init 0 \
                   -warmup_steps 8000 \
                   -learning_rate 0\.0001 \
                   -decay_method noam \
                   -label_smoothing 0.1 \
                   -adam_beta2 0.998 \
                   -batch_size 32 \
                   -batch_type sents \
                   -normalization sents \
                   -max_generator_batches 128 \
                   -train_steps 200000 \
                   -valid_steps 1000 \
                   -save_checkpoint_steps 1000 \
                   -keep_checkpoint 5 \
                   -accum_count 4 \
                   -share_embeddings \
                   -copy_attn \
                   -param_init_glorot \
                   -pre_word_vecs_enc /home/LAB/zhangzy/ShareModels/Embeddings/bert/embeddings.pt \
                   -pre_word_vecs_dec /home/LAB/zhangzy/ShareModels/Embeddings/bert/embeddings.pt \
                   -log_file ${enc_dec_type}-${layers}.log \
                   -world_size 4 \
                   -gpu_ranks 0 1 2 3 \
                   #-train_from /home/LAB/zhangzy/ProjectModels/
