#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='2'
layer_num=2
python ./build_copy_transformer.py \
                   -data /home/LAB/zhangzy/ProjectData/openNMT/knowledgeData \
                   -save_model /home/LAB/zhangzy/ProjectModels/knowledgeComprehension/model-L$layer_num \
                   -layers $layer_num \
                   -rnn_size 768 \
                   -word_vec_size 768 \
                   -max_grad_norm 0 \
                   -optim adam \
                   -seed 13711 \
                   -position_encoding \
                   -dropout 0\.2 \
                   -param_init 0 \
                   -warmup_steps 8000 \
                   -learning_rate 1e-5 \
                   -decay_method noam \
                   -label_smoothing 0.1 \
                   -adam_beta2 0.998 \
                   -batch_size 6 \
                   -batch_type sents \
                   -normalization tokens \
                   -max_generator_batches 2 \
                   -train_steps 100000 \
                   -valid_steps 500 \
                   -save_checkpoint_steps 2000 \
                   -keep_checkpoint 10 \
                   -report_every 50 \
                   -accum_count 8 \
                   -copy_attn \
                   -param_init_glorot \
                   -world_size 1 \
                   -gpu_ranks 0 \
                   #-share_embeddings \
