#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'
gpu_name='/device:GPU'

python ./build_rl_transformer.py --model_name transformer-rl \
        --run_mode train_and_evaluate  \
        --model_dir /home/LAB/zhangzy/ProjectModels/rlmodel \
        --bert_config /home/LAB/zhangzy/ShareModels/uncased_L-12_H-768_A-12/bert_config.json \
        --bert_ckpt /home/LAB/zhangzy/ShareModels/uncased_L-12_H-768_A-12/bert_model.ckpt \
        --gpu_num 1 \
        --device_name $gpu_name \
        --use_rl \
        #--train_from
