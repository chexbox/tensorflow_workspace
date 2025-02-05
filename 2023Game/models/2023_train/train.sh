#!/bin/bash

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/targets/x86_64-linux/lib/stubs
MODEL_DIR=/home/ubuntu/tensorflow_workspace/2023Game/models/2023_train
PIPELINE_CONFIG_PATH=$MODEL_DIR/ssd_mobilenet_v2_512x512_coco.config
NUM_TRAIN_STEPS=300000
SAMPLE_1_OF_N_EVAL_EXAMPLES=100
python3 /home/ubuntu/tensorflow_workspace/2023Game/models/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
