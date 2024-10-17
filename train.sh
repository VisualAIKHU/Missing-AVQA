#!/bin/bash

#CUDA_VISIBLE_DEVICES=2 python net_grd_avst/main_avst.py --batch-size 4 --mode train --checkpoint memory_step3 --step step3 --pretrained_path net_grd_avst/avst_models/memory/best.pt --log_file memory_step3.txt
CUDA_VISIBLE_DEVICES=1 python net_grd_avst/main_avst.py --batch-size 4 --mode train --checkpoint time_step10 --pretrained_path /shared_code/ICASSP_75/net_grd_avst/avst_models/memory/weight_epoch_50.pt --epoch 50 --log_file timestep_10.txt --time_step 10
#CUDA_VISIBLE_DEVICES=4 python net_grd_avst/main_avst.py --batch-size 4 --mode train --checkpoint etc --pretrained_path /shared_code/ICASSP_75/net_grd_avst/avst_models/memory/weight_epoch_50.pt --epoch 50 --log_file etc.txt 
