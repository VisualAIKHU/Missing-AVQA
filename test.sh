#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python net_grd_avst/main_avst.py --mode test --test_path net_grd_avst/avst_models/memory/best_audio_missing.pt --missing_situation audio

#CUDA_VISIBLE_DEVICES=0 python net_grd_avst/main_avst.py --mode test --test_path net_grd_avst/avst_models/memory/best_visual_missing.pt --missing_situation visual
# CUDA_VISIBLE_DEVICES=3 python net_grd_avst/main_avst.py --mode test --test_path net_grd_avst/avst_models/time_step20/weight_epoch_50.pt --missing_situation visual --log_file visual_noise_timestep20.txt --time_step 20
# CUDA_VISIBLE_DEVICES=3 python net_grd_avst/main_avst.py --mode test --test_path net_grd_avst/avst_models/time_step20/weight_epoch_50.pt --missing_situation audio --log_file audio_noise_timestep20.txt --time_step 20
CUDA_VISIBLE_DEVICES=0 python net_grd_avst/main_avst.py --mode test --test_path net_grd_avst/avst_models/time_step5/weight_epoch_50.pt --missing_situation both --log_file inference_time.txt --time_step 5

#CUDA_VISIBLE_DEVICES=0 python net_grd_avst/main_avst.py --mode test --test_path /home/kyuri0924/shared_code/MUSIC-AVQA_memory_use_psuudo/net_grd_avst/avst_models/memory/best_visual_missing.pt --missing_situation visual
