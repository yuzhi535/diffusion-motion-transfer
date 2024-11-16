#!/bin/zsh

# conda
eval "$(conda shell.zsh hook)"
conda activate dmt

echo "device: 'cuda'
seed: null
output_path: 'results/outputs'
data_path: 'data/inputs'
latents_path: 'data/ddim_latents'
source_prompt: 'Amazing quality, masterpiece, A flower vase is sitting on a porch stand.'
target_prompt: 'Amazing quality, masterpiece, A squirrel admires a flower vase on a porch stand.'
negative_prompt: 'bad quality, distortions, unrealistic, distorted image, watermark, signature'
guidance_scale: 10

with_lr_decay: True
optim_lr: 0.01
scale_range: [0.007 , 0.004]
optimization_steps: 10


max_frames: 24
n_timesteps: 50
max_guidance_timestep: 1
min_guidance_timestep: 0.6
features_loss_weight: 0
global_averaging: True
features_diff_loss_weight: 1

restart_sampling: True
random_init: False
high_freq_replacement_init: True
downsample_factor: 4

use_upsampler_features: True
use_temporal_attention_features: True
use_spatial_attention_features: True
use_conv_features: True
use_temp_conv_features: True
up_res_dict: {1: [1]}
guidance_before_res: True
" > configs/preprocess_config.yaml

echo "video_path: data/inputs
save_dir: data/ddim_latents
max_number_of_frames: 24

n_timesteps: 999
prompt: Amazing quality, masterpiece, A locomotive rides in a forest
negative_prompt: ''
save_ddim_reconstruction: False" > configs/preprocess_config.yaml

python preprocess_video_ddim.py --config_path configs/preprocess_config.yaml
python run.py --config_path configs/guidance_config.yaml