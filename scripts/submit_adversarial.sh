#!/bin/bash

sourcedomain="D2"
targetdomain="D3"
rgbdatapath="data/rgb"
flowdatapath="data/flow"
rgbpretrained="kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt"
flowpretrained="kinetics-i3d/data/checkpoints/flow_imagenet/model.ckpt"

python src/train/train.py  --train=True --results_path="EPIC_"$sourcedomain"_"$targetdomain"_adversarial_only_pretrain"  \
                           --restore_model_rgb="$rgbpretrained" --restore_model_flow="$flowpretrained" \
                           --rgb_data_path="$rgbdatapath" --flow_data_path="$flowdatapath" \
                           --datasets="Annotations/$sourcedomain" --unseen_dataset="Annotations/$targetdomain"\
                           --lr=0.01 --num_gpus=8 --batch_size=128 --max_steps=3001  --modality="joint" --num_labels=8  \
                           --steps_before_update 1 --restore_mode="pretrain" --domain_mode="PretrainM"

python src/train/train.py  --train=True --results_path="EPIC_"$sourcedomain"_"$targetdomain"_adversarial_only"   \
                           --restore_model_joint="EPIC_"$sourcedomain"_"$targetdomain"_adversarial_only_pretrain/saved_model_"$sourcedomain"_"$targetdomain"_0.01_0.9/joint/model.ckpt-3000" \
                           --restore_model_rgb="EPIC_"$sourcedomain"_"$targetdomain"_adversarial_only_pretrain/saved_model_"$sourcedomain"_"$targetdomain"_0.01_0.9/rgb/model.ckpt-3000" \
                           --restore_model_flow="EPIC_"$sourcedomain"_"$targetdomain"_adversarial_only_pretrain/saved_model_"$sourcedomain"_"$targetdomain"_0.01_0.9/flow/model.ckpt-3000" \
                           --rgb_data_path="$rgbdatapath" --flow_data_path="$flowdatapath" \
                           --datasets="Annotations/$sourcedomain" --unseen_dataset="Annotations/$targetdomain" \
                           --lr=0.001 --num_gpus=8 --batch_size=128 --max_steps=6001 --modality="joint" --num_labels=8 \
                           --steps_before_update 1 --restore_mode="model" --domain_mode="DANN" --lambda_in=0.2

