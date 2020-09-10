#!/bin/bash

sourcedomain="D2"
targetdomain="D3"
rgbdatapath="data/rgb"
flowdatapath="data/flow"
rgbpretrained="kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt"
flowpretrained="kinetics-i3d/data/checkpoints/flow_imagenet/model.ckpt"
experimentdirectory="EPIC_"$sourcedomain"_"$targetdomain"_mmsada"

for modelnum in {5600..6000..50}; do
     python src/train/train.py  --train=False --features=False --results_path="$experimentdirectory"  \
                                --restore_model_rgb="$rgbpretrained" --restore_model_flow="$flowpretrained" \
                                --rgb_data_path="$rgbdatapath" --flow_data_path="$flowdatapath" \
                                --datasets="Annotations/$sourcedomain" --unseen_dataset="Annotations/$targetdomain"\
                                --lr=0.001 --num_gpus=1 --batch_size=64  --modality="joint" --num_labels=8 \
			                          --modelnum=$modelnum
done
