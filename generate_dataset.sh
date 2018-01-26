#!/bin/bash

# Generate randomly sampled rotated images and downscale by a factor of 2 to mitigate the effects of interpolation
python3 prepare_data.py --image_folder=$1 --sample_size=$2 --rotate=$3 --scale=2 --save_images --save_dataset

# Resize original images by the same factor of 2 to ensure similarity of content
python3 prepare_data.py --image_folder=$1 --scale=2 --save_images --dataset_csv=dataset.csv

# Split generated images in new dataset into patches and save as an hdf5 archive
python3 prepare_data.py --image_folder=generated --hdf5_path=train.h5
