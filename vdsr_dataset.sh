#!/bin/bash

# Generate images with downsampled sizes of 1, 0.7, and 0.5 factors
python prepare_data.py --image_folder=$1 --sample_size=-1 --save_images --save_dataset

downsize=( 1.4286 2 )
for i in "${downsize[@]}"
do
	python prepare_data.py --image_folder=$1 --scale=$i --save_images --dataset_csv=dataset.csv
done

# Augment with rotated images of 90, 180 and 270
python prepare_data.py --image_folder=generated --sample_size=-1 --save_dataset

rotate=( 90 180 -90 )
for i in "${rotate[@]}"
do
	python prepare_data.py --image_folder=generated --rotate=$i --save_images --dataset_csv=dataset.csv
done

# Augment with flipped images
python prepare_data.py --image_folder=generated --sample_size=-1 --save_dataset

python prepare_data.py --image_folder=generated --flip_horizontal --save_images --dataset_csv=dataset.csv
python prepare_data.py --image_folder=generated --flip_vertical --save_images --dataset_csv=dataset.csv

# Create hdf5 file for scale factors 2, 3, 4
python prepare_data.py --image_folder generated --sample_size -1 --hdf5_path train.h5
