#!/bin/bash

# Generate images with downsampled sizes of 1, 0.9, 0.8, 0.7, and 0.6 factors
python prepare_data.py --image_folder=$1 --sample_size=-1 --save_images --save_dataset

downsize=( 1.1111 1.25 1.4286 1.6667 )
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

# Create hdf5 file for scale factor 3
python prepare_data.py --image_folder generated --sample_size -1 --hdf5_path train.h5
