Deep Learning Framework for Super-Resolution
============================================
This README serves as a brief description of the Python-based deep learning framework used for our super-resolution research.

Objectives
----------
The main objective of this project is to aggregate the most well-known state-of-the-art deep learning super-resolution architectures under a single framework.
This allows us to perform an apple-to-apple comparison of each of these architectures and ensure that the parameters used for training and testing are consistent, and obtain reliable performance metrics for these architectures.

Usage
-----

**Data Augmentation**

The main script used for data augmentation is `prepare_data.py`. Example usage is as follows:

	python3 prepare_data.py --image_folder=/path/to/training/dataset --sample-size=10 --rotate=15 --scale=2 --hdf5_path=/path/to/h5/file

Running the script as above creates 10 samples which have been rotated by 15 degrees and downsampled by a factor of 2. To prevent the network from learning feature distortions introduced by black borders during training, we take the maximum cropped area of the image content for the rotated image without those borders.
Specifying the hdf5 path (--hdf5_path) means that we intend to save the resulting pre-processed images using the h5 file format.
The specified h5 file that has been created saves crops of the pre-processed images as specified by JSON config file `config.json`.
Further clarity of `prepare_data.py` usage can be seen in the `generate_dataset.sh` shell script. Executing `generate_dataset.sh` is shown as follows:

	sh generate_dataset.sh /path/to/training/dataset 100 15

The resulting output of executing the above command would be a `generated` folder created in the script location, which contains 200 preprocessed images generated from 100 randomized samples of the specified image dataset.
The `generated` training dataset contain 100 images which have been rotated and downsampled by 15 and a factor of 2 respectively, and another 100 images that have been downsampled by a factor of 2, both coming from the original 100 randomly selected samples of the specified image dataset.
This is done for the purpose of studying the generalization performance of a network when the training dataset is augmented with rotated transformations of original images.

**Training**

Training a super-resolution (SR) model with a specified input training dataset is done by `sr_train.py`, executed as follows:

	python3 sr_train.py --hdf5_path=/path/to/h5/file --epochs=1000

The above command takes in an hdf5 file which had been previously generated by `prepare_data.py`, and trains the network for 1000 epochs.
Training parameters, such scaling factor and number of color channels, can be found in the JSON config file `config.json`.
Details of other parameters for training such as checkpoint directory, batch size, learning rate, etc. can be found by passing --help to `sr_train.py`.

**Testing**

For testing the trained SR model, we execute the following command:

	python3 sr_test.py --image_folder=/path/to/test/dataset --scale=3 --sample_size=100 --save_result --save_test

Super-resolution is carried out by the inference script `sr_test.py` as shown in the above command.
In this case, we are super-resolving 100 randomly sampled test images by a factor of 3, which will be first downsampled by the same factor using bicubic interpolation prior to super-resolution.
At the end of execution, we will obtain the average PSNR between the ground truth images and the super-resolved images, and individual PSNR results will be saved into a `result.csv` file for later analysis.
The list of images used for the test are finally saved in a `test.csv` file.
To re-use the same set of test images with a differently trained model, we would load the `test.csv` generated by a previous super-resolution execution instance of `sr_test.py` as shown in the above command.
This is done as follows:

	python3 sr_test.py --image_folder=/path/to/test/dataset --checkpoint_dir=/path/to/checkpoint/folder --scale=3 --load_test='test.csv' --save_result

In this case, the new model checkpoints saved by Tensorflow during training are specified for loading into the network to perform inferencing.


TODO
----
* Only model implemented so far is ESPCN by Shi et. al (2016). Other models such as FSRCNN, VDSR, SRGAN and LapSRN are being highly considered for implementation.
* Statistical analysis tools for visualization

License
-------
This project is made available under the terms of the Apache License v2.0.