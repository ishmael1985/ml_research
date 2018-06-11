import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import argparse
import re
import sys

from os import listdir
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument('--results_folder',
                    type=str,
                    required=True,
                    help="path to results")
parser.add_argument('--save_folder',
                    type=str,
                    required=False,
                    help="folder to save plots")
parser.add_argument('--scale',
                    type=int,
                    required=False,
                    default=3,
                    help="downsampling factor")

def main(args):
    opt = parser.parse_args(args)

    result_csv_regex = re.compile('results_nonaugmented_(?P<index>\d+).csv')
    csv_results = [f for f in listdir(opt.results_folder) \
                   if result_csv_regex.match(f)]
    global_avg_diffs = []
    intra_std_devs = []

    for f in csv_results:
        nonaugmented_set = {}
        augmented_set = {}

        index = result_csv_regex.match(f).group('index')
        non_augmented_csv = join(opt.results_folder,
                                 'results_nonaugmented_' + index + '.csv')
        augmented_csv = join(opt.results_folder,
                             'results_augmented_' + index + '.csv')

        with open(non_augmented_csv, "r") as csvfile:
            dataset_csv = csv.reader(csvfile)
            for row in dataset_csv:
                if row and int(row[1]) == opt.scale:
                    nonaugmented_set[row[0]] = float(row[2])

        if not nonaugmented_set:
            continue
                    
        with open(augmented_csv, "r") as csvfile:
            dataset_csv = csv.reader(csvfile)
            for row in dataset_csv:
                if row and int(row[1]) == opt.scale:
                    augmented_set[row[0]] = float(row[2])

        if not augmented_set:
            continue
                    
        diffs = []
        for image_file in augmented_set.keys():
            diffs.append(augmented_set[image_file] - nonaugmented_set[image_file])

        print("\nNumber of samples : " + str(len(diffs)))
        
        # mean_diff = sum(diffs) / float(len(diffs))
        mean_diff = np.mean(diffs)
        print("Average performance difference : " + str(mean_diff))

        global_avg_diffs.append((mean_diff, int(index)))

        print("Max difference : " + str(max(diffs)))
        print("Min difference : " + str(min(diffs)))

        deviation = np.std(diffs)
        print("Standard deviation : " + str(deviation))

        intra_std_devs.append(deviation)
        if opt.save_folder:
            fig = plt.figure()
            plt.plot(diffs)

            plt.title("Dataset " + index)
            plt.ylabel("PSNR difference (dB)")
            plt.xlabel("Image index")
            
            plt.savefig('{}/intra_diff_{}.png'.format(opt.save_folder, index))
            plt.close(fig)
        
    global_avg_diffs.sort(key=lambda x: x[1])
    mean_diffs, class_indices = zip(*global_avg_diffs)

    print("\n\nGlobal mean and standard deviation statistics:\n")
    print("Number of datasets : " + str(len(global_avg_diffs)))

    # Compute global mean PSNR diff and plot the graph
    global_mean_diff = np.mean(mean_diffs)
    print("Global average performance difference : " + str(global_mean_diff))

    if opt.save_folder:
        fig = plt.figure()
        plt.plot(class_indices, mean_diffs)

        plt.ylabel("PSNR difference (dB)")
        plt.xlabel("Dataset index")

        plt.savefig('{}/global_diff_average.png'.format(opt.save_folder))
        plt.close(fig)
    
    # Compute estimated intra-class standard deviation
    intra_std_dev = np.mean(intra_std_devs)
    print("Estimated intra-class standard deviation : " + str(intra_std_dev))

    # Compute inter-class standard deviation
    inter_std_dev = np.std(mean_diffs)
    print("Estimated inter-class standard deviation : " + str(inter_std_dev))

    # Compute total standard deviation
    total_std_dev = intra_std_dev + inter_std_dev
    print("Estimated total standard deviation : " + str(total_std_dev))
    
if __name__ == "__main__":
    main(sys.argv[1:])
