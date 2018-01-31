import numpy as np
import scipy.stats as st
import csv
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('csv_file', nargs=2)

def main():
    opt = parser.parse_args()
    
    augmented_set = {}
    with open(opt.csv_file[0], "r") as csvfile:
        dataset_csv = csv.reader(csvfile)
        for row in dataset_csv:
            if row:
                augmented_set[row[0]] = float(row[1])
            
    nonaugmented_set = {}
    with open(opt.csv_file[1], "r") as csvfile:
        dataset_csv = csv.reader(csvfile)
        for row in dataset_csv:
            if row:
                nonaugmented_set[row[0]] = float(row[1])
                
    
    
    diffs = []
    for image_file in augmented_set.keys():
        diffs.append(augmented_set[image_file] - nonaugmented_set[image_file])
    
    print("Number of samples : " + str(len(diffs)))    
    
    # mean_diff = sum(diffs) / float(len(diffs))
    mean_diff = np.mean(diffs)
    print("Average performance difference : " + str(mean_diff))
    
    '''
    deviation = 0
    for diff in diffs:
        deviation += (diff - mean_diff) ** 2
    deviation = math.sqrt(deviation / float(len(diffs)))
    '''
    deviation = np.std(diffs)
    print("Standard deviation : " + str(deviation))
    
    confidence_interval = st.t.interval(0.95, len(diffs)-1, loc=np.mean(diffs), scale=st.sem(diffs))
    print("Confidence interval range : " + str(confidence_interval))
    
    W, p = st.wilcoxon(diffs)
    print("Wilcoxon signed-rank test\n{:<10} : {}\n{:<10} : {}".format("statistic", W, "p-value", p))
    
    W, p = st.shapiro(diffs)
    print("Shapiro-Wilk test\n{:<10} : {}\n{:<10} : {}".format("statistic", W, "p-value", p))
    
if __name__ == "__main__":
    main()
