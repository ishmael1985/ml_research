import tensorflow as tf
import numpy as np
import argparse
import csv
import pprint

from torchvision.transforms import CenterCrop, Compose
from model import ESPCN
from dataset import DatasetFromFolder
from utils import compute_psnr
from PIL import Image

flags = tf.app.flags
flags.DEFINE_string("image_folder", None, "image_folder containing test images")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_integer("sample_size", 10, "number of random samples to test [10]")
flags.DEFINE_integer("scale", 3, "downsampling factor [3]")
flags.DEFINE_boolean('save_result', False, "save PSNR results [False]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()


def get_center_crop(image, width, height):
    composed_transform = Compose([CenterCrop(size=(height, width))])
    return composed_transform(image)

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    
    average_psnr = 0
    sampled_dataset = DatasetFromFolder(image_dir=FLAGS.image_folder,
                                        sample_size=FLAGS.sample_size,
                                        scale=FLAGS.scale)
    with tf.Session() as session:
        espcn = ESPCN(session)

        if FLAGS.save_result:
            results_file = open("results.csv", "w")
            results_csv = csv.writer(results_file)
            
        for ground_truth in sampled_dataset:
            downsampled_image = sampled_dataset.transform(ground_truth)
            out_img = espcn.test(FLAGS.checkpoint_dir, downsampled_image)
            
            out_img *= 255.0
            out_img = out_img.clip(0, 255)
            
            height, width = out_img.shape[:2]
            ground_truth = get_center_crop(ground_truth, width, height)
            gt_img = np.asarray(ground_truth.split()[0])

            #img = Image.fromarray(out_img.astype('uint8'), 'L')
            #img.show()

            psnr = compute_psnr(out_img, gt_img)

            if FLAGS.save_result:
                results_csv.writerow([sampled_dataset.current_image_file, psnr])

            average_psnr += compute_psnr(out_img, gt_img)
        
    print("Upscale factor = ", FLAGS.scale)
    print("Average PSNR for {} samples = {}".format(FLAGS.sample_size,
                                                    average_psnr / FLAGS.sample_size))

if __name__ == "__main__":
    tf.app.run()
