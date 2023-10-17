'''
@author: Junwei Yu
@contact: yuju@tcd.ie
@file : preprocess.py
@time: 2023/07/30 
'''
import cv2
import numpy as np
import os

def calculate_mean_and_std_per_channel(dataset_path):
    sum_image = np.zeros((3,), dtype=np.float64)
    sum_sq_diff = np.zeros((3,), dtype=np.float64)
    count = 0

    for filename in os.listdir(dataset_path):
        filepath = os.path.join(dataset_path, filename)
        image = cv2.imread(filepath)

        sum_image += np.sum(image, axis=(0,1))
        count += image.shape[0] * image.shape[1]

    mean_image = sum_image / count

    for filename in os.listdir(dataset_path):
        filepath = os.path.join(dataset_path, filename)
        image = cv2.imread(filepath)

        sq_diff = (image - mean_image) ** 2
        sum_sq_diff += np.sum(sq_diff, axis=(0,1))

    variance_image = sum_sq_diff / count
    std_image = np.sqrt(variance_image)

    mean_per_channel = mean_image
    std_per_channel = std_image

    return mean_per_channel, std_per_channel

dataset_path = "../data/train/images"
mean_per_channel, std_per_channel = calculate_mean_and_std_per_channel(dataset_path)
print("Mean per channel (B, G, R):", mean_per_channel)
print("Std per channel (B, G, R):", std_per_channel)


