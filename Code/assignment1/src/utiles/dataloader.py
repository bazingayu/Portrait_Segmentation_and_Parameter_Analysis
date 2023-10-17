'''
@author: Junwei Yu
@contact: yuju@tcd.ie
@file : dataloader.py
@time: 2023/07/28 
'''

import os
import cv2
import numpy as np
import tensorflow as tf
from .dataaugment import DataAugmenter

color_to_label = {
    (75, 25, 230): 0,    # Red, 1, glomerulus
    (0, 255, 0): 1,    # Green, 2, normal tissue
    (255, 255, 0): 2,    # Blue, 3, background
    (0, 0, 0): 3       # Black, 4, unlabeled data
}


class SegmentationDataset:
    def __init__(self, img_dir, mask_dir, batch_size=8, buffer_size=1000, img_size=(256, 256), shuffle=True, augmenter=None):
        # produce the corresponding img & mask name
        self.img_names = [os.path.join(img_dir, filename) for filename in os.listdir(img_dir)]
        self.mask_names = []
        for img_name in self.img_names:
            self.mask_names.append(os.path.join(mask_dir, os.path.basename(img_name)[:-5] + "A.png"))

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augmenter = augmenter
        self.class_weights = self.compute_class_weights()
        self.mean_value = [125.40095372, 194.61185261, 173.29846314]
        self.std_value = [69.33723314, 41.24116595, 46.72013978]

        print(self.class_weights)

    def compute_class_weights(self):

        # Initialize the class frequencies dictionary
        class_freqs = {i: 0 for i in range(4)}

        # Iterate over the mask dataset
        for mask_name in self.mask_names:
            mask = cv2.imread(mask_name)  # Assuming the masks are saved as grayscale images

            # Convert the color mask to label mask
            label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
            for color, label in color_to_label.items():
                label_mask[np.all(mask == color, axis=-1)] = label


            # Update the class frequencies
            for i in range(len(color_to_label)):
                class_freqs[i] += np.sum(label_mask == i)

        # Compute the total number of pixels
        total_pixels = sum(class_freqs.values()) - class_freqs[3]
        # Compute the class weights
        print(class_freqs)
        class_weights = {i: total_pixels / np.sqrt(class_freqs[i] + 1e-6) for i in range(len(color_to_label))}
        class_weights[3] = 0.0
        print(class_weights)
        sum_weights = sum(class_weights.values())
        class_weights = {i: weight * len(color_to_label) / sum_weights for i, weight in class_weights.items()}
        print(class_weights)

        return class_weights

    def _load_image(self, img_path):
        img = tf.io.read_file(img_path)   #RGB

        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, self.img_size)

        img = tf.cast(img, tf.float32)

        channels = tf.split(img, num_or_size_splits=3, axis=-1)

        # normalize
        for i in range(3):
            channels[i] = (channels[i] - self.mean_value[i]) / self.std_value[i]

        img = tf.concat(channels, axis=-1)

        return img

    def _load_mask(self, mask_path):
        [mask,] = tf.py_function(self._load_and_convert_mask, [mask_path], [tf.uint8])
        return mask

    def _load_and_convert_mask(self, mask_path):
        mask = cv2.imread(mask_path.numpy().decode())
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        mask_label = np.zeros(mask.shape[:2], dtype=np.int32)
        for color, label in color_to_label.items():
            mask_label[(mask == color).all(axis=-1)] = label
        return mask_label

    def _load_sample_weights(self, mask):
        # Convert the one-hot mask back to label indices
        # mask_indices = tf.argmax(mask, axis=-1)

        # Initialize the sample weights with zeros
        weights = tf.zeros(self.img_size, dtype=tf.float32)

        # Assign the class weights
        for label, class_weight in self.class_weights.items():
            class_weight_tensor = tf.constant(class_weight, dtype=tf.float32)
            weights = tf.where(mask == label, class_weight_tensor, weights)

        return weights


    def _process_path(self, img_path, mask_path, ):
        img = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        # Perform data augmentation
        if self.augmenter is not None:
            img, mask = self.augmenter(img, mask)

        weights = self._load_sample_weights(mask)
        mask = tf.one_hot(mask, depth=len(color_to_label))
        mask = mask[..., :-1]
        return img, mask, weights

    def get_dataset(self):
        img_dataset = tf.data.Dataset.from_tensor_slices(self.img_names)
        mask_dataset = tf.data.Dataset.from_tensor_slices(self.mask_names)

        dataset = tf.data.Dataset.zip((img_dataset, mask_dataset))
        dataset = dataset.map(self._process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
