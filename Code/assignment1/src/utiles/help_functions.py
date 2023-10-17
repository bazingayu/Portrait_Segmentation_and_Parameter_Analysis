'''
@author: Junwei Yu
@contact: yuju@tcd.ie
@file : help_functions.py
@time: 2023/07/29 
'''

import visdom
import numpy as np
import tensorflow as tf

class ImageVisualizer:
    def __init__(self, viz):
        self.viz = viz
        self.mean_value = [125.40095372, 194.61185261, 173.29846314]
        self.std_value = [69.33723314, 41.24116595, 46.72013978]

    def _process_image(self, img, sample_weight):
        # Map logits to RGB for visualization
        visualization = self.map_logits_to_rgb(img)[0]
        # Convert to proper format
        visualization = tf.transpose(visualization, perm=[2, 0, 1])  # switch to (C, H, W)
        # Convert data type to float32
        visualization = tf.cast(visualization, tf.float32)
        visualization = visualization.numpy()
        sample_weight_numpy = sample_weight.numpy()[0]
        visualization[0, ...] = np.where(sample_weight_numpy == 0, 0, visualization[0, ...] )
        visualization[1, ...]  = np.where(sample_weight_numpy == 0, 0, visualization[1, ...] )
        visualization[2, ...]  = np.where(sample_weight_numpy == 0, 0, visualization[2, ...] )
        return visualization

    def show_images(self, x_batch_val, y_batch_val, val_logits, sample_weight):
        x_batch_val = x_batch_val.numpy()[0]
        for i in range(3):
            x_batch_val[:, :, i] = x_batch_val[:, :, i] * self.std_value[i] + self.mean_value[i]
        x_batch_val_visualization = np.transpose(x_batch_val, (2, 0, 1))
        x_batch_val_visualization = x_batch_val_visualization[np.newaxis, :]

        gt_visualization = self._process_image(y_batch_val, sample_weight)
        rgb_visualization = self._process_image(val_logits, sample_weight)
        x_batch_val_visualization = x_batch_val_visualization[0]
        class_map = tf.argmax(y_batch_val, axis=-1).numpy()[0]
        sample_weight_numpy = sample_weight.numpy()[0]

        x_batch_val_visualization[2, ...] = np.where(class_map==0, 255, x_batch_val_visualization[2, ...])
        x_batch_val_visualization[1, ...] = np.where(sample_weight_numpy == 0, 255, x_batch_val_visualization[1, ...])


        # Visualize in Visdom
        self.viz.images([x_batch_val_visualization, gt_visualization, rgb_visualization], win="Predictions")

    def map_logits_to_rgb(self, logits):
        # Convert logits to probabilities and take argmax to get the most probable class
        class_map = tf.argmax(logits, axis=-1).numpy()

        rgb_map = np.zeros((*class_map.shape, 3))

        # Define the mapping from class to RGB color
        label_to_color = {
            0: (230, 25, 75),  # Red, 1, glomerulus
            1: (0, 255, 0),  # Green, 2, normal tissue
            2: (0, 255, 255)  # Blue, 3, background
        }
        # Map each class in the class_map to its corresponding color
        for class_id, color in label_to_color.items():
            rgb_map[class_map == class_id] = color

        return tf.convert_to_tensor(rgb_map, dtype=tf.uint8)
