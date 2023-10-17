'''
@author: Junwei Yu
@contact : yuju@tcd.ie
@file: train.py
@time: 2023/7/28 12:24
'''
import sys
sys.path.append('../')
sys.path.append('./')
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from src.utiles.dataloader import SegmentationDataset
from src.models.unet import build_unet
from src.utiles.losses import get_loss, combine_loss
from src.utiles.metrics import get_metrics, class_wise_iou, WeightedMeanIoU
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from visdom import Visdom
from src.utiles.help_functions import ImageVisualizer
from src.utiles.dataaugment import DataAugmenter
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
import os



def main():
    # Initialize the data generator
    batch_size = 8
    buffer_size = 1000
    img_size = (256, 256)
    best_miou = 0.95

    # 定义保存模型的路径，您可以更改这个路径
    save_path = './saved_models/'
    os.makedirs(save_path, exist_ok=True)
    augmenter = DataAugmenter(rotation_range=0.2, brightness_range=0.1, contrast_range=0.1, noise_stddev=0.001)
    train_data = SegmentationDataset("./data/train/images", "./data/train/mask_ori", batch_size, buffer_size, img_size, augmenter=augmenter).get_dataset()
    val_data = SegmentationDataset("./data/val/images", "./data/val/mask_ori", batch_size, buffer_size, img_size).get_dataset()

    # Create the model
    model = build_unet()

    # Specify the learning rate decay
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.0001,
        decay_steps=300,
        decay_rate=0.5)

    # Specify the optimizer
    optimizer = Adam(learning_rate=lr_schedule)

    # Specify the loss function
    loss_fn = combine_loss

    # Specify the metrics
    train_mean_iou = WeightedMeanIoU(num_classes=3)
    val_mean_iou = WeightedMeanIoU(num_classes=3)

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[train_mean_iou])

    # set the visdom visualization
    viz = Visdom("yujunwei.love", port=8097, env='deciphex-assignment-exp3')

    loss_window = viz.line(Y=[0.], X=[0.], opts=dict(title='Loss'))
    metric_train_window = viz.line(Y=[0.], X=[0.], opts=dict(title='train Mean IoU'))
    metric_val_window = viz.line(Y=[0.], X=[0.], opts=dict(title='val Mean IoU'))

    visualizier = ImageVisualizer(viz)

    # Train the model
    epochs = 500
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train, sample_weight) in enumerate(train_data):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits, weights=sample_weight)
                visualizier.show_images(x_batch_train, y_batch_train, logits[0], sample_weight)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))



            # Update the metrics
            train_mean_iou.update_state(y_batch_train, logits[0], sample_weight=sample_weight)

        # Display metrics at the end of each epoch.
        train_miou = train_mean_iou.result()
        print("Training MeanIoU over epoch: %.4f" % (float(train_miou),))

        # Update Visdom
        loss_value = tf.reduce_mean(loss_value)

        viz.line(Y=[float(loss_value)], X=[epoch], win=loss_window, update='append')
        viz.line(Y=[float(train_miou)], X=[epoch], win=metric_train_window, update='append')

        # Reset training metrics at the end of each epoch
        train_mean_iou.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val, sample_weight in val_data:
            # Run a validation loop at the end of each epoch.
            val_logits = model(x_batch_val, training=False)

            visualizier.show_images(x_batch_val, y_batch_val, val_logits[0], sample_weight)

            # Update val metrics
            val_mean_iou.update_state(y_batch_val, val_logits[0], sample_weight=sample_weight)

        val_miou = val_mean_iou.result()
        val_mean_iou.reset_states()
        print("Validation MeanIoU: %.4f" % (float(val_miou),))
        viz.line(Y=[float(val_miou)], X=[epoch], win=metric_val_window, update='append')
        # check the best IOU
        if float(val_miou) > best_miou:
            best_miou = float(val_miou)
            model_file_name = f'model_epoch_{epoch}_best_miou_{best_miou:.4f}.h5'
            model.save(save_path + model_file_name)
            print(f"New best MIOU reached! Model saved to {model_file_name}")
        print(best_miou)

if __name__ == "__main__":
    main()
