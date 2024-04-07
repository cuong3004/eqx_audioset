import tensorflow as tf
from tensorflow import keras
import keras_cv


mix_up = keras_cv.layers.MixUp()

@tf.function(jit_compile=True)
def random_linear_fader(sample):
    lms = sample
    head_tail = (2.0 * tf.random.uniform((2,))) - 1.0
    T = lms.shape[-2]
    slope = tf.linspace(head_tail[0], head_tail[1], T)
    slope = tf.reshape(slope, (1, T, 1))
    sample = lms + slope
    return sample

@tf.function(jit_compile=True)
def min_max_normalize(images, epsilon=1e-7):
        min_val = -80.0
        max_val = 0.0
        images = (images - min_val) / (max_val - min_val + epsilon)
        return images
    
def augmentation_single(sample):
    images = sample['images']
    images = min_max_normalize(images)
    images = random_linear_fader(images)
    sample['images'] = images
    return sample