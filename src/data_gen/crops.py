import tensorflow as tf

""" Cropping functions to apply the same crop to all frames in a video segment
"""

def _crop_segment(segment, o_h, crop_height, o_w, crop_width, input_size):
    return segment[:, o_h:(o_h + crop_height), o_w:(o_w + crop_width)]


def _training_offsets(crop_pairs, width, height,h_rand,w_rand):
    w_step = tf.cast((width - crop_pairs[1]) / 4, dtype=tf.int32)
    h_step = tf.cast((height - crop_pairs[0]) / 4, dtype=tf.int32)
    o_h = tf.multiply(h_rand, h_step)
    o_w = tf.multiply(w_rand, w_step)
    return o_h, o_w


def training_crop(segment, height, width, input_size, scales, s_rand, h_rand, w_rand):
    crop_pairs = scales[s_rand, :]
    # Choose random crop given scale
    o_h, o_w = _training_offsets(crop_pairs, width, height, h_rand, w_rand)
    # Crop segment
    return _crop_segment(segment, o_h, crop_pairs[0], o_w, crop_pairs[1], input_size)


def testing_crop(segment, height, width, input_size):
    o_w = tf.cast((width - input_size) / 2, tf.int32)
    o_h = tf.cast((height - input_size) / 2, tf.int32)
    return segment[:, o_h:(o_h + input_size), o_w:(o_w + input_size)]


def random_flip(segment, f_rand, invert=False):
    # Choose random number
    mirror_cond = tf.less(f_rand, .5)
    tf.summary.scalar("To Flip?", f_rand)
    invert_cond = tf.math.logical_and(tf.constant(invert), mirror_cond)
    segment = tf.cond(mirror_cond, lambda: tf.reverse(segment, [-2]), lambda: segment)
    inverted_segment = 255 - segment
    return tf.cond(invert_cond, lambda: inverted_segment, lambda: segment)


def normalise(segment):
    segment = tf.div(segment, 256.0)
    segment = tf.subtract(segment, 0.5)
    return tf.multiply(segment, 2.0)