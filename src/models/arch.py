from src.models.flip_gradient import flip_gradient
from src.data_gen.preprocessing import DataAugmentation
from src.models import i3d
#import i3d
import tensorflow as tf
import numpy as np

""" TensorFlow Architecture definitions for classification heads + base model.
"""

def i3d_model(input_images, is_training, num_labels,dropout,flip_classifier_gradient=False,flip_weight=1.0,
              aux_classifier=False, feat_level='features'):
    rgb_model = i3d.InceptionI3d(num_labels+num_labels, spatial_squeeze=True, final_endpoint='Logits',
                                 aux_classifier=aux_classifier)
    logits, endpoints = rgb_model(input_images, is_training=is_training,dropout_keep_prob=dropout,
                                  flip_classifier_gradient=flip_classifier_gradient,flip_weight=flip_weight)

    if aux_classifier:
        aux_classifier_logits = endpoints['aux_classifier']
    else:
        aux_classifier_logits = None
    #domain_feat = endpoints['features']
    features = endpoints[feat_level]
    return logits, aux_classifier_logits, features

def build_i3d(reuse_variables, input_images, is_training, num_labels, flow, temporal_window,dropout,
              flip_classifier_gradient,flip_weight=1.0,aux_classifier=False, feat_level='features'):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        return i3d_model(input_images, is_training,num_labels,dropout,
                         flip_classifier_gradient=flip_classifier_gradient,flip_weight=flip_weight,
                         aux_classifier=aux_classifier,feat_level=feat_level)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def domain_classifier(feat):
    shape = feat.get_shape().as_list()
    dim = np.prod(shape[1:])
    feat = tf.reshape(feat, [-1, dim])
    with tf.variable_scope("Domain_Classifier"):
        d_h_fc0 = tf.layers.dense(feat,100,kernel_initializer=tf.initializers.truncated_normal(stddev=0.1),name="first")
        d_h_fc0 = tf.nn.relu(d_h_fc0)
        d_logits = tf.layers.dense(d_h_fc0,2,
                                  kernel_initializer=tf.initializers.truncated_normal(stddev=0.1),name="second")
        return d_logits

def predict_synch(feat):
    shape = feat.get_shape().as_list()
    dim = np.prod(shape[1:])
    feat = tf.reshape(feat, [-1, dim])

    d_h_fc0 = tf.layers.dense(feat, 100, kernel_initializer=tf.initializers.truncated_normal(stddev=0.1),
                              name="first")
    d_h_fc0 = tf.nn.relu(d_h_fc0)

    d_logits = tf.layers.dense(d_h_fc0, 2,
                               kernel_initializer=tf.initializers.truncated_normal(stddev=0.1), name="second")
    return d_logits, d_h_fc0