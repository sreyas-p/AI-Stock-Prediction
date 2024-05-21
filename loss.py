import tensorflow as tf
import numpy as np
import pandas as pd

def my_loss_function(y_true, y_pred):
    threshold = 1
    error = y_true - y_pred
    senario_check = tf.abs(error) <= threshold
    senario_big = threshold * (tf.abs(error) - (0.5*threshold))
    senario_small = tf.square(error)/2
    return tf.where(senario_check, senario_small, senario_big)