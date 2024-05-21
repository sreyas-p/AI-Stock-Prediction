import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split

def format_data(data):
  y1 = data.pop('Y1')
  y1 = np.array(y1)
  return y1

def norm(x):
  return (x - train_stats['mean']/train_stats['std'])

data_url = w
