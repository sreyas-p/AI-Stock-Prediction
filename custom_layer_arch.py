import tensorflow as tf
from tensorflow.keras.layers import Layer

class DenseLayer(Layer):
    
    def __init__(self, units= 32):
        super(DenseLayer, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        #weight parameter w init
        w_init = tf.random_normal_initializer()
        #training weight 
        self.w = tf.Variable(name = "kernel", initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'), trainable=True)
        #weight parameter b init
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name = "bias", initial_value=b_init(shape=(self.units,), dtype='float32'), trainable=True)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
mydense = DenseLayer(units=1)
x = tf.ones(1,1)
y = mydense(x)