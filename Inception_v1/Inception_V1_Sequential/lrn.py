
# lrn.py

"""
Thanks for the guys for the contributions. swghosh modified the script based on the 
original scrip of joelouismarino. Mike modifies it to adatp to TensorFlow 2.3 and 
Keras 2.4.3
"""

from tensorflow.keras.layers import Layer, Lambda
import tensorflow as tf

# wraps up the tf.nn.local_response_normalisation into the keras.layers.Lambda inn order 
# to  have a custom keras layer as a class that will perform LRN ops.     
class LRN(Lambda):

    def __init__(self, alpha=0.0001, beta=0.75, depth_radius=5, **kwargs):
        # using parameter defaults as per GoogLeNet
        params = {
            "alpha": alpha,
            "beta": beta,
            "depth_radius": depth_radius
        }
        # Construct a function for use with Keras Lambda
        lrn_fn = lambda inputs: tf.nn.local_response_normalization(inputs, **params)

        # Pass the function to Keras Lambda
        return super().__init__(lrn_fn, **kwargs)

# The layer is also required by GoogLeNet (same as above)
class PoolHelper(Layer):
    
    def __init__(self, **kwargs):
        super(PoolHelper, self).__init__(**kwargs)
    
    def call(self, x, mask=None):
        return x[:,:,1:,1:]
    
    def get_config(self):
        config = {}
        base_config = super(PoolHelper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))