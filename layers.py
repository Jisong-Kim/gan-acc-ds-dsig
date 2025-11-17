from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal

def conv(x, filters, kernel_size=4, strides=2, padding="same", use_activation=True):
    # Conv2D + LeakyReLU
    x = layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    
    if use_activation:
        x = layers.LeakyReLU(alpha=0.2)(x)
        
    return x

def tpconv(x, filters, kernel_size=4, strides=2, padding="same", use_activation=True):
    # Conv2DTranspose + LeakyReLU
    x = layers.Conv2DTranspose(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    
    if use_activation:
        x = layers.LeakyReLU(alpha=0.2)(x)
        
    return x

def dense(x, units):
    # Dense layer
    x = layers.Dense(
        units,
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(x)
    return x