from tensorflow import keras
from tensorflow.keras import layers
from layers import conv, tpconv, dense

def build_generator(dso_dim=200):
    # Inputs: dso (-1, 200), tx2d (-1, 64, 64, 1), noise (-1, 64, 64, 1)
    dso_input = keras.Input(shape=(dso_dim,), name="dso")
    tx2d_input = keras.Input(shape=(64, 64, 1), name="tx2d")
    noise_input = keras.Input(shape=(64, 64, 1), name="noise")
    
    # branch 1: dso: (-1, 200) ==> (-1, 8, 8, 64)
    x_dso = dense(dso_input, 8 * 8 * 64) # (4096,)
    x_dso = layers.Reshape((8, 8, 64))(x_dso) # (8, 8, 64)
    x_dso = layers.LeakyReLU(alpha=0.2)(x_dso)
    
    # branch 2: tx2d + noise: (-1, 64, 64, 1) + (-1, 64, 64, 1) ==> (-1, 64, 64, 2)
    x = layers.Concatenate(axis=-1)([tx2d_input, noise_input])
    
    # branch 2: continued
    x = conv(x, 16) # (32, 32, 16)
    x = conv(x, 32) # (16, 16, 32)
    tx2d_noise_feat = conv(x, 64) # (8, 8, 64)
    
    # Stem: branch 1 + branch 2 (-1, 8, 8, 64+64) ==> (-1, 8, 8, 128)
    combined = layers.Concatenate(axis=-1)([tx2d_noise_feat, x_dso])
    
    # Stem: continued
    x = tpconv(combined, 64) # (16, 16, 64)
    x = tpconv(x, 32) # (32, 32, 32)
    x = tpconv(x, 16) # (64, 64, 16)
    
    output = conv(x, filters=1, kernel_size=3, strides=1, padding='same', use_activation=False) # (64, 64, 1)
    
    model = keras.Model(inputs=[dso_input, tx2d_input, noise_input], outputs=output, name="Generator")
    
    return model