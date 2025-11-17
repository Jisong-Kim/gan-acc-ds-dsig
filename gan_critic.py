from tensorflow import keras
from tensorflow.keras import layers
from layers import conv, dense

def build_critic(dso_dim=200):
    # Inputs: acc2d (-1, 64, 64, 1), dso (-1, 200), tx2d (-1, 64, 64, 1)
    acc2d_input = keras.Input(shape=(64, 64, 1), name="acc2d")
    dso_input = keras.Input(shape=(dso_dim,), name="dso")
    tx2d_input = keras.Input(shape=(64, 64, 1), name="tx2d")
    
    # branch 1: acc2d: (-1, 64, 64, 1) ==> (-1, 4, 4, 128)
    x_acc2d = conv(acc2d_input, 16) # (32, 32, 16)
    x_acc2d = conv(x_acc2d, 32) # (16, 16, 32)
    x_acc2d = conv(x_acc2d, 64) # (8, 8, 64)
    x_acc2d = conv(x_acc2d, 128) # (4, 4, 128)
    
    # branch 2: dso: (-1, 200) ==> (-1, 4, 4, 16)
    x_dso = dense(dso_input, 4 * 4 * 16)
    x_dso = layers.Reshape((4, 4, 16))(x_dso)
    
    # branch 3: tx2d: (-1, 64, 64, 1) ==> (-1, 4, 4, 16)
    x_tx2d = conv(tx2d_input, 8) # (32, 32, 8)
    x_tx2d = conv(x_tx2d, 16) # (16, 16, 16)
    x_tx2d = conv(x_tx2d, 32) # (8, 8, 32)
    x_tx2d = conv(x_tx2d, 16) # (4, 4, 16)
    
    # Stem: branch 1 + branch 2 + branch 3 (-1, 4, 4, 128+16+16) ==> (-1, 4, 4, 160)
    x = layers.Concatenate(axis=-1)([x_acc2d, x_dso, x_tx2d])
    
    # Stem: continued
    x = conv(x, 128, kernel_size=3, strides=1) # (4,4,128)
    x = layers.Flatten()(x)
    
    output = dense(x, 1) # (-1, 1)
    
    model = keras.Model(inputs=[acc2d_input, dso_input, tx2d_input], outputs=output, name="Critic")
    
    return model
