import tensorflow as tf

def gradient_penalty(critic, real_acc2d, fake_acc2d, dso, tx2d):
    batch_size = tf.shape(real_acc2d)[0]
    
    epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], 
                                minval=0., maxval=1.)
    
    x_hat = epsilon * real_acc2d + (1 - epsilon) * fake_acc2d
    
    with tf.GradientTape() as tape:
        tape.watch(x_hat)
        d_out = critic([x_hat, dso, tx2d])
        
    grads = tape.gradient(d_out, [x_hat])[0]
    grads = tf.reshape(grads, [batch_size, -1])
    l2_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
    gp = tf.reduce_mean((l2_norms - 1.0) ** 2)
    
    return gp