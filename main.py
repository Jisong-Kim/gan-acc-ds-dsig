import os
import warnings
import argparse
import tensorflow as tf
from tensorflow import keras

from config import build_paths, load_dso, get_config
from data_loader import DataLoader
from gan_generator import build_generator
from gan_critic import build_critic
from gp import gradient_penalty

warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # ==> choose GPU

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data.h5",
                        help="Path to HDF5 data file.")
    parser.add_argument("--dso", type=str, default="./dso_directory",
                        help="Directory path to design spectrum csv files.")
    parser.add_argument("--save", type=str, default="./save",
                        help="Directory path to save trained models.")
    return parser.parse_args()

def train():
    # -- Paths
    args = parse_args()
    path = build_paths(args.data, args.dso, args.save) 
    os.makedirs(path.savepth, exist_ok=True)
    
    dso_dict = load_dso(path.dso)
    
    # -- hyperparameters
    cfg = get_config()
    dso_dim = cfg.dso_dim
    batch_size = cfg.batch_size
    total_iters = cfg.total_iters
    n_critic = cfg.n_critic
    lambda_gp = cfg.lambda_gp
    lr = cfg.lr
    beta_1 = cfg.beta_1
    beta_2 = cfg.beta_2
    
    # -- Model build ==> summary
    generator = build_generator(dso_dim=dso_dim)
    critic = build_critic(dso_dim=dso_dim)

    generator_optimizer = keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2)
    critic_optimizer = keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2)

    generator.summary()
    critic.summary()
    
    # -- Data Loader
    loader = DataLoader(path.data, dso_dict, batch_size)
    
    # -- Training loop
    print("Starting training...")
    
    for iteration in range(total_iters):

        # -- Optimize Critic 
        for _ in range(n_critic):
            noise_batch = tf.random.normal(shape=(batch_size, 64, 64, 1))
            
            acc2d_batch, tx2d_batch, dso_batch = loader.get_batch()
            acc2d_batch = tf.convert_to_tensor(acc2d_batch, dtype=tf.float32)
            tx2d_batch = tf.convert_to_tensor(tx2d_batch, dtype=tf.float32)
            dso_batch = tf.convert_to_tensor(dso_batch, dtype=tf.float32)
            
            # assert acc2d_batch.shape == (batch_size, 64, 64, 1), f"Expected acc2d_batch shape: (-1, 64, 64, 1), but got {acc2d_batch.shape}"
            # assert tx2d_batch.shape == (batch_size, 64, 64, 1), f"Expected tx2d_batch shape: (-1, 64, 64, 1), but got {tx2d_batch.shape}"
            # assert dso_batch.shape == (batch_size, 200), f"Expected dso_batch shape: (-1, 200), but got {dso_batch.shape}"

            with tf.GradientTape() as tape:
                G_out = generator([dso_batch, tx2d_batch, noise_batch], training=True) # Generate artificial eq gms (2d forms)
                real_score = critic([acc2d_batch, dso_batch, tx2d_batch], training=True)
                fake_score = critic([G_out, dso_batch, tx2d_batch], training=True)
                
                d_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score) # -E[real] + E[fake]
                gp = gradient_penalty(critic, acc2d_batch, G_out, dso_batch, tx2d_batch) 
                d_loss_total = d_loss + lambda_gp * gp
                
            grads = tape.gradient(d_loss_total, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(grads, critic.trainable_variables))
            
        # -- Optimize Generator
        noise_batch = tf.random.normal(shape=(batch_size, 64, 64, 1))
        with tf.GradientTape() as tape:
            G_out = generator([dso_batch, tx2d_batch, noise_batch], training=True)
            fake_score = critic([G_out, dso_batch, tx2d_batch], training=True)
            G_adv_loss = -tf.reduce_mean(fake_score) # = -E[fake]
            
        grads = tape.gradient(G_adv_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    # -- Save
    generator.save(f'{path.savepth}/generator.h5')
    critic.save(f'{path.savepth}/critic.h5')

    print(f'Training finished')
    
if __name__ == "__main__":
    train()