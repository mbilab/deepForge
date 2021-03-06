#!/usr/bin/env/ python3

from __future__ import print_function, division

import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, Reshape, Flatten, Dropout, multiply, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Concatenate, Lambda, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
import keras.backend as K
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, RMSprop
from keras import metrics

import matplotlib.pyplot as plt
import os
import h5py

import numpy as np
import tensorflow as tf
from util import load_mnist, onehot, plot_table
from model import build_discriminator, build_generator, build_generator_incep

def exclude(arr):
    result = [ np.random.choice(list({0,1,2,3,4,5,6,7,8,9}-{digit}), 1)[0] for digit in arr ]
    return np.array(result)

class CGAN():
    def __init__(self, model_name=None):

        self.imgs, self.digits, self.test_imgs, self.test_digits = load_mnist()
        self.img_rows, self.img_cols, self.channels = self.imgs.shape[1:]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        loss_func = 'binary_crossentropy'

        optimizer_D = Adam(lr=0.0002)
        optimizer_G = Adam(lr=0.0002)

        self.D = build_discriminator()
        self.D.compile(loss= loss_func,
            optimizer=optimizer_D,
            metrics=[metrics.binary_accuracy])
        self.D.summary()
        self.G, self.G_mask = build_generator_incep()
        # self.G, self.G_mask = build_generator()

        img_input = Input(shape=self.img_shape)
        digit_input = Input(shape=(10,))
        img_added = self.G([img_input, digit_input])

        self.D.trainable = False
        D_output = self.D([img_added, digit_input])
        self.combined = Model([img_input, digit_input], D_output)
        self.combined.compile(loss=loss_func,
            optimizer=optimizer_G,
            metrics=[metrics.binary_accuracy])
        self.combined.summary()

        self.tb = keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=0,
            batch_size=64,
            write_graph=True,
            write_grads=True
        )
        self.tb.set_model(self.combined)

    def train(self, iterations, batch_size=128, sample_interval=100, save_model_interval=100,
                            train_D_iters=1, train_G_iters=1, img_dir='./', model_dir='./'):

        imgs, digits = self.imgs, self.digits

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        for itr in range(1, iterations + 1):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for _ in range(train_D_iters):
                # Select a random half batch of images
                idx_real = np.random.randint(0, imgs.shape[0], batch_size)
                idx_fake = np.random.randint(0, imgs.shape[0], batch_size)
                random_target_digits = onehot( np.random.randint(0, 10, batch_size), 10 )
                unmatch_digits = onehot( exclude(digits[idx_real]), 10 )
                real_imgs = imgs[idx_real]
                real_digits = onehot( digits[idx_real], 10 )
                fake_imgs = self.G.predict([imgs[idx_fake], random_target_digits])

                # real image and correct digit
                d_loss_real = self.D.train_on_batch([real_imgs, real_digits], valid)
                # fake image and random digit
                d_loss_fake = self.D.train_on_batch([fake_imgs, random_target_digits], fake)
                # real image but wrong digit
                d_loss_fake2 = self.D.train_on_batch([real_imgs, unmatch_digits], fake)
                # d_loss_fake2 = self.D.train_on_batch([real_imgs, unmatch_digits], fake)


            # tensorboard
            logs = {
                'D_loss_real': d_loss_real[0],
                'D_loss_fake': d_loss_fake[0],
                'D_loss_fake2': d_loss_fake2[0]
            }
            self.tb.on_epoch_end(itr, logs)

            # ---------------------
            #  Train Generator
            # ---------------------
            for _ in range(train_G_iters):
                # Condition on labels
                idx = np.random.randint(0, imgs.shape[0], batch_size)
                random_target_digits = onehot( np.random.randint(0, 10, batch_size), 10 )

                g_loss = self.combined.train_on_batch([imgs[idx], random_target_digits], valid)

                # tensorboard
                logs = {
                    'G_loss': g_loss[0],
                }
                self.tb.on_epoch_end(itr, logs)


            # If at save interval => save generated image samples
            if sample_interval > 0 and itr % sample_interval == 0:
                # self.sample_imgs(itr, img_dir)
                plot_table(self.G, self.D, os.path.join(img_dir, f'{itr}.png'), save=True)

            if save_model_interval > 0 and itr % save_model_interval == 0:
                if not os.path.isdir(model_dir):
                    os.makedirs(model_dir)
                self.D.save(os.path.join(model_dir, f'D{itr}.hdf5'))
                self.G.save(os.path.join(model_dir, f'G{itr}.hdf5'))
                self.G_mask.save(os.path.join(model_dir, f'G_mask{itr}.hdf5'))

            # Plot the progress
            print(f'{itr} [G loss: {g_loss[0]} | acc: {g_loss[1]}]')
            print(f'{itr} [D real: {d_loss_real[0]} | acc: {d_loss_real[1]}]')
            print(f'{itr} [D fake: {d_loss_fake[0]} | acc: {d_loss_fake[1]}]')
            print(f'{itr} [D fake2: {d_loss_fake2[0]} | acc: {d_loss_fake2[1]}]')
            print()

        self.tb.on_train_end(None)

    def sample_imgs(self, itr, img_dir):
        n = 5
        targets = onehot( np.full((n, 1), 4), 10 )
        test_imgs = self.test_imgs[:n]

        gen_imgs = self.G.predict([test_imgs, targets])
        masks = self.G_mask.predict([test_imgs, targets])
        D_pred_T = self.D.predict([test_imgs, targets])
        D_pred_F = self.D.predict([gen_imgs, targets])

        fig, axs = plt.subplots(n, 3, figsize=(8, 6))
        fig.tight_layout()
        for i in range(n):
            for no, img in enumerate([test_imgs, masks, gen_imgs]):
                axs[i, no].imshow(img[i, :, :, 0], cmap='gray')
                axs[i, no].axis('off')
                if 0 == no:
                    axs[i, no].text(-20, -2, f'D_pred_T: {D_pred_T[i]}')
                elif 2 == no:
                    axs[i, no].text(-20, -2, f'D_pred_F: {D_pred_F[i]}')
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)
        fig.savefig(os.path.join(img_dir, f'{itr}.png'))
        plt.close()



if __name__ == '__main__':

    version_name = 'test'
    model = CGAN()
    model.train(
            iterations=20000,
            batch_size=128,
            sample_interval=1000,
            save_model_interval=1000,
            train_D_iters=1,
            train_G_iters=1,
            img_dir=f'./outputs/{version_name}/imgs',
            model_dir=f'./outputs/{version_name}/models')

