from keras.datasets import mnist
import numpy as np
import os
from matplotlib import pyplot as plt
from keras.models import load_model


#! where here?
def load_mnist(): # {{{
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # pad to 32*32 (from 28*28) and normalize to 0~1
    # the shape of `x_train` and `x_test` is (#sample, height, width)
    # so only the last two axes have to be padded
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'constant') / 255
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), 'constant') / 255
    # expand channel dim
    x_train, x_test = x_train[:, :, :, np.newaxis], x_test[:, :, :, np.newaxis]

    return x_train, y_train, x_test, y_test
# }}}


#! where here?
def onehot(x, size): # {{{
    x2 = np.zeros((x.size, size))
    x2[range(x.size), x] = 1
    return x2
# }}}


def plot_table(G, name, random=True, save=False):
    if type(G) == str:
        G = load_model(G)
    _, _, imgs, digits = load_mnist()
    input_imgs = [] # imgs of 0 ~ 9
    for i in range(10):
        imgs_filtered = imgs[np.where(digits == i)[0]]
        if random:
            idx = np.random.randint(0, imgs_filtered.shape[0])
        else:
            idx = 0
        input_imgs.append(imgs_filtered[idx])

    fig, axs = plt.subplots(10, 10)
    fig.set_size_inches(50, 50)
    for i in range(10): # for input img
        for j in range(10): # for target digit
            gen_img = G.predict([np.expand_dims(input_imgs[i], axis=0), onehot(np.full((1, 1), j), 10)])
            axs[i, j].imshow(gen_img[0, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
    plt.show()

    if save:
        fig.savefig(name, dpi=150)


def plot_fig(G, G_mask, n, t):

    _, _, imgs, digits = load_mnist()

    target = onehot(np.full((1, 1), t), 10)
    use_img = imgs[n][np.newaxis, ...]
    gen_img = G.predict([use_img, target])
    mask_img = G_mask.predict([use_img, target])
    fig, axs = plt.subplots(1, 3, figsize=(4, 3))
    axs[0].imshow(use_img[0, :, :, 0], cmap='gray')
    axs[0].axis('off')
    axs[1].imshow(mask_img[0, :, :, 0], cmap='gray')
    axs[1].axis('off')
    axs[2].imshow(gen_img[0, :, :, 0], cmap='gray')
    axs[2].axis('off')
    fig.savefig(os.path.join(f'./tmp/fig_{n}_{t}'))
