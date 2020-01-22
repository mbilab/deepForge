# standard import
from math import ceil
import os

# third-part import
from keras.datasets import mnist
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np


class Evaluator:

    def __init__(self, gen_model, gen_mask_model, mnist_model='./outputs/D_digit.hdf5'):
        self.G = load_model(gen_model)
        self.G_mask = load_model(gen_mask_model)
        self.clf = load_model(mnist_model)

    def classify(self, imgs):
        if imgs.ndim < 4:
            imgs = imgs[np.newaxis, :]
        clf_prob = self.clf.predict(imgs)
        clf_res = np.argmax(clf_prob, axis=1)
        return clf_res

    def run(self, images, digits):
        self.images = images
        self.digits = np.full(len(images), digits) if int == type(digits) else digits
        onehot_digits = onehot(np.array(self.digits), 10)
        #! why two models?
        self.adds = self.G.predict([images, onehot_digits])
        self.masks = self.G_mask.predict([images, onehot_digits])
        self.predicted_digits = self.classify(self.adds)

    def score(self): # {{{
        hits = self.predicted_digits == self.digits
        return {
            '#samples': len(self.images),
            '#hits': np.sum(hits),
            '%hit': np.mean(hits),
        }
    # }}}

    def _plot_fig(self, image, mask, add, dpi=96, text=None, wspace=.02): # {{{
        plt.figure(dpi=dpi, figsize=(3, 1))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=wspace, hspace=0)
        for i, _image in enumerate([image, mask, add]):
            plt.subplot(1, 3, i+1)
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # _image[:, :, 0] converts shape from 32x32x1 to 32x32
            plt.imshow(_image[:, :, 0], cmap='gray')
        if text:
            plt.title(text)
        plt.show()
    # }}}

    def plot_all(self, **kwargs): # {{{
        for image, mask, add, digit, predicted_digit in zip(self.images, self.masks, self.adds, self.digits, self.predicted_digits):
            text = 'success' if predicted_digit == digit else f'fail: {predicted_digit}'
            self._plot_fig(image, mask, add, text=text, **kwargs)
    # }}}

#! why here?
class Sampler: # {{{

    def __init__(self, data='test'):
        if 'test' == data:
            _, _, self.images, self.digits = load_mnist()
        else:
            self.images, self.digits, _, _ = load_mnist()
        self.images_by_digit = [[] for _ in range(10)]
        for image, digit in zip(self.images, self.digits):
            self.images_by_digit[digit].append(image)
        for digit in range(10):
            self.images_by_digit[digit] = np.array(self.images_by_digit[digit])

    def sample(self, index):
        return self.images[index], self.digits[index]

    def sample_by_digit(self, digit, index):
        images = self.images_by_digit[digit][index]
        return images, [digit] * len(images)
# }}}

def acc(x, y):
    return np.sum(x==y, axis=1)


def evaluate_model(G):
    s = Sampler()
    e = Evaluator(G)
    hit_amount, hit_rate = np.zeros((10, 10)), np.zeros((10, 10))
    for input_number in range(10):
        for target_number in range(10):
            imgs = np.array(s.set[input_number])
            e.run(imgs, [target_number]*len(imgs))
            score = e.score()
            hit_amount[input_number, target_number] = score['hit_amount']
            hit_rate[input_number, target_number] = score['hit_rate']
    total_score = np.sum(hit_amount)
    for i in range(10): # substract hits on the same number
        total_score -= hit_amount[i, i]
    return hit_amount, hit_rate, total_score

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


def plot_matrix(m, save_dir=None):
    fig, ax = plt.subplots()
    ax.matshow(m, cmap=plt.cm.Blues)
    ax.xaxis.set_ticks(np.arange(0, 10, 1))
    ax.yaxis.set_ticks(np.arange(0, 10, 1))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            color = 'white' if m[i, j] > .5 else 'black'
            ax.text(j, i, f'{m[i,j]:.2f}'.lstrip('0'), va='center', ha='center', color=color)
    if save_dir:
        fig.savefig(save_dir)
