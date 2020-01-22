# standard import
from math import ceil
import os

# third-part import
from keras.datasets import mnist
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np


class Evaluator:

    def __init__(self, gen_mask_model, mnist_model='./outputs/D_digit.hdf5'):
        self.G_mask = load_model(gen_mask_model)
        self.clf = load_model(mnist_model)

    def classify(self, images):
        if images.ndim < 4:
            images = images[np.newaxis, :]
        clf_prob = self.clf.predict(images)
        clf_res = np.argmax(clf_prob, axis=1)
        return clf_res

    def run(self, images, digits):
        self.images = images
        self.digits = np.full(len(images), digits) if int == type(digits) else digits
        onehot_digits = onehot(np.array(self.digits), 10)
        self.masks = self.G_mask.predict([images, onehot_digits])
        self.adds = np.clip(self.masks + images, 0, 1)
        self.predicted_digits = self.classify(self.adds)


    def score(self): # {{{
        hits = self.predicted_digits == self.digits
        return {
            '#samples': len(self.images),
            '#hits': np.sum(hits),
            '%hit': np.mean(hits),
        }
    # }}}

    def _plot_fig(self, image, mask, add, dpi=96, wspace=.02): # {{{
        fig = plt.figure(dpi=dpi, figsize=(96./dpi, 32./dpi))
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=wspace, hspace=0)
        for i, _image in enumerate([image, mask, add]):
            ax = fig.add_subplot(1, 3, i+1)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # _image[:, :, 0] converts shape from 32x32x1 to 32x32
            ax.imshow(_image[:, :, 0], cmap='gray')
        return fig
    # }}}

    def plot_all(self, pad_inches=None, **kwargs): # {{{
        n = 0
        for image, mask, add, digit, predicted_digit in zip(self.images, self.masks, self.adds, self.digits, self.predicted_digits):
            text = 'success' if predicted_digit == digit else f'fail: {predicted_digit}'
            fig = self._plot_fig(image, mask, add, **kwargs)
            print(text)
            fig.show()
            # fig.savefig(f'./tmp/image_{n:03}.png')
            n += 1
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

    def sample(self, index=None):
        if index is None:
            return self.images, self.digits
        return self.images[index], self.digits[index]

    def sample_by_digit(self, digit, index=None):
        if index is None:
            images = self.images_by_digit[digit]
        else:
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
            images, digits = s.sample_by_digit(input_number)
            e.run(images, digits)
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
    _, _, images, digits = load_mnist()
    input_images = [] # images of 0 ~ 9
    for i in range(10):
        images_filtered = images[np.where(digits == i)[0]]
        if random:
            idx = np.random.randint(0, images_filtered.shape[0])
        else:
            idx = 0
        input_images.append(images_filtered[idx])

    fig, axs = plt.subplots(10, 10)
    fig.set_size_inches(50, 50)
    for i in range(10): # for input image
        for j in range(10): # for target digit
            gen_image = G.predict([np.expand_dims(input_images[i], axis=0), onehot(np.full((1, 1), j), 10)])
            axs[i, j].imshow(gen_image[0, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
    plt.show()

    if save:
        fig.savefig(name, dpi=150)


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
