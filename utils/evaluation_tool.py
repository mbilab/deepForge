from math import ceil

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from .util import load_mnist, onehot


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
        onehot_digits = onehot(np.array(digits), 10)
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


if '__main__' == __name__:

    batch_size = 20
    e = Evaluator('./outputs/19_inception_G1D10_model_100000iter/models/G40000.hdf5')
    s = Sampler(batch_size)
    imgs, labels = s.sample_by_key(key=1)

    c.add_imgs(imgs, labels, [4]*len(imgs))
    res = c.score()
    print(res)
