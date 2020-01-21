from math import ceil

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from .util import load_mnist, onehot












if '__main__' == __name__:

    batch_size = 20
    e = Evaluator('./outputs/19_inception_G1D10_model_100000iter/models/G40000.hdf5')
    s = Sampler(batch_size)
    imgs, labels = s.sample_by_key(key=1)

    c.add_imgs(imgs, labels, [4]*len(imgs))
    res = c.score()
    print(res)
