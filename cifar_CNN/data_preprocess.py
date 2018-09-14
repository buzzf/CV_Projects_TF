import pickle
import numpy as np
import matplotlib.pyplot as plt
import random


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def clean(data):
    # print('data ', data.shape)
    imgs = data.reshape(data.shape[0], 3, 32, 32)
    # print('imgs ', imgs.shape)
    grayscale_imgs = imgs.mean(1)
    # print('grayscal_imgs ', grayscale_imgs.shape)
    cropped_imgs = grayscale_imgs[:, 4:28, 4:28]
    # print('cropped_imgs ', cropped_imgs.shape)
    img_data = cropped_imgs.reshape(data.shape[0], -1)
    # print('img_data ', img_data.shape)
    img_size = np.shape(img_data)[1]
    means = np.mean(img_data, axis=1)
    # print('means ', means.shape)
    meansT = means.reshape(len(means), 1)
    # print('meansT ', meansT.shape)
    stds = np.std(img_data, axis=1)
    # print('stds', stds.shape)
    stdsT = stds.reshape(len(stds), 1)
    # print('stdsT ', stdsT.shape)
    adj_stds = np.maximum(stdsT, 1.0 / np.sqrt(img_size))
    # 标准化处理 z = (x - μ)/σ
    normalized = (img_data - meansT) / adj_stds
    # print('normalized', normalized.shape)
    return normalized


def read_data(directory):
    names = unpickle('{}/batches.meta'.format(directory))['label_names']
    print('names', names)

    data, labels = [], []
    for i in range(1, 6):
        filename = '{}/data_batch_{}'.format(directory, i)
        batch_data = unpickle(filename)
        if len(data) > 0:
            data = np.vstack((data, batch_data['data']))
            # print('data_{}: {}'.format(i, data))
            # print(len(data))
            labels = np.hstack((labels, batch_data['labels']))
            # print('labels_{}: {}'.format(i, labels))
            # print(len(labels))
        else:
            data = batch_data['data']
            labels = batch_data['labels']

    print(np.shape(data), np.shape(labels))

    data = clean(data)
    print(data.shape)
    data = data.astype(np.float32)
    return names, data, labels


def show_some_examples(names, data, labels):
    plt.figure()
    rows, cols = 4, 4
    random_idxs = random.sample(range(len(data)), rows * cols)
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        j = random_idxs[i]
        plt.title(names[labels[j]])
        img = np.reshape(data[j, :], (24, 24))
        plt.imshow(img, cmap='Greys_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('cifar_examples.png')


random.seed(1)

names, data, labels = read_data('./cifar-10-batches-py')
# show_some_examples(names, data, labels)