import numpy as np
import utils

from tensorflow.examples.tutorials.mnist import input_data


class Dataset(object):
    def __init__(self, path, name):
        super(Dataset, self).__init__()
        self.name = name
        self.path = path


class MnistDataset(Dataset):
    def __init__(self, path='.', name='mnist'):
        super(MnistDataset, self).__init__(path, name)
        self.image_size = 24
        self.channels_size = 1
        self.labels_size = 10
        self.labels_names = [str(i) for i in range(10)]

    def get(self, params):
        mnist = input_data.read_data_sets(self.path, one_hot=True)
        images = mnist.train.images.astype(np.float32) * 2 - 1
        labels = mnist.train.labels.astype(np.float32)
        images = np.reshape(images, [-1, 28, 28, 1])
        # Crop 24x24 sub-image.
        images = images[:, 2:26, 2:26, :]
        params.labels_names = self.labels_names
        return images, labels


class Cifar10Dataset(Dataset):
    def __init__(self, path='.', name='cifar10'):
        super(Cifar10Dataset, self).__init__(path, name)
        self.image_size = 32
        self.channels_size = 3
        self.labels_size = 10
        self.labels_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"]

    def _load(self, filenames):
        images, labels = None, []
        for i, filename in enumerate(filenames):
            datafile = utils.unpickle(filename)
            if i == 0:
                images = datafile['data']
            else:
                images = np.append(images, datafile['data'], axis=0)
            labels.extend(datafile['labels'])
            print(images.shape, len(labels))
        return images, utils.onehot(np.asarray(labels), label_size=self.labels_size)

    def get(self, params):
        params.labels_names = self.labels_names
        filenames = ['%s/data_batch_%d' % (self.path, i) for i in range(1, 6)]
        images, labels = self._load(filenames)
        images = (images.astype(np.float32) / 255) * 2 - 1
        labels = labels.astype(np.float32)
        images = np.reshape(images, [-1, self.channels_size, self.image_size, self.image_size])
        images = np.transpose(images, (0, 2, 3, 1))
        return images, labels
