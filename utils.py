import datetime
import matplotlib
import pickle
import numpy as np

from tensorflow.python.client import device_lib

matplotlib.use('Agg')  # Force not to use $DISPLAY
import matplotlib.pyplot as plt

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def get_date():
    return datetime.datetime.now().strftime('%02m-%02d_%02H-%02M-%02S')


def unpickle(filename):
    handler = open(filename, 'rb')
    data = pickle.load(handler, encoding='latin1')
    handler.close()
    return data


def save_images(images, labels, dis_outputs, step, params):
    h, w = params.show_h_images, params.show_w_images
    fig, axs = plt.subplots(h, w, figsize=(params.show_figsize, params.show_figsize))
    for ax in np.asarray(axs).flatten():
        ax.xaxis.set_ticks([]), ax.yaxis.set_ticks([])

    data = list(zip(images, labels, dis_outputs))

    # Sort by discriminator output in decreasing order.
    data = sorted(data, key=lambda k: -k[2])

    for i, (img, label, dis_output) in enumerate(data[:h * w]):
        y, x = i // w, i % w
        axs[y, x].set_title("%s (%.3f)" % (params.labels_names[label], dis_output))
        axs[y, x].imshow(np.squeeze(img))
    fig.savefig('%s/%d.png' % (params.images_dir, step))
    plt.close()


def onehot(x, label_size):
    onehot_x = np.zeros((x.shape[0], label_size))
    onehot_x[np.arange(x.shape[0]), x] = 1
    return onehot_x
