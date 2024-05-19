import numpy as np
import matplotlib.pyplot as plt
import os

from tools.create_dataset import square, square_diff


def plot_image(image, cmap='gray'):
    '''
    Method to plot an image
    :param image: (C x H x W) channel-first-format image
    :return:
    '''
    if image.shape[0] == 1:
        image = image[0, :, :]
    else:
        image = np.swapaxes(image, 0, 1)
        image = np.swapaxes(image, 1, 2)

    plt.imshow(image, cmap=cmap)
    plt.show()


def plot_heatmap(image, colorbar=True):
    '''
    Method to plot a heatmap
    :param image: (1 x H x W) channel-first-format image
    :return:
    '''
    image = image[0, :, :]

    plt.imshow(image, cmap='jet')#, vmin=0.0, vmax=1.0)
    if colorbar:
        plt.colorbar()
    plt.show()


def plot_results(X, GT, htmaps, ids):
    '''
    Plot the results for the first 5 samples ids. The first row show the images, the second shows the heatmaps given by the network,
    the last one plots the ground truths.
    :param X: ndarray (n_samples x C x H x W), images
    :param GT: ndarray (n_samples x C x H x W), ground truths
    '''
    plt.figure(figsize=(9,15))
    for i in range(5):
        plt.subplot(5, 3, (i * 3) + 1)
        plt.tight_layout()
        plt.axis(False)
        plt.imshow(X[ids[i]])
        plt.subplot(5, 3, (i * 3) + 2)
        plt.tight_layout()
        plt.axis(False)
        plt.imshow(htmaps[ids[i]][0], cmap='jet', vmin=htmaps[ids[i]][0].min(), vmax=htmaps[ids[i]][0].max())
        #plt.colorbar()
        plt.subplot(5, 3, (i * 3) + 3)
        plt.tight_layout()
        plt.axis(False)
        plt.imshow(GT[ids[i]][0], cmap='jet')
    plt.show()