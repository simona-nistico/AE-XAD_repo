import numpy as np
import matplotlib.pyplot as plt
import os

import warn
from scipy.ndimage import gaussian_filter


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


def rad_from_lines(gt, d_thres=3):
    nc = gt.shape[1]
    gtrow = gt.cumsum(axis=1) * gt
    drow = gtrow[np.where(gtrow[:, 0:nc - 1] > gtrow[:, 1:nc])]
    drow = np.concatenate((drow, gtrow[gtrow[:, nc - 1] > 0, nc - 1]))
    drow = drow[drow > d_thres]
    rrow = 0.5 * drow.mean()
    nr = gt.shape[0]
    gtcol = gt.cumsum(axis=0) * gt
    dcol = gtcol[np.where(gtcol[0:nr - 1, :] > gtcol[1:nr, :])]
    dcol = np.concatenate((dcol, gtcol[nr - 1, gtcol[nr - 1, :] > 0]))
    dcol = dcol[dcol > d_thres]
    rcol = 0.5 * dcol.mean()
    return min(rcol, rrow)


def get_filter_sigma(ht, thres=0.5, scale=0.25):
    if ht.max() > 1:
        warn.warn('Error: heatmap not in [0,1]')
    binht = np.where(ht >= thres, 1, 0)
    rad = rad_from_lines(binht)
    return max(rad * scale, 1)


def blurred_htmaps(hts, thres=0.5, scale=0.25):
    hts_b = np.empty_like(hts)
    for i in range(hts.shape[0]):
        sigma = get_filter_sigma(hts[i], thres=thres, scale=scale)
        hts_b[i] = gaussian_filter(hts[i], sigma=sigma)

    return hts_b
