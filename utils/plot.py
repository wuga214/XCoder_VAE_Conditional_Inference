import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
import os

from moviepy.editor import ImageSequenceClip
from scipy.ndimage import imread
from scipy import stats
from scipy.stats import multivariate_normal
from scipy.interpolate import griddata
from scipy.stats import bernoulli

from mpl_toolkits.axes_grid1 import ImageGrid


def show_faces(g, batch_size, dim_size, name, n=10, save=True, folder='vae_canvas'):
    """
    Creates and saves a 'canvas' of images decoded from a grid in the latent space.
    :param vae: instance of VAE which performs decoding
    :param batch_size: little hack to get dimensions right
    :param epoch: current epoch of training
    :param n: number of points in each dimension of grid
    :param bound: samples from [-bound, bound] in both z1 and z2 direction
    """
    # create grid (could be done once but requires refactoring)
    noise = np.random.normal(size=(batch_size, dim_size))
    images = g.e2x(noise).reshape([-1, 64, 64, 3])
    images = (images+1)/2

    # create and fill canvas
    canvas = np.empty((64 * n, 64 * n, 3))
    for i in range(n):
        for j in range(n):
            canvas[(n - i - 1) * 64:(n - i) * 64, j * 64:(j + 1) * 64, 0:3] = images[i*n+j]

    # make figure and save
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if save:
        plt.savefig('figs/{0}/{1}.pdf'.format(folder, name), format='pdf')
    else:
        plt.show()


def show_faces_with_mask(g, mask, evidence, batch_size, dim_size, name, n=10, save=True, folder='vae_canvas'):
    """
    Creates and saves a 'canvas' of images decoded from a grid in the latent space.
    :param vae: instance of VAE which performs decoding
    :param batch_size: little hack to get dimensions right
    :param epoch: current epoch of training
    :param n: number of points in each dimension of grid
    :param bound: samples from [-bound, bound] in both z1 and z2 direction
    """
    # create grid (could be done once but requires refactoring)
    noise = np.random.normal(size=(batch_size, dim_size))
    images = g.e2x(noise).reshape([-1, 64, 64, 3])
    images = images*(1-mask) + evidence*mask
    images = (images + 1) / 2

    # create and fill canvas
    canvas = np.empty((64 * n, 64 * n, 3))
    for i in range(n):
        for j in range(n):
            canvas[(n - i - 1) * 64:(n - i) * 64, j * 64:(j + 1) * 64, 0:3] = images[i*n+j]

    # make figure and save
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if save:
        plt.savefig('figs/{0}/{1}.pdf'.format(folder, name), format='pdf')
    else:
        plt.show()


def show_vae_samples(g, n, bound=3):
    # create grid (could be done once but requires refactoring)
    spaced_z1 = np.linspace(-bound, bound, n)
    spaced_z2 = np.linspace(-bound, bound, n)

    # create and fill canvas
    noise = []
    canvas = np.empty((28 * n, 28 * n))
    for i, z1 in enumerate(spaced_z1):
        for j, z2 in enumerate(spaced_z2):
            noise.append([z1, z2])

    imgs = g.e2x(np.array(noise))
    for i, z1 in enumerate(spaced_z1):
        for j, z2 in enumerate(spaced_z2):
            canvas[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = imgs[i*n+j].reshape([28, 28])

    # make figure and save
    plt.figure(figsize=(20, 20))
    plt.imshow(canvas, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def show_mnist(g, batch_size, dim_size, name, n=10, save=True):
    # create grid (could be done once but requires refactoring)
    noise = np.random.normal(size=(batch_size, dim_size))
    imgs = g.e2x(np.array(noise))
    canvas = np.empty((28 * n, 28 * n))
    for i in range(n):
        for j in range(n):
            canvas[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = imgs[i*n+j].reshape([28, 28])

    # make figure and save
    plt.figure(figsize=(20, 20))
    plt.imshow(canvas, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if save:
        plt.savefig('figures/vae_canvas/' + name + '.pdf', format='pdf')
    else:
        plt.show()

def show_single_mnist(image_vector, save=False, name='partial', folder='autofill'):
    fig = plt.figure()
    images = bernoulli.rvs(image_vector, 784).reshape(28, 28)
    plt.imshow(images, cmap='gray')
    plt.axis('off')
    if save:
        plt.tight_layout()
        plt.savefig('figs/{0}/{1}.pdf'.format(folder, name), bbox_inches="tight", pad_inches=0, format='pdf')
    else:
	plt.show()

def show_mnist2(images, row, col, name="Unknown", save=True, folder='canvas', noise=True, exception=None):
    num_images = row*col
    fig = plt.figure(figsize=(col, row))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(row, col),
                     axes_pad=0.1)
    for i in xrange(num_images):
        if noise == True and (i is not exception or exception is None):
            im = images[i].reshape(28, 28)
            im[im < 0.5] = 0
            im[im >= 0.5] = 1
        else:
            im = images[i].reshape(28, 28)
        axis = grid[i]
        axis.axis('off')
        axis.imshow(im, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    if save:
        fig.savefig('figs/{0}/{1}.eps'.format(folder, name), bbox_inches="tight", pad_inches=0, format='eps')
    else:
        plt.show()


def show_faces_masks(images, row, col, name="Unknown", save=True, folder='canvas'):
    num_images = row*col
    fig = plt.figure(figsize=(col, row))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(row, col),
                     axes_pad=0.1)
    for i in xrange(num_images):
        im = images[i].reshape(64, 64, 3)
        im = (im + 1.)/2.
        axis = grid[i]
        axis.axis('off')
        axis.imshow(im)
    plt.axis('off')
    plt.tight_layout()
    if save:
        fig.savefig('figs/{0}/{1}.pdf'.format(folder, name), bbox_inches="tight", pad_inches=0, format='pdf')
    else:
        plt.show()
