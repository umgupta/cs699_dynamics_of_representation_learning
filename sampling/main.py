"""demonstrating some utilties in the starter code"""
import argparse
import os

import jax
import matplotlib.image
import matplotlib.pyplot
import numpy

import NPEET.npeet.entropy_estimators
from utils.metrics import get_discretized_tv_for_image_density
from utils.density import continuous_energy_from_image, prepare_image, sample_from_image_density

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", type=str, default="results")

    args = parser.parse_args()

    key = jax.random.PRNGKey(0)
    os.makedirs(f"{args.result_folder}", exist_ok=True)

    # load some image
    img = matplotlib.image.imread('./data/labrador.jpg')

    # plot and visualize
    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    matplotlib.pyplot.show()

    # convert to energy function
    # first we get discrete energy and density values
    density, energy = prepare_image(
        img, crop=(10, 710, 240, 940), white_cutoff=225, gauss_sigma=3, background=0.01
    )

    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(density)
    matplotlib.pyplot.show()
    fig.savefig(f"{args.result_folder}/labrador_density.png")

    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(energy)
    matplotlib.pyplot.show()
    fig.savefig(f"{args.result_folder}/labrador_energy.png")

    # create energy fn and its grad
    x_max, y_max = density.shape
    xp = jax.numpy.arange(x_max)
    yp = jax.numpy.arange(y_max)
    zp = jax.numpy.array(density)

    # You may use fill value to enforce some boundary conditions or some other way to enforce boundary conditions
    energy_fn = lambda coord: continuous_energy_from_image(coord, xp, yp, zp, fill_value=0)
    energy_fn_grad = jax.grad(energy_fn)

    # NOTE: JAX makes it easy to compute fn and its grad, but you can use any other framework.

    num_samples = 100000

    # generate samples from true distribution
    key, subkey = jax.random.split(key)
    samples = sample_from_image_density(num_samples, density, subkey)

    # (scatter) plot the samples with image in background
    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(numpy.array(samples)[:, 1], numpy.array(samples)[:, 0], s=0.5, alpha=0.5)
    ax.imshow(density, alpha=0.3)
    matplotlib.pyplot.show()
    fig.savefig(f"{args.result_folder}/labrador_sampled.png")

    # generate another set of samples from true distribution, to demonstrate comparisons
    key, subkey = jax.random.split(key)
    second_samples = sample_from_image_density(num_samples, density, subkey)

    # We have samples from two distributions. We use NPEET package to compute kldiv directly from samples.
    # NPEET needs nxd tensors
    kldiv = NPEET.npeet.entropy_estimators.kldiv(samples, second_samples)
    print(f"KL divergence is {kldiv}")

    # TV distance between discretized density
    # The discrete density bin from the image give us a natural scale for discretization.
    # We compute discrete density from sample at this scale and compute the TV distance between the two densities
    tv_dist = get_discretized_tv_for_image_density(
        numpy.asarray(density), numpy.asarray(samples), bin_size=[7, 7]
    )
    print(f"TV distance is {tv_dist}")
