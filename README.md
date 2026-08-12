# MNIST Zero GAN

This repo contains a cleaned project version of the original `GANs.ipynb` notebook from the deep learning source folder. It trains a compact dense GAN on the zero class from MNIST and keeps the workflow small enough to run on a CPU laptop.

## What This Project Does

The notebook trains two networks against each other:

- a generator that turns random latent vectors into `28x28` grayscale digits
- a discriminator that tries to separate real zero digits from synthetic ones

The result is a baseline adversarial image generation project that shows the core GAN loop, training curves, and generated samples over time.

## Dataset Notes

- Dataset file: `data/mnist.pkl.gz`
- Source: the classic MNIST handwritten digit dataset stored in a local pickle archive
- This project uses only the zero class so training stays light and the learning signal is easier to inspect

## Model And Approach

- Dense generator with a `64`-dimensional latent space
- Dense discriminator with dropout for basic regularization
- Binary cross-entropy losses for both players
- Adam optimizers with `beta_1=0.5`, a common GAN setting
- One-sided label smoothing on real samples to make the discriminator slightly less brittle

## How To Run

1. Install the dependencies from `requirements.txt`.
2. Keep `data/mnist.pkl.gz` in the repo as provided.
3. Open `mnist-zero-gan.ipynb`.
4. Run the notebook from a clean kernel.

The notebook will:

- load zero digits from MNIST
- train the dense GAN for `20` epochs
- plot discriminator and generator trends
- show generated samples at a few checkpoints
- print a few lightweight sanity-check metrics at the end

## Important Results

On the executed notebook saved in this repo, the model learns the rough circular structure of handwritten zeros and shifts more pixel mass toward the center of the image than the border. The final samples are still soft and imperfect, which is expected for a dense CPU-friendly GAN baseline.
