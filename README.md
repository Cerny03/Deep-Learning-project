# Image Denoising with Autoencoders

## Overview

This project studies image denoising on Fashion-MNIST using classical filtering and neural autoencoders.

The main objective is not only to reconstruct clean images from corrupted inputs, but also to investigate:

- how fully connected and convolutional inductive biases affect denoising;
- how the models behave under noise levels not seen during training;
- whether the models generalize to different corruption mechanisms;
- how noise and reconstruction affect the geometry and semantic organization of the latent representations.

The project compares:

1. the noisy input;
2. a fixed $3 \times 3$ Gaussian filter;
3. a fully connected denoising autoencoder;
4. a convolutional denoising autoencoder.

Both neural models are trained with Gaussian noise of standard deviation

$$
\sigma_{\text{train}} = 0.3.
$$

---

## Research questions

The experiments address the following questions:

1. How do classical filtering, a fully connected autoencoder, and a convolutional autoencoder compare in terms of reconstruction quality?
2. How robust are the trained models to Gaussian noise intensities that differ from the training condition?
3. Do the models generalize to unseen corruption mechanisms such as salt-and-pepper noise and random masking?
4. Does denoising recover the latent representation of the corresponding clean image?
5. Is the semantic and local neighborhood structure of the latent space preserved after corruption and reconstruction?

---

## Dataset

The project uses **Fashion-MNIST**, which contains:

- 60,000 training images;
- 10,000 test images;
- grayscale images of shape $1 \times 28 \times 28$;
- 10 clothing categories.

Pixel values are converted to `float32` tensors in the interval [0,1].

The class labels are not used to train the denoising models. They are used only in the final representation analysis.

---

## Installation

Clone the repository

Open a terminal and clone the project:
```bash
git clone https://github.com/Cerny03/Deep-Learning-project.git
```

Move into the project directory:
```bash
cd Deep-Learning-project
```

Create and activate a virtual environment or Conda environment.
Example with Conda:
```bash
conda create -n dl_denoising python=3.12 -y
conda activate dl_denoising
```

Install the required packages:
```bash
python -m pip install -r requirements.txt
```

To run the entire project pipeline in the correct order, simply execute:
```bash
python run_project.py
```

---

## Main conclusions

1. Both neural autoencoders substantially outperform the noisy input and the classical Gaussian filter under the training corruption.
2. The convolutional autoencoder achieves the best standard denoising performance with far fewer parameters.
3. The advantage of the convolutional architecture is not universal and disappears under severe Gaussian noise and random masking.
4. Denoising is most effective near the corruption distribution used during training.
5. Better pixel reconstruction does not necessarily imply better semantic or geometric recovery in latent space.
6. The convolutional representation is more stable under Gaussian corruption, while the fully connected autoencoder produces a stronger latent recovery after reconstruction.
