# MNIST Neural Network in C

## Overview

Simple NN implementation in C for MNIST dataset. Program allows you to train the dataset and validate it with the test images from the MNIST DB, as well as draw your own numbers in terminal emulator via 'draw.c'.

## Getting Started

### Prerequisites

- GCC
- Make (if you want)
- A basic knowledge of C and neural networks (if you wish to change things like alpha, epochs, no improve functions, etc.)

### Getting the datasets

The files are too large for me to put on this repo.

You can download the datasets I used from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset?select=t10k-labels.idx1-ubyte). Place the images (both training & validation) in imgs/ and the labels in labels/. 

### Installation & Usage

Clone this repo to your local machine:
`git clone https://www.github.com/masonpynt/neural-net`

Then download and place the datasets in the required folders as discussed above.

Run `make` in the root folder to build both the neural-network and the draw program (or you can run `make neural` or `make draw` to build either independently).

Run `./neural` to train the network from scratch (or validate using the neural network included in the repo, which is boring).

Run `./draw trained_model.bin` to run the drawing program. This will allow you to draw a number in your terminal and check what the NN thinks you drew.
