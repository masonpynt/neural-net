#include "backprop.h"
#include "layer.h"
#include "neuron.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__APPLE__) && defined(__MACH__)
#include <libkern/OSByteOrder.h>
#include <machine/endian.h>
#define htobe16(x) OSSwapHostToBigInt16(x)
#define htole16(x) OSSwapHostToLittleInt16(x)
#define be16toh(x) OSSwapBigToHostInt16(x)
#define le16toh(x) OSSwapLittleToHostInt16(x)
#define htobe32(x) OSSwapHostToBigInt32(x)
#define htole32(x) OSSwapHostToLittleInt32(x)
#define be32toh(x) OSSwapBigToHostInt32(x)
#define le32toh(x) OSSwapLittleToHostInt32(x)
#define htobe64(x) OSSwapHostToBigInt64(x)
#define htole64(x) OSSwapHostToLittleInt64(x)
#define be64toh(x) OSSwapBigToHostInt64(x)
#define le64toh(x) OSSwapLittleToHostInt64(x)
#elif defined(__linux__)
#endif

layer *lay = NULL;
int num_layers = 3;
int num_neurons[] = {784, 128, 10};
float alpha;
float *cost;
float full_cost;
float **input;
float **desired_outputs;
int num_training_ex;
int n = 1;
float alpha = 0.15;

// Read integers

int read_int(FILE *fp) {
  uint32_t integer;
  fread(&integer, sizeof(uint32_t), 1, fp);
  return be32toh(integer);
}

// Load mnist images

void load_images(const char *filename, float ***images, int *num_images) {
  printf("Loading images\n");
  FILE *file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "Failed to open file %s\n", filename);
    return;
  }

  int magic_number = read_int(file);
  if (magic_number != 2051) {
    fprintf(stderr, "Invalid magic number for images: %d\n", magic_number);
    fclose(file);
    return;
  }

  *num_images = read_int(file);
  int rows = read_int(file);
  int cols = read_int(file);

  *images = malloc(*num_images * sizeof(float *));
  if (!*images) {
    fprintf(stderr, "Memory allocation failed\n");
    fclose(file);
    return;
  }

  for (int i = 0; i < *num_images; i++) {
    (*images)[i] = malloc(rows * cols * sizeof(float));
    if (!(*images)[i]) {
      fprintf(stderr, "Memory allocation failed for image %d\n", i);
      for (int k = 0; k < i; k++) {
        free((*images)[k]);
      }
      free(*images);
      fclose(file);
      return;
    }

    for (int j = 0; j < rows * cols; j++) {
      unsigned char pixel;
      fread(&pixel, sizeof(unsigned char), 1, file);
      (*images)[i][j] = pixel / 255.0f; // Normalize pixel value
    }
  }

  fclose(file);
}

void load_labels(const char *filename, int8_t **labels, int *num_labels) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "Failed to open file %s\n", filename);
    return;
  }

  int magic_number = read_int(file);
  if (magic_number != 2049) {
    fprintf(stderr, "Invalid magic number for labels: %d\n", magic_number);
    fclose(file);
    return;
  }

  *num_labels = read_int(file);
  *labels = malloc(*num_labels * sizeof(int8_t));
  if (*labels == NULL) {
    fprintf(stderr, "Memory allocation failed for labels\n");
    fclose(file);
    return;
  }

  if (fread(*labels, sizeof(int8_t), *num_labels, file) != *num_labels) {
    fprintf(stderr, "Failed to read all labels\n");
    free(*labels);
    fclose(file);
    return;
  }

  fclose(file);
}

void train_nn(float **images) {
  printf("start training\n");
  int it, i;
  for (it = 0; it < 20000; it++) {
    full_cost = 0.0;
    for (i = 0; i < num_training_ex; i++) {
      feed_input(i, images);
      forward_prop();
      compute_cost(i);
      back_prop(i);
      update_weights();
    }
    if (it % 1000 == 0) {
      printf("Epoch %d, Cost %f\n", it, full_cost / num_training_ex);
    }
  }
}
int main(void) {
  float **images;
  int8_t *labels;
  int num_images, num_labels;

  // Load images and labels
  load_images("train-images-idx3-ubyte", &images, &num_images);
  load_labels("train-labels-idx1-ubyte", &labels, &num_labels);

  if (num_images != num_labels) {
    fprintf(stderr,
            "Error: Number of images (%d) and labels (%d) do not match\n",
            num_images, num_labels);
    exit(EXIT_FAILURE);
  }

  num_training_ex = num_images;

  // Vectorize labels
  desired_outputs = malloc(num_labels * sizeof(float *));
  if (desired_outputs == NULL) {
    fprintf(stderr, "Error: Memory allocation failed for desired_outputs\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < num_labels; i++) {
    desired_outputs[i] = calloc(10, sizeof(float));
    if (desired_outputs[i] == NULL) {
      fprintf(stderr,
              "Error: Memory allocation failed for desired_outputs[%d]\n", i);
      exit(EXIT_FAILURE);
    }
    desired_outputs[i][labels[i]] = 1.0;
  }

  if (create_architecture() != CREATION_ARCH_SUCCESS) {
    fprintf(stderr, "Error: Failed to create architecture\n");
    return ERR_CREATE_ARCHITECTURE;
  }

  if (initialize_weights() != SUCCESS_INIT_WEIGHTS) {
    fprintf(stderr, "Error: Failed to initialize weights\n");
    return ERR_INIT_WEIGHTS;
  }

  cost = malloc(num_neurons[num_layers - 1] * sizeof(float));
  if (cost == NULL) {
    fprintf(stderr, "Error: Memory allocation failed for cost\n");
    exit(EXIT_FAILURE);
  }

  train_nn(images);

  // Free resources
  for (int i = 0; i < num_images; i++) {
    free(images[i]);
    free(desired_outputs[i]);
  }
  free(images);
  free(desired_outputs);
  free(labels);
  free(cost);

  // Free neural network resources
  for (int i = 0; i < num_layers; i++) {
    for (int j = 0; j < num_neurons[i]; j++) {
      if (i < num_layers - 1) {
        free(lay[i].neu[j].out_weights);
        free(lay[i].neu[j].dw);
      }
    }
    free(lay[i].neu);
  }
  free(lay);

  return 0;
}
