#include "backprop.h"
#include "layer.h"
#include "neuron.h"
#include <endian.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
  *labels = malloc(*num_labels * sizeof(uint8_t));
  fread(*labels, sizeof(uint8_t), *num_labels, file);
  fclose(file);
}

int main(void) {
  float **images;
  int8_t *labels;
  int num_images, num_labels;
  load_images("/home/mason/Dev/neural-net/imgs/t10k-images.idx3-ubyte", &images,
              &num_images);
  load_labels("/home/mason/Dev/neural-net/labels/t10k-labels-idx1-ubyte",
              &labels, &num_labels);

  //  void vectorize_labels(int num_labels, int8_t *labels,
  //                        float **desired_outputs) {
  //    for (int i = 0; i < num_labels; i++) {
  //      desired_outputs[i] = calloc(10, sizeof(float));
  //      desired_outputs[i][labels[i]] = 1.0;
  //    }
  //  }

  void feed_input(int idx, float **images) {
    for (int j = 0; j < 784; j++) {
      lay[0].neu[j].actv = images[idx][j];
    }
  }

void forward_prop(void) {
    int i, j, k;
    for (i = 1; i < num_layers; i++) {
      for (j = 0; j < num_neurons[i]; j++) {
        lay[i].neu[j].z = lay[i].neu[j].bias;
        for (k = 0; k < num_neurons[i - 1]; k++) {
          lay[i].neu[j].z += lay[i - 1].neu[k].out_weights[j] * lay[i - 1].neu[k].actv;
        }
        printf("Input to Activation Function[%d][%d] = %f\n", i, j, lay[i].neu[j].z); // Log input
        lay[i].neu[j].actv = (i < num_layers - 1)
                                 ? ((lay[i].neu[j].z < 0) ? 0 : lay[i].neu[j].z)
                                 : 1 / (1 + exp(-lay[i].neu[j].z));
        printf("Activation[%d][%d] = %f\n", i, j, lay[i].neu[j].actv); // Check activations
      }
    }
  }

  void back_prop(int p) {
    int i, j, k;
    for (i = num_layers - 1; i >= 0; i--) {
      for (j = 0; j < num_neurons[i]; j++) {
        lay[i].neu[j].dz = (lay[i].neu[j].actv - desired_outputs[p][j]) *
                           lay[i].neu[j].actv * (1 - lay[i].neu[j].actv);
 //       printf("Delta z[%d][%d] = %f\n", i, j,
//             lay[i].neu[j].dz); // Debug statement
        for (k = 0; k < num_neurons[i - 1]; k++) {
          lay[i - 1].neu[k].dw[j] = lay[i].neu[j].dz * lay[i - 1].neu[k].actv;
          //          printf("Weight derivative[%d][%d][%d] = %f\n", i - 1, k,
          //          j,
          //                lay[i - 1].neu[k].dw[j]); // Debug statement
          lay[i - 1].neu[k].dactv +=
              lay[i - 1].neu[k].out_weights[j] * lay[i].neu[j].dz;
        }
        lay[i].neu[j].dbias = lay[i].neu[j].dz;
//        printf("Bias derivative[%d][%d] = %f\n", i, j,
//               lay[i].neu[j].dbias); // Debug statement
      }
    }
  }

  void update_weights(void) {
    int i, j, k;
    for (i = 0; i < num_layers - 1; i++) {
      for (j = 0; j < num_neurons[i]; j++) {
        for (k = 0; k < num_neurons[i + 1]; k++) {
          lay[i].neu[j].out_weights[k] -= alpha * lay[i].neu[j].dw[k];
        }
      }
    }
  }

  int create_architecture(void) {
    printf("start arch\n");
    int i, j;
    lay = (layer *)malloc(num_layers * sizeof(layer));
    if (lay == NULL) {
      fprintf(stderr, "Memory allocation failed for layers\n");
      exit(EXIT_FAILURE);
    }
    for (i = 0; i < num_layers; i++) {
      printf("Enter loop arch\n");
      lay[i] = create_layer(num_neurons[i]);
      lay[i].num_neu = num_neurons[i];
      for (j = 0; j < num_neurons[i]; j++) {
        if (i < (num_layers - 1)) {
          lay[i].neu[j] = create_neuron(num_neurons[i + 1]);
        }
      }
    }
    printf("Before finish\n");
    return CREATION_ARCH_SUCCESS;
  }

  int initialize_weights(void) {
    if (lay == NULL) {
      printf("No layers in NN.\n");
      return ERR_INIT_WEIGHTS;
    }
    printf("Init weights...\n");
    for (int i = 0; i < num_layers - 1; i++) {
      for (int j = 0; j < num_neurons[i + 1]; j++) {
        lay[i].neu[j].out_weights =
            (float *)malloc(num_neurons[i] * sizeof(float));
        lay[i].neu[j].dw = (float *)malloc(num_neurons[i] * sizeof(float));
        if (lay[i].neu[j].out_weights == NULL || lay[i].neu[j].dw == NULL) {
          printf("Failed to allocate deriv or ow\n");
          fprintf(stderr, "Memory allocation failed for neuron weights\n");
          exit(EXIT_FAILURE);
        }
        for (int k = 0; k < num_neurons[i]; k++) {
          lay[i].neu[j].out_weights[k] =
              ((double)rand() / RAND_MAX - 0.5) * 0.1;
          lay[i].neu[j].dw[k] = 0.0;
          printf("Initial weight[%d][%d][%d] = %f\n", i, j, k,
                 lay[i].neu[j].out_weights[k]); // Debug statement
        }
        lay[i].neu[j].bias = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        printf("Initial bias[%d][%d] = %f\n", i, j,
               lay[i].neu[j].bias); // Debug statement
      }
    }
    return SUCCESS_INIT_WEIGHTS;
  }

  void compute_cost(int idx) {
    cost[idx] = 0.0;
    for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
      float error = desired_outputs[idx][j] - lay[num_layers - 1].neu[j].actv;
      cost[idx] += 0.5 * error * error;
    }
    full_cost += cost[idx];
  }
  void train_nn(float **images) {
    printf("start training\n");
    int it, i;
    for (it = 0; it < 20000; it++) {
      full_cost = 0.0;
      for (i = 0; i < num_training_ex; i++) {
        feed_input(i, images); // Pass images here
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

  desired_outputs = malloc(num_labels * sizeof(float *));
  for (int i = 0; i < num_labels; i++) {
    desired_outputs[i] = calloc(10, sizeof(float));
    desired_outputs[i][labels[i]] = 1.0;
  }

  if (create_architecture() != CREATION_ARCH_SUCCESS) {
    printf("Failed to create architecture\n");
    return ERR_CREATE_ARCHITECTURE;
  }

  if (initialize_weights() != SUCCESS_INIT_WEIGHTS) {
    printf("Failed to initialize weights\n");
    return ERR_INIT_WEIGHTS;
  }

  cost = (float *)malloc(num_neurons[num_layers - 1] * sizeof(float));
  num_training_ex = num_images;
  if (num_images != num_labels) {
    printf("Number of images and labels do not match\n");
    exit(EXIT_FAILURE);
  }

  train_nn(images);

  // Free resources here

  return 0;
}
