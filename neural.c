#include "neural.h"
#include "cost.h"
#include "layer.h"
#include "neuron.h"
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
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
#include <endian.h>
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
int num_testing_ex;
int n = 1;
float init_alpha = 0.01;
float min_alpha = 0.001;
int decay_steps = 1000;
int total_iterations = 0;
float lambda = 0.001;

int read_int(FILE *fp) {
  uint32_t integer;
  fread(&integer, sizeof(uint32_t), 1, fp);
  return be32toh(integer);
}

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
      (*images)[i][j] = pixel / 255.0f;
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

void feed_input(int idx, float **images) {
  if (idx < 0 || idx >= num_training_ex) {
    fprintf(stderr, "Error: Invalid training example index %d\n", idx);
    return;
  }
  for (int j = 0; j < num_neurons[0] && j < 784; j++) {
    lay[0].neu[j].actv = images[idx][j];
  }
}
void forward_prop(void) {
  int i, j, k;
  for (i = 1; i < num_layers; i++) {
    if (num_neurons[i - 1] != 784 && i == 1) {
      fprintf(stderr, "Error: input layer has %d neurons instead of 784\n",
              num_neurons[i - 1]);
    }

    float max_z = -INFINITY;

    for (j = 0; j < num_neurons[i]; j++) {
      if (num_neurons[i] != 10 && i == num_layers - 1) {
        fprintf(stderr, "Error: output layer has %d neurons instead of 10\n",
                num_neurons[i]);
      }
      lay[i].neu[j].z = lay[i].neu[j].bias;

      for (k = 0; k < num_neurons[i - 1] && k < 784; k++) {
        if (k >= num_neurons[i - 1] || j >= num_neurons[i] ||
            lay[i - 1].neu[k].out_weights == NULL) {
          fprintf(
              stderr,
              "Error: Invalid array access in layer %d, neuron %d, weight %d\n",
              i, j, k);
          exit(EXIT_FAILURE);
        }

        float weight = lay[i - 1].neu[k].out_weights[j];
        float prev_actv = lay[i - 1].neu[k].actv;
        lay[i].neu[j].z += weight * prev_actv;
      }

      if (isnan(lay[i].neu[j].z) || isinf(lay[i].neu[j].z)) {
        fprintf(stderr, "Error: NaN or Inf detected in layer %d, neuron %d\n",
                i, j);
        lay[i].neu[j].z = 0;
      }

      if (i < num_layers - 1) {
        lay[i].neu[j].actv = (lay[i].neu[j].z > 0) ? lay[i].neu[j].z : 0;
      } else {
        if (lay[i].neu[j].z > max_z) {
          max_z = lay[i].neu[j].z;
        }
      }
    }

    if (i == num_layers - 1) {
      float sum_exp = 0.0;
      for (j = 0; j < num_neurons[i]; j++) {
        lay[i].neu[j].actv = exp(lay[i].neu[j].z - max_z);
        sum_exp += lay[i].neu[j].actv;
      }
      for (j = 0; j < num_neurons[i]; j++) {
        lay[i].neu[j].actv /= sum_exp;
        if (lay[i].neu[j].actv < 1e-15)
          lay[i].neu[j].actv = 1e-15;
      }
    }
  }
}
void back_prop(int p) {
  int i, j, k;
  for (i = num_layers - 1; i > 0; i--) {
    for (j = 0; j < num_neurons[i]; j++) {
      if (i == num_layers - 1) {
        lay[i].neu[j].dz = (lay[i].neu[j].actv - desired_outputs[p][j]) *
                           lay[i].neu[j].actv * (1 - lay[i].neu[j].actv);
      } else {
        lay[i].neu[j].dz = 0;
        for (k = 0; k < num_neurons[i + 1]; k++) {
          lay[i].neu[j].dz +=
              lay[i + 1].neu[k].dz * lay[i].neu[j].out_weights[k];
        }
        lay[i].neu[j].dz *= (lay[i].neu[j].actv > 0) ? 1 : 0;
      }
      for (k = 0; k < num_neurons[i - 1]; k++) {
        lay[i - 1].neu[k].dw[j] = lay[i].neu[j].dz * lay[i - 1].neu[k].actv;
      }
      lay[i].neu[j].dbias = lay[i].neu[j].dz;
    }
  }
}
void clip_gradients(float threshold) {
  for (int i = 1; i < num_layers; i++) {
    for (int j = 0; j < num_neurons[i]; j++) {
      if (lay[i].neu[j].dz > threshold)
        lay[i].neu[j].dz = threshold;
      if (lay[i].neu[j].dz < -threshold)
        lay[i].neu[j].dz = -threshold;
    }
  }
}

void update_weights(void) {
  float current_alpha = fmax(
      init_alpha * pow(0.1, (float)total_iterations / decay_steps), min_alpha);
  for (int i = 0; i < num_layers - 1; i++) {
    for (int j = 0; j < num_neurons[i]; j++) {
      for (int k = 0; k < num_neurons[i + 1]; k++) {
        float update = current_alpha * lay[i].neu[j].dw[k];
        float l2_reg = current_alpha * lambda * lay[i].neu[j].out_weights[k];
        float total_update = update + l2_reg;
        if (isnan(total_update) || isinf(total_update)) {
          printf("NaN/Inf detected in weight update: layer=%d, neuron=%d, "
                 "weight=%d, value=%f\n",
                 i, j, k, total_update);
          continue;
        }
        lay[i].neu[j].out_weights[k] -= update;
      }
    }
  }
  total_iterations++;
}

int backup_file(const char *original_filename, const char *backup_filename) {
  if (rename(original_filename, backup_filename) != 0) {
    fprintf(stderr, "Error backing up file: %s\n", strerror(errno));
    return -1;
  }
  printf("Existing model backed up to %s\n", backup_filename);
  return 0;
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
      lay[i].neu[j] =
          create_neuron(i < num_layers - 1 ? num_neurons[i + 1] : 0);
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

  return SUCCESS_INIT_WEIGHTS;
}

void compute_cost(int idx) {
  if (idx < 0 || idx >= num_training_ex) {
    fprintf(stderr,
            "Error: Invalid index %d in compute_cost (num_training_ex: %d)\n",
            idx, num_training_ex);
    return;
  }

  cost[idx] = 0.0;
  for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
    float desired = desired_outputs[idx][j];
    float actual = lay[num_layers - 1].neu[j].actv;
    float error = desired - actual;
    cost[idx] += 0.5 * error * error;
  }

  if (isnan(cost[idx]) || isinf(cost[idx])) {
    printf("Warning: NaN/Inf detected in cost for example %d\n", idx);
    cost[idx] = 0.0;
  }

  full_cost += cost[idx];
}

int train_nn(float **images, const char *model_filename) {
  printf("start training\n");
  int it, i;
  float min_delta = 0.0001;
  int patience = 10;

  for (it = 0; it < 100; it++) {
    full_cost = 0.0;
    for (i = 0; i < num_training_ex; i++) {
      if (i % 10000 == 0) {
        printf("Training example %d of epoch %d\n", i, it);
      }
      feed_input(i, images);
      forward_prop();
      compute_cost(i);
      back_prop(i);
      update_weights();
      if (i % 10000 == 0) {
        printf("Example %d completed\n", i);
      }
    }
    float avg_cost = full_cost / num_training_ex;
    printf("Epoch %d, Avg Cost %f\n", it, avg_cost);
    if (check_early_stopping(avg_cost, it, min_delta, patience)) {
      printf("Training stopped early at epoch %d\n", it);
      break;
    }
  }
  printf("Saving trained model...\n");
  int save_neural = save_nn(model_filename);
  if (save_neural != SAVE_NEURAL_SUCCESS) {
    printf("Failed to save the model.\n");
    return ERR_SAVE_NEURAL;
  }
  return TRAIN_NEURAL_SUCCESS;
}

int save_nn(const char *filename) {
  FILE *file = fopen(filename, "wb");
  if (file == NULL) {
    fprintf(stderr, "Failed to open file %s for writing\n", filename);
    return ERR_SAVE_NEURAL;
  }

  fwrite(&num_layers, sizeof(int), 1, file);
  fwrite(num_neurons, sizeof(int), num_layers, file);

  for (int i = 0; i < num_layers; i++) {
    for (int j = 0; j < num_neurons[i]; j++) {
      fwrite(&lay[i].neu[j].bias, sizeof(float), 1, file);
      if (i < num_layers - 1) {
        fwrite(lay[i].neu[j].out_weights, sizeof(float), num_neurons[i + 1],
               file);
      }
    }
  }
  fclose(file);
  printf("NN saved to %s\n", filename);
  return SAVE_NEURAL_SUCCESS;
}

int load_neural_network(const char *filename) {
  printf("Attempting to load neural network from %s\n", filename);
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    fprintf(stderr, "Failed to open file %s for reading\n", filename);
    return ERR_LOAD_NEURAL;
  }
  printf("File opened successfully\n");

  int loaded_num_layers;
  fread(&loaded_num_layers, sizeof(int), 1, file);
  printf("Loaded number of layers: %d\n", loaded_num_layers);

  if (loaded_num_layers != num_layers) {
    fprintf(stderr,
            "Number of layers in file (%d) does not match expected (%d)\n",
            loaded_num_layers, num_layers);
    fprintf(stderr, "Mismatch in nn layers\n");
    fclose(file);
    return ERR_LOAD_NEURAL;
  }
  int loaded_num_neurons[num_layers];
  fread(loaded_num_neurons, sizeof(int), loaded_num_layers, file);
  for (int i = 0; i < num_layers; i++) {
    if (loaded_num_neurons[i] != num_neurons[i]) {
      fprintf(stderr, "Mismatch in num neurons\n");
      fclose(file);
      return ERR_LOAD_NEURAL;
    }
  }

  for (int i = 0; i < num_layers; i++) {
    for (int j = 0; j < num_neurons[i]; j++) {
      fread(&lay[i].neu[j].bias, sizeof(float), 1, file);
      if (i < num_layers - 1) {
        fread(lay[i].neu[j].out_weights, sizeof(float), num_neurons[i + 1],
              file);
      }
    }
  }
  fclose(file);
  printf("NN loaded from %s\n", filename);
  return LOAD_NEURAL_SUCCESS;
}

void load_validation_data(float ***images, int8_t **labels,
                          int *num_validation) {
  const char *validation_images_path = "imgs/t10k-images.idx3-ubyte";
  const char *validation_labels_path = "labels/t10k-labels.idx1-ubyte";

  load_images(validation_images_path, images, num_validation);
  int num_labels;
  load_labels(validation_labels_path, labels, &num_labels);

  if (*num_validation != num_labels) {
    fprintf(stderr,
            "Number of validation images (%d) does not match number of labels "
            "(%d)\n",
            *num_validation, num_labels);
    *num_validation = 0;
  }
}

float test_model(float **images, int8_t *labels, int num_validation) {
  int correct = 0;
  for (int i = 0; i < num_validation; i++) {
    feed_input(i, images);
    forward_prop();

    int predicted = 0;
    float max_prob = lay[num_layers - 1].neu[0].actv;
    for (int j = 1; j < num_neurons[num_layers - 1]; j++) {
      if (lay[num_layers - 1].neu[j].actv > max_prob) {
        max_prob = lay[num_layers - 1].neu[j].actv;
        predicted = j;
      }
    }
    if (predicted == labels[i]) {
      correct++;
    }
  }
  float accuracy = (float)correct / num_validation;
  printf("Accuracy: %f\n", accuracy);
  return accuracy;
}
#ifndef DRAW_PROGRAM

int main(void) {
  float **images = NULL, **validation_images = NULL;
  int8_t *labels = NULL, *validation_labels = NULL;
  int num_images, num_labels, num_validation;
  const char *model_filename = "trained_model.bin";
  const char *backup_filename = "trained_model.bin.old";
  bool model_loaded = false;
  char user_choice;

  load_images("imgs/train-images.idx3-ubyte", &images, &num_images);
  load_labels("labels/train-labels.idx1-ubyte", &labels, &num_labels);

  if (num_images != num_labels) {
    printf("Number of images and labels do not match\n");
    goto cleanup;
  }

  num_training_ex = num_images;

  if (create_architecture() != CREATION_ARCH_SUCCESS) {
    printf("Failed to create architecture\n");
    goto cleanup;
  }

  cost = (float *)malloc(num_training_ex * sizeof(float));
  if (cost == NULL) {
    fprintf(stderr, "Failed to allocate memory for cost\n");
    goto cleanup;
  }

  desired_outputs = malloc(num_labels * sizeof(float *));
  if (desired_outputs == NULL) {
    fprintf(stderr, "Failed to allocate memory for desired outputs\n");
    goto cleanup;
  }
  for (int i = 0; i < num_labels; i++) {
    desired_outputs[i] = calloc(10, sizeof(float));
    if (desired_outputs[i] == NULL) {
      fprintf(stderr, "Failed to allocate memory for desired output %d\n", i);
      goto cleanup;
    }
    desired_outputs[i][labels[i]] = 1.0;
  }

  if (access(model_filename, F_OK) != -1) {
    printf("Existing model found. Do you want to:\n");
    printf("(L)oad and use the existing model\n");
    printf("(C)ontinue training the existing model\n");
    printf("(T)rain a new model from scratch\n");
    printf("(V)alidate the model\n");
    printf("Enter your choice (L/C/T/V): ");
    scanf(" %c", &user_choice);

    switch (user_choice) {
    case 'L':
    case 'l':
      if (load_neural_network(model_filename) == LOAD_NEURAL_SUCCESS) {
        model_loaded = true;
        printf("Model loaded successfully.\n");
      } else {
        printf("Failed to load model.\n");
      }
      break;
    case 'T':
    case 't':
      if (backup_file(model_filename, backup_filename) == 0) {
        if (initialize_weights() != SUCCESS_INIT_WEIGHTS) {
          printf("Failed to initialize weights\n");
          goto cleanup;
        }
        if (train_nn(images, model_filename) != TRAIN_NEURAL_SUCCESS) {
          printf("Failed to train the model\n");
          goto cleanup;
        }
        model_loaded = true;
      } else {
        printf("Failed to backup existing model, training aborted.\n");
        goto cleanup;
      }
      break;
    case 'C':
    case 'c':
      if (!model_loaded) {
        if (load_neural_network(model_filename) != LOAD_NEURAL_SUCCESS) {
          printf("Failed to load model. Please load a model first.\n");
          break;
        }
      }
      if (train_nn(images, model_filename) != TRAIN_NEURAL_SUCCESS) {
        printf("Failed to train the model\n");
        goto cleanup;
      }
      break;
    case 'V':
    case 'v':
      if (!model_loaded) {
        if (load_neural_network(model_filename) != LOAD_NEURAL_SUCCESS) {
          printf("Failed to load model. Please load a model first.\n");
          break;
        }
        model_loaded = true;
      }
      load_validation_data(&validation_images, &validation_labels,
                           &num_validation);
      if (num_validation > 0) {
        int temp_num_training_ex = num_training_ex;
        num_training_ex = num_validation;
        float accuracy =
            test_model(validation_images, validation_labels, num_validation);
        printf("Validation accuracy: %.2f%%\n", accuracy * 100);
        num_training_ex = temp_num_training_ex;
      } else {
        printf("Failed to load validation data or no validation data "
               "available.\n");
      }
      break;
    default:
      printf("Invalid choice. Please try again.\n");
    }
  }

cleanup:
  if (images) {
    for (int i = 0; i < num_images; i++) {
      free(images[i]);
    }
    free(images);
  }
  free(labels);
  free(cost);
  if (desired_outputs) {
    for (int i = 0; i < num_labels; i++) {
      free(desired_outputs[i]);
    }
    free(desired_outputs);
  }
  if (lay) {
    for (int i = 0; i < num_layers; i++) {
      for (int j = 0; j < num_neurons[i]; j++) {
        free(lay[i].neu[j].out_weights);
        free(lay[i].neu[j].dw);
      }
      free(lay[i].neu);
    }
    free(lay);
  }
  return 0;
}

#endif
