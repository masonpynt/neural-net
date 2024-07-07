#ifndef BACKPROP_H
#define BACKPROP_H

#include "layer.h"
#include <stdint.h>

// Success and error codes
#define SUCCESS_INIT 0
#define ERR_INIT 1
#define SUCCESS_DINIT 0
#define ERR_DINIT 1
#define SUCCESS_INIT_WEIGHTS 0
#define ERR_INIT_WEIGHTS 1
#define SUCCESS_UPDATE_WEIGHTS 0
#define CREATION_ARCH_SUCCESS 0
#define ERR_CREATE_ARCHITECTURE 1
#define SAVE_NEURAL_SUCCESS 0
#define ERR_SAVE_NEURAL 1
#define LOAD_NEURAL_SUCCESS 0
#define ERR_LOAD_NEURAL 1
#define TRAIN_NEURAL_SUCCESS 0

extern layer *lay;
extern int num_layers;
extern int num_neurons[];
extern float alpha;
extern float *cost;
extern float full_cost;
extern float **input;
extern float **desired_outputs;
extern int num_training_ex;

int init(void);
int dinit(void);
int create_architecture(void);
int initialize_weights(void);
int initialize_dummy_weights(void);
void feed_input(int idx, float **images);
void forward_prop(void);
void compute_cost(int idx);
void back_prop(int p);
void update_weights(void);
void get_inputs(void);
void get_desired_outputs(void);
void test_nn(void);
int train_nn(float **images, const char *model_filename);
void clip_gradients(float threshold);
int save_nn(const char *filename);
int load_neural_network(const char *filename);
void load_images(const char *filename, float ***images, int *num_images);
void load_labels(const char *filename, int8_t **labels, int *num_labels);

#endif 
