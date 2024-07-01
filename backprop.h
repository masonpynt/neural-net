#ifndef BACKPROP_H
#define BACKPROP_H

#include "layer.h"

#define SUCCESS_INIT 0
#define ERR_INIT 1

#define SUCCESS_DINIT 0
#define ERR_DINIT 1

#define SUCCESS_INIT_WEIGHTS 0
#define ERR_INIT_WEIGHTS 1

#define SUCCESS_UPDATE_WEIGHTS 0

#define CREATION_ARCH_SUCCESS 0
#define ERR_CREATE_ARCHITECTURE 1



int init(void);
int dinit(void);

int create_architecture(void);
int initialize_weights(void);
void feed_input(int idx, float **images);
void forward_prop(void);
void compute_cost(int i);
void back_prop(int p);
void update_weights(void);
void get_inputs(void);
void get_desired_outputs(void);
void test_nn(void);
void compute_cost(int idx);
void train_nn(float **images);
void forward_prop(void);
void back_prop(int p);
void update_weights(void);
int initialize_weights(void);
int initialize_dummy_weights(void);


#endif
