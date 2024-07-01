#include "neuron.h"
#include <stdlib.h>
#include <stdio.h>

neuron create_neuron(int num_out_weights) {
    neuron neu;

    neu.actv = 0.0;
    neu.out_weights = (float *) malloc(num_out_weights * sizeof(float));
    if (neu.out_weights == NULL) {
        fprintf(stderr, "Memory allocation failed for out_weights\n");
        exit(EXIT_FAILURE);
    }

    neu.bias = 0.0;
    neu.z = 0.0;

    neu.dactv = 0.0;
    neu.dw = (float *) malloc(num_out_weights * sizeof(float));
    if (neu.dw == NULL) {
        fprintf(stderr, "Memory allocation failed for dw\n");
        exit(EXIT_FAILURE);
    }

    neu.dbias = 0.0;
    neu.dz = 0.0;

    return neu;
}

