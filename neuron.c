#include "neuron.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

neuron create_neuron(int num_out_weights) {
    neuron neu;
    neu.actv = 0.0;
    neu.bias = 0.0;
    neu.z = 0.0;
    neu.dactv = 0.0;
    neu.dbias = 0.0;
    neu.dz = 0.0;

    if (num_out_weights > 0) {
        neu.out_weights = (float *)calloc(num_out_weights, sizeof(float));
        if (neu.out_weights == NULL) {
            fprintf(stderr, "Memory allocation failed for out_weights\n");
            exit(EXIT_FAILURE);
        }

        neu.dw = (float *)calloc(num_out_weights, sizeof(float));
        if (neu.dw == NULL) {
            fprintf(stderr, "Memory allocation failed for dw\n");
            free(neu.out_weights); 
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < num_out_weights; i++) {
            neu.out_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * sqrt(2.0f / num_out_weights);
        }
    } else {
        neu.out_weights = NULL;
        neu.dw = NULL;
    }

    return neu;
}
