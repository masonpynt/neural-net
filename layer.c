#include "layer.h"
#include "neuron.h"
#include <stdlib.h>
#include <stdio.h> 

layer create_layer(int number_of_neurons) {
    layer lay;
    lay.num_neu = number_of_neurons; 
    lay.neu = (struct neuron_t *) malloc(number_of_neurons * sizeof(struct neuron_t));
    if (lay.neu == NULL) {
        fprintf(stderr, "Memory allocation failed for neurons in layer\n");
        exit(EXIT_FAILURE);
    }
    return lay;
}

