#ifndef LAYER_H
#define LAYER_H

typedef struct layer_t {
  int num_neu;
  struct neuron_t *neu;
} layer;

layer create_layer(int num_neurons);

#endif
