#ifndef NEURON_H
#define NEURON_H

typedef struct neuron_t {
    float actv;
    float *out_weights;
    float bias;
    float z;

    float dactv;
    float *dw;
    float dbias;
    float dz;
} neuron;

neuron create_neuron(int num_out_weights);

#endif // NEURON_H

