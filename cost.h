#ifndef COST_H
#define COST_H
#include <float.h>

typedef struct {
  float best_cost;
  int epochs_no_improve;
  int best_epoch;
} EarlyStoppingState;


int check_early_stopping(float current_cost, int current_epoch, float min_delta, int patience);

extern EarlyStoppingState early_stopping_state;

#endif
