#include <stdio.h>
#include "cost.h"


EarlyStoppingState early_stopping_state = {FLT_MAX, 0, 0};

int check_early_stopping(float current_cost, int current_epoch, float min_delta, int patience) {
    if (current_cost < early_stopping_state.best_cost - min_delta) {
        early_stopping_state.best_cost = current_cost;
        early_stopping_state.epochs_no_improve = 0;
        early_stopping_state.best_epoch = current_epoch;
    } else {
        early_stopping_state.epochs_no_improve++;
    }

    if (early_stopping_state.epochs_no_improve >= patience) {
        printf("Early stopping triggered. No improvement for %d epochs.\n", patience);
        printf("Best cost was %.6f at epoch %d\n", early_stopping_state.best_cost, early_stopping_state.best_epoch);
        return 1; 
    }

    return 0; 
}
