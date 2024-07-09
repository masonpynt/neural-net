#ifndef DRAW_H
#define DRAW_H

#include <ncurses.h>
#include <stdbool.h>

#define GRID_SIZE 28
#define DRAW_SIZE 56

void init_grid(void);
void draw_grid(WINDOW *win);
float *convert_to_input(void);
void draw_point(int x, int y);
void test_network(float *input);
void center_digit(void);

extern float grid[GRID_SIZE][GRID_SIZE];

#endif
