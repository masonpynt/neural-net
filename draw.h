#ifndef DRAW_H
#define DRAW_H

#include <ncurses.h>
#include <stdbool.h>

#define GRID_SIZE 28

void init_grid(void);
void draw_grid(WINDOW *win);
float *convert_to_input(void);
void draw_point(WINDOW *win, int x, int y);
void test_network(float *input);
void center_digit(void);

extern char grid[GRID_SIZE][GRID_SIZE];

#endif
