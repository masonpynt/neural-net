#include "draw.h"
#include "layer.h"
#include "neural.h"
#include "neuron.h"
#include <locale.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

float grid[GRID_SIZE][GRID_SIZE];
int prev_x = -1, prev_y = -1;

void init_grid() {
  for (int y = 0; y < GRID_SIZE; y++) {
    for (int x = 0; x < GRID_SIZE; x++) {
      grid[y][x] = 0.0f;
    }
  }
}
void draw_line(float x0, float y0, float x1, float y1) {
  float dx = x1 - x0;
  float dy = y1 - y0;
  float gradient = (dx == 0) ? 1 : dy / dx;

  // Vert
  if (dx == 0) {
    for (int y = (int)fminf(y0, y1); y <= (int)fmaxf(y0, y1); y++) {
      if (y >= 0 && y < GRID_SIZE) {
        grid[y][(int)x0] = fminf(grid[y][(int)x0] + 1, 1);
      }
    }
    return;
  }

  if (x0 > x1) {
    float tmp = x0;
    x0 = x1;
    x1 = tmp;
    tmp = y0;
    y0 = y1;
    y1 = tmp;
  }
  for (int x = (int)x0; x <= (int)x1; x++) {
    if (x < 0 || x >= GRID_SIZE)
      continue;

    float y = y0 + gradient * (x - x0);
    int iy = (int)y;
    float fy = y - iy;

    if (iy >= 0 && iy < GRID_SIZE) {
      grid[iy][x] = fminf(grid[iy][x] + (1 - fy), 1);
    }
    if (iy + 1 >= 0 && iy + 1 < GRID_SIZE) {
      grid[iy + 1][x] = fminf(grid[iy + 1][x] + fy, 1);
    }
  }
}

void draw_point(int x, int y) {
  if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
    if (prev_x != -1 && prev_y != -1) {
      draw_line(prev_x, prev_y, x, y);
    } else {
      grid[y][x] = 1.0f;
    }
    prev_x = x;
    prev_y = y;
  }
}

void gaussian_blur() {
  float kernel[3][3] = {{1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
                        {2 / 16.0f, 4 / 16.0f, 2 / 16.0f},
                        {1 / 16.0f, 2 / 16.0f, 1 / 16.0f}};
  float temp_grid[GRID_SIZE][GRID_SIZE] = {0};
  for (int y = 1; y < GRID_SIZE - 1; y++) {
    for (int x = 1; x < GRID_SIZE - 1; x++) {
      float sum = 0;
      for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) { 
          sum += grid[y + ky][x + kx] * kernel[ky + 1][kx + 1];
        }
      }
      temp_grid[y][x] = sum;
    }
  }
  memcpy(grid, temp_grid, sizeof(grid));
}

void draw_grid(WINDOW *win) {
  for (int y = 0; y < GRID_SIZE; y++) {
    for (int x = 0; x < GRID_SIZE; x++) {
      float value = grid[y][x];
      char display_char = (value > 0.5f) ? '#' : ' ';
      mvwaddch(win, y, x * 2, display_char);
      mvwaddch(win, y, x * 2 + 1, ' ');
    }
  }
  wrefresh(win);
}

float *convert_to_input() {
  float *input = malloc(GRID_SIZE * GRID_SIZE * sizeof(float));
  if (input == NULL) {
    mvprintw(GRID_SIZE + 4, 1, "Memory allocation failed");
    refresh();
    return NULL;
  }

  float min_val = 1.0f, max_val = 0.0f;

  for (int i = 0; i < GRID_SIZE; i++) {
    for (int j = 0; j < GRID_SIZE; j++) {
      if (grid[i][j] < min_val)
        min_val = grid[i][j];
      if (grid[i][j] > max_val)
        max_val = grid[i][j];
    }
  }

  for (int i = 0; i < GRID_SIZE; i++) {
    for (int j = 0; j < GRID_SIZE; j++) {
      if (max_val > min_val) {
        input[i * GRID_SIZE + j] = (grid[i][j] - min_val) / (max_val - min_val);
      } else {
        input[i * GRID_SIZE + j] =
            grid[i][j]; 
      }
    }
  }
  return input;
}

void test_network(float *input) {
  for (int i = 0; i < num_neurons[0]; i++) {
    lay[0].neu[i].actv = input[i];
  }

  forward_prop();

  int predicted = 0;
  float max_prob = lay[num_layers - 1].neu[0].actv;
  float second_max_prob = 0.0f;
  int second_predicted = 0;
  for (int i = 1; i < num_neurons[num_layers - 1]; i++) {
    if (lay[num_layers - 1].neu[i].actv > max_prob) {
      second_max_prob = max_prob;
      second_predicted = predicted;
      max_prob = lay[num_layers - 1].neu[i].actv;
      predicted = i;
    } else if (lay[num_layers - 1].neu[i].actv > second_max_prob) {
      second_max_prob = lay[num_layers - 1].neu[i].actv;
      second_predicted = i;
    }
  }

  float confidence = max_prob - second_max_prob;

  mvprintw(GRID_SIZE + 3, 1, "Network predicts: %d (%.2f%% confident)",
           predicted, max_prob * 100);
  mvprintw(GRID_SIZE + 4, 1, "Second guess: %d (%.2f%% confident)",
           second_predicted, second_max_prob * 100);
  mvprintw(GRID_SIZE + 5, 1, "Confidence margin: %.2f%%", confidence * 100);

  mvprintw(GRID_SIZE + 7, 1, "All probabilities:");
  for (int i = 0; i < num_neurons[num_layers - 1]; i++) {
    mvprintw(GRID_SIZE + 8 + i, 1, "%d: %.2f%%", i,
             lay[num_layers - 1].neu[i].actv * 100);
  }
  refresh();
}

void center_digit() {
  int min_x = GRID_SIZE, max_x = 0, min_y = GRID_SIZE, max_y = 0;
  float threshold = 0.1f; 
  for (int y = 0; y < GRID_SIZE; y++) {
    for (int x = 0; x < GRID_SIZE; x++) {
      if (grid[y][x] > threshold) {
        min_x = (x < min_x) ? x : min_x;
        max_x = (x > max_x) ? x : max_x;
        min_y = (y < min_y) ? y : min_y;
        max_y = (y > max_y) ? y : max_y;
      }
    }
  }
  int width = max_x - min_x + 1;
  int height = max_y - min_y + 1;
  int new_min_x = (GRID_SIZE - width) / 2;
  int new_min_y = (GRID_SIZE - height) / 2;
  float temp_grid[GRID_SIZE][GRID_SIZE] = {{0}};

  for (int y = min_y; y <= max_y; y++) {
    for (int x = min_x; x <= max_x; x++) {
      int new_y = new_min_y + (y - min_y);
      int new_x = new_min_x + (x - min_x);
      temp_grid[new_y][new_x] = grid[y][x];
    }
  }

  memcpy(grid, temp_grid, sizeof(grid));
}


void redraw_grid(WINDOW *win) {
  for (int y = 0; y < GRID_SIZE; y++) {
    for (int x = 0; x < GRID_SIZE; x++) {
      mvwaddch(win, y, x * 2, grid[y][x]);
      mvwaddch(win, y, x * 2 + 1, ' ');
    }
  }
  wrefresh(win);
}

int main(int argc, char *argv[]) {

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <model file>\n", argv[0]);
    return 1;
  }
  const char *model_file = argv[1];

  if (create_architecture() != CREATION_ARCH_SUCCESS) {
    fprintf(stderr, "Failed to create neural network architecture\n");
    return 1;
  }

  if (load_neural_network(model_file) != LOAD_NEURAL_SUCCESS) {
    fprintf(stderr, "Failed to load neural network from file %s\n", model_file);
    return 1;
  }
  setlocale(LC_ALL, "");
  WINDOW *win;
  int ch;
  MEVENT event;
  bool is_mouse_down = false;

  initscr();
  cbreak();
  noecho();
  keypad(stdscr, TRUE);

  mousemask(ALL_MOUSE_EVENTS | REPORT_MOUSE_POSITION, NULL);
  printf("\033[?1003h\n");

  win = newwin(GRID_SIZE, GRID_SIZE * 2, 1, 1);
  box(win, 0, 0);

  memset(grid, 0, sizeof(grid));
  draw_grid(win);

  mvprintw(GRID_SIZE + 2, 1,
           "Draw using mouse. Press 'c' to clear the grid. Press 'Enter' to "
           "test & clear. 'q' to quit.");
  refresh();

  while ((ch = getch()) != 'q') {
    if (ch == KEY_MOUSE) {
      if (getmouse(&event) == OK) {
        if (event.bstate & BUTTON1_PRESSED) {
          is_mouse_down = true;
          mvprintw(GRID_SIZE + 5, 1, "Mouse button down detected at x=%d, y=%d",
                   event.x, event.y);
        } else if (event.bstate & BUTTON1_RELEASED) {
          is_mouse_down = false;
        }
        if (is_mouse_down && (event.bstate & REPORT_MOUSE_POSITION)) {
          int x = (event.x - 1) / 2;
          int y = event.y - 1;
          draw_point(x, y);
          draw_grid(win);
        }
      }
    } else if (ch == '\n') {
      gaussian_blur();
      center_digit();
      redraw_grid(win);
      refresh();
      sleep(1);
      float *nn_input = convert_to_input();
      if (nn_input != NULL) {
        test_network(nn_input);
        free(nn_input);
      }
      sleep(2);
      memset(grid, 0, sizeof(grid));
      draw_grid(win);
      prev_x = -1;
      prev_y = -1;
      is_mouse_down = false;
      mvprintw(GRID_SIZE + 3, 1, "                        ");
      mvprintw(GRID_SIZE + 4, 1, "                        ");
      mvprintw(GRID_SIZE + 5, 1, "                        ");
      refresh();
    } else if (ch == 'c') {
      memset(grid, 0, sizeof(grid));
      prev_x = -1;
      prev_y = -1;
      redraw_grid(win);
      refresh();
    } else {
      mvprintw(GRID_SIZE + 7, 1, "Key pressed: %d        ", ch);
      refresh();
    }
  }
  printf("\033[?1003l\n");
  endwin();
  return 0;
}
