#include "draw.h"
#include "backprop.h"
#include "layer.h"
#include "neuron.h"
#include <locale.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

char grid[GRID_SIZE][GRID_SIZE];

void init_grid() {
  for (int y = 0; y < GRID_SIZE; y++) {
    for (int x = 0; x < GRID_SIZE; x++) {
      grid[y][x] = ' ';
    }
  }
}

void draw_grid(WINDOW *win) {
  for (int y = 0; y < GRID_SIZE; y++) {
    for (int x = 0; x < GRID_SIZE; x++) {
      mvwaddch(win, y, x * 2, grid[y][x]);
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
  for (int i = 0; i < GRID_SIZE; i++) {
    for (int j = 0; j < GRID_SIZE; j++) {
      input[i * GRID_SIZE + j] = (grid[i][j] == '#') ? 1.0f : 0.0f;
    }
  }
  return input;
}

void draw_point(WINDOW *win, int x, int y) {
  if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
    grid[y][x] = '#';
    mvwaddch(win, y, x * 2, '#');
    mvwaddch(win, y, x * 2 + 1, ' ');
    wrefresh(win);
  }
  refresh();
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
  for (int y = 0; y < GRID_SIZE; y++) {
    for (int x = 0; x < GRID_SIZE; x++) {
      if (grid[y][x] == '#') {
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
  char temp_grid[GRID_SIZE][GRID_SIZE] = {{' '}};

  for (int y = min_y; y <= max_y; y++) {
    for (int x = min_x; x <= max_x; x++) {
      if (grid[y][x] == '#') {
        int new_y = new_min_y + (y - min_y);
        int new_x = new_min_x + (x - min_x);
        temp_grid[new_y][new_x] = '#';
      }
    }
  }

  memcpy(grid, temp_grid, sizeof(grid));
}

void smooth_digit() {
  char temp[GRID_SIZE][GRID_SIZE] = {{0}};
  for (int y = 1; y < GRID_SIZE - 1; y++) {
    for (int x = 1; x < GRID_SIZE - 1; x++) {
      int cnt = 0;
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          if (grid[y + dy][x + dx] == '#')
            cnt++;
        }
      }
      temp[y][x] = (cnt >= 1) ? '#' : ' ';
    }
  }
  memcpy(grid, temp, sizeof(grid));
}

// Temp function to test the centering of my digits;

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

  init_grid();
  draw_grid(win);

  mvprintw(GRID_SIZE + 2, 1,
           "Draw using mouse. Press 'c' to clear the grid. Press 'Enter' to test & clear. 'q' to quit.");
  refresh();

  while ((ch = getch()) != 'q') {
    if (ch == KEY_MOUSE) {
      if (getmouse(&event) == OK) {
        if (event.bstate & BUTTON1_PRESSED) {
          is_mouse_down = true;
          mvprintw(GRID_SIZE + 5, 1, "Mouse button down detected at x=%d, y=%d",
                   event.x,
                   event.y);
        } else if (event.bstate & BUTTON1_RELEASED) {
          is_mouse_down = false;
        }
        if (is_mouse_down && (event.bstate & REPORT_MOUSE_POSITION)) {
          int x = (event.x - 1) / 2;
          int y = event.y - 1;
          draw_point(win, x, y);
        }
      }
    } else if (ch == '\n') {
      center_digit();
      smooth_digit();
      redraw_grid(win);
      refresh();
      sleep(1);
      float *nn_input = convert_to_input();
      if (nn_input != NULL) {
        test_network(nn_input);
        free(nn_input);
      }
      sleep(2);
      init_grid();
      draw_grid(win);
      mvprintw(GRID_SIZE + 3, 1, "                        ");
      mvprintw(GRID_SIZE + 4, 1, "                        ");
      mvprintw(GRID_SIZE + 5, 1, "                        ");
      refresh();
    } else if (ch == 'c') {
      init_grid();
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
