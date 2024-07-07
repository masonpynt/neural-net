# Compiler and linker configurations
CC = gcc
CFLAGS = -g -Wall -Werror -fsanitize=address -fno-omit-frame-pointer
LDFLAGS = -pthread -lpthread -fsanitize=address -lm -lncurses

# Executable names
NEURAL_EXEC = neural
DRAW_EXEC = draw

# Object files
COMMON_OBJS = layer.o neuron.o cost.o
NEURAL_OBJS = neural.o $(COMMON_OBJS)
DRAW_OBJS = draw.o neural_draw.o $(COMMON_OBJS)

# Default target
all: $(NEURAL_EXEC) $(DRAW_EXEC)

# Linking object files to create the neural network executable
$(NEURAL_EXEC): $(NEURAL_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

# Linking object files to create the drawing interface executable
$(DRAW_EXEC): $(DRAW_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

# Generic rule for compiling .c to .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Specific rule for draw.o
draw.o: draw.c
	$(CC) $(CFLAGS) -c $< -o $@

# Specific rules for neural.o when compiling for neural program
neural.o: neural.c
	$(CC) $(CFLAGS) -c $< -o $@

# Specific rule for neural.o when compiling for draw program
neural_draw.o: neural.c
	$(CC) $(CFLAGS) -DDRAW_PROGRAM -c $< -o $@

# Clean target to remove object files and the executables
clean:
	rm -f *.o $(NEURAL_EXEC) $(DRAW_EXEC)

# Phony targets
.PHONY: all clean

