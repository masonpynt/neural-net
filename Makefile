# Compiler and linker configurations
CC = gcc
CFLAGS = -g -Wall -Werror
LDFLAGS = -pthread -lpthread

# Executable name
EXEC = backprop

# Object files
OBJS = neural.o layer.o neuron.o

# Default target
all: $(EXEC)

# Linking all object files to create the executable
$(EXEC): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ -lm

# Compiling neural.c to neural.o
neural.o: neural.c
	$(CC) $(CFLAGS) -c $<

# Compiling layer.c to layer.o
layer.o: layer.c
	$(CC) $(CFLAGS) -c $<

# Compiling neuron.c to neuron.o
neuron.o: neuron.c
	$(CC) $(CFLAGS) -c $<

# Clean target to remove object files and the executable
clean:
	rm -f *.o $(EXEC)
