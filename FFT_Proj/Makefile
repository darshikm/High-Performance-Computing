.DEFAULT_GOAL := all
CC = gcc
CFLAGS = -lm
COPTFLAGS = -Wall -O3 -g
COMP = -fopenmp

TARGET = sequential_comp OpenMP_fft

all: $(TARGET)

sequential_comp: sequential_comp.o
	$(CC) $(COPTFLAGS) -o $@ $^ $(CFLAGS)

OpenMP_fft: OpenMP_fft.o
	$(CC) $(COMP) $(COPTFLAGS) -o $@ $^ $(CFLAGS)

%.o: %.cc
	$(CC) $(COMP) $(COPTFLAGS) -o $@ $(CFLAGS) -c $<

clean:
	rm -f core *.o *~ sequential_comp OpenMP_fft
#eof
