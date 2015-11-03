CC = g++
CFLAGS =
COPTFLAGS = -fopenmp -g -O3
LDFLAGS =

default:
	@echo "=================================================="
	@echo "To build your OpenMP code, use:"
	@echo "  make qsort-omp        # For Quicksort"
	@echo "  make mergesort-omp    # For Mergesort"
	@echo ""
	@echo "To clean this subdirectory (remove object files"
	@echo "and other junk), use:"
	@echo "  make clean"
	@echo "=================================================="

# Quicksort driver using OpenMP
qsort-omp: driver.o sort.o parallel-qsort.o
	$(CC) $(COPTFLAGS) -o $@ $^

# Mergesort driver using OpenMP
mergesort-omp: driver.o sort.o parallel-mergesort.o
	$(CC) $(COPTFLAGS) -o $@ $^

%.o: %.cc
	$(CC) $(CFLAGS) $(COPTFLAGS) -o $@ -c $<

clean:
	rm -f core *.o *~ qsort-omp mergesort-omp

# eof
