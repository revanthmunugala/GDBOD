CC = nvcc
CFLAGS = -Xcompiler -fopenmp -O3 -std=c++14
COMPUTE_CAPABILITY = 75
COMPUTE_FLAGS = -arch=compute_$(COMPUTE_CAPABILITY) -code=sm_$(COMPUTE_CAPABILITY)

SOURCES = hySortOD_encoded.cu hySortOD_lib.cu
OBJECTS = hySortOD_encoded.o hySortOD_lib.o
EXECUTABLE = main

all: $(EXECUTABLE)

hySortOD_encoded.o: hySortOD_encoded.cu
	$(CC) $(CFLAGS) $(COMPUTE_FLAGS) -c $< -o $@

hySortOD_lib.o: hySortOD_lib.cu
	$(CC) $(CFLAGS) $(COMPUTE_FLAGS) -c $< -o $@


$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(COMPUTE_FLAGS) $^ -o $@


clean:
	rm -f $(OBJECTS) $(EXECUTABLE)


