CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++14

SOURCES = hySortOD_encoded.cpp hySortOD_lib.cpp
OBJECTS = hySortOD_encoded.o hySortOD_lib.o
EXECUTABLE = main

all: $(EXECUTABLE)

hySortOD_encoded.o: hySortOD_encoded.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

hySortOD_lib.o: hySortOD_lib.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)


