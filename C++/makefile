NVCC=nvcc
CUDAFLAGS=-arch=sm_70
RM=/bin/rm -f

all: main

main: main.o caller.o reduction.o
	g++ main.o caller.o reduction.o -o main -L/usr/local/cuda/lib64 -lcuda -lcudart

main.o: main.cpp
	g++ -std=c++11 -c main.cpp

caller.o: caller.cpp
	g++ -std=c++11 -c caller.cpp

reduction.o: reduction.cu
	${NVCC} ${CUDAFLAGS} -c reduction.cu

clean:
	${RM} *.o main