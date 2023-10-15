NVCC=nvcc
CUDAFLAGS=-arch=sm_70
RM=/bin/rm -f

all: main

main: main.o caller.o reduction.o
	g++ main.o caller.o reduction.o -o main -L/usr/local/cuda/lib64 -lcuda -lcudart

main.o: main.cpp reduction_header.cuh
	g++ -std=c++11 -c main.cpp

wrapperCaller.o: caller.cpp reduction_header.cuh
	g++ -std=c++11 -c caller.cpp

reduction.o: reduction.cu reduction_header.cuh
	${NVCC} ${CUDAFLAGS} -std=c++11 -c reduction.cu

clean:
	${RM} *.o main
