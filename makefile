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

#link.o: reduction.o
#	${NVCC} ${CUDAFLAGS} -dlink reduction.cu
#reduction_1.o: reduction_1.cu reduction_header.cuh
#	${NVCC} ${CUDAFLAGS} -std=c++11 -c reduction_1.cu
#reduction_2.o: reduction_2.cu reduction_header.cuh
#	${NVCC} ${CUDAFLAGS} -std=c++11 -c reduction_2.cu
#reduction_3.o: reduction_3.cu reduction_header.cuh
#	${NVCC} ${CUDAFLAGS} -std=c++11 -c reduction_3.cu
#reduction_4.o: reduction_4.cu reduction_header.cuh
#	${NVCC} ${CUDAFLAGS} -std=c++11 -c reduction_4.cu
#reduction_5.o: reduction_5.cu reduction_header.cuh
#	${NVCC} ${CUDAFLAGS} -std=c++11 -c reduction_5.cu
#reduction_6.o: reduction_6.cu reduction.cu reduction_header.cuh
#	${NVCC} ${CUDAFLAGS} -std=c++11 -c reduction.cu reduction_6.cu

clean:
	${RM} *.o main