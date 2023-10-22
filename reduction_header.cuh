#include <iostream>
#include <utility>
#include <type_traits>
#include <stdio.h>
#include <stdlib.h>

#define DIMS 1
#define BLOCKS 8
#define THREADS 512
#define CUDASIZE 4096
//VARIANT is one of the 1-7 variants of CUDA reduction
#define VARIANT 2

extern void caller();
extern void wrapper();
extern int checkResults(int *a);
extern int* initArray();