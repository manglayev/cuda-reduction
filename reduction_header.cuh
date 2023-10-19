#include <iostream>
#include <utility>
#include <type_traits>
#include <stdio.h>
#include <stdlib.h>

#define DIMS 1
#define BLOCKS 4
#define THREADS 32
#define CUDASIZE 32
//VARIANT is one of the 1-7 variants of CUDA reduction
#define VARIANT 1

extern void caller();
extern void wrapper();
extern int checkResults(int *a);
extern int* initArray();