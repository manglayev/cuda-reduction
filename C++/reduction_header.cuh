#include <iostream>
#include <utility>
#include <type_traits>
#include <stdio.h>
#include <stdlib.h>

/*
The size of array must be equal to the multiplication of the number of threads to the number of blocks
CUDASIZE = THREADS x BLOCKS
*/

#define DIMS 1
//#define BLOCKS 32
#define BLOCKS 1024
#define THREADS 1024
//#define CUDASIZE 32768
#define CUDASIZE 1048576
//VARIANT is one of the 1-7 variants of CUDA reduction
#define VARIANT 5

extern void caller();
extern void wrapper();
extern int checkResults(int *a);
extern int* initArray();
