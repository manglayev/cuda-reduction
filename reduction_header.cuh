#define DIMS 1
#define BLOCKS 4
#define THREADS 32
#define CUDASIZE 10
//VARIANT is on of the 1-7 variants of CUDA reduction
#define VARIANT 5

extern void caller();
extern void wrapper();
extern int checkResults(int *a);