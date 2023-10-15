#define DIMS 1
#define BLOCKS 4
#define THREADS 32
#define CUDASIZE 10

extern void caller();
extern void wrapper();
extern int checkResults();
